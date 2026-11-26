# /// script
# requires-python = ">=3.12"
# dependencies = ["tinker>=0.40.1", "tinker-cookbook>=0.7.1"]
# ///
"""Train and sample a tuned checker on Tinker (Thinking Machines).

Standalone: it does not import Hinbox. It reads the chat files that
``just citecheck export`` / ``prompts`` / ``merge-data`` write, and writes
replies that ``just citecheck score --replies`` / ``merge-score`` read.
Needs ``TINKER_API_KEY`` in the environment. Run with
``uv run --script scripts/tinker_citecheck.py <step> ...``.

Steps:

- ``smoke --data <train.jsonl> --prompts <test_prompts.jsonl> --out <dir>``:
  the half-day test. Trains 15 small steps, samples 3 prompts with the new
  weights (prints each reply and its token count, to see whether the model
  answers briefly without long reasoning), saves the weights in external
  format, downloads them, lists the tensors and tries the cookbook's PEFT
  adapter build. It answers the three unknowns before any real training:
  can a tuned Inkling leave Tinker, in what form, and does it stop
  reasoning. Downloads go through Tinker's own ``tinker checkpoint
  download``. Free-tier storage holds at most 100 checkpoints or 100 GB, so
  delete smoke checkpoints afterwards (``tinker checkpoint delete``).
- ``train --data <train.jsonl> [--tune <tune.jsonl>] --out <dir>``: one or
  more epochs of LoRA training, loss on the tune file every
  ``--eval-every`` steps, a checkpoint at the end (path printed and saved in
  ``<dir>/checkpoints.jsonl``).
- ``sample --checkpoint <tinker://...> --prompts <file> --out <replies.jsonl>``:
  one reply per prompt at temperature 0, with seconds and token counts.
  ``--checkpoint base`` samples the untuned model.
- ``distill --data <train.jsonl> --out <distilled.jsonl>``: self-distillation
  (rejection sampling). The untuned model (or ``--checkpoint``) answers each
  example's prompt ``--samples`` times at ``--temperature``, reasoning
  included; the first answer whose verdicts all match the example's own
  labels is kept, with its reasoning, as that example's training reply. An
  example no answer matches is left out. Up to ``--workers`` prompts are
  sampled at once (``--loose``: partial and not supported count as one
  answer, as in the test score); resumable (``<out>.log.jsonl`` lists every example tried).
  Training on the result teaches the model to keep reasoning the way it
  already does, only to land on the labelled verdict: the plain export has
  no reasoning in its replies, so a model trained on it stops thinking.
- ``download --checkpoint <tinker://.../external_weights/...|weights/...> --out <dir>``:
  ``tinker checkpoint download`` into ``<dir>``, then the tensors are listed.

Every example is rendered with reasoning effort ``--effort`` (default 0.0,
the lowest the Inkling renderer accepts) both in training and in sampling,
so the model learns to answer at that setting.

Files here hold article text: keep ``--out`` under ``data/``.
"""

import argparse
import json
import os
import random
import sys
import time

FREE_MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"


def _read(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _renderer(model: str):
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    base = model.split(":", 1)[0]
    tokenizer = get_tokenizer(base)
    name = model_info.get_recommended_renderer_name(base)
    return renderers.get_renderer(name, tokenizer), tokenizer


def _datums(rows: list, renderer, effort: float, max_length: int) -> list:
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    out = []
    for row in rows:
        pairs = renderer.build_supervised_examples(
            row["messages"], TrainOnWhat.LAST_ASSISTANT_MESSAGE, effort=effort
        )
        for model_input, weights in pairs:
            if model_input.length > max_length:
                continue
            out.append(
                datum_from_model_input_weights(
                    model_input, weights, max_length, reduction="mean"
                )
            )
    return out


def _mean_nll(result, batch) -> float:
    from tinker_cookbook.supervised.common import compute_mean_nll

    logprobs = [x["logprobs"] for x in result.loss_fn_outputs]
    weights = [d.loss_fn_inputs["weights"] for d in batch]
    return float(compute_mean_nll(logprobs, weights))


def _resolve(x):
    return x.result() if hasattr(x, "result") else x


def train_loop(args, rows: list, tune: list, steps: int = 0):
    """LoRA training over ``rows``; returns the training client."""
    import tinker
    from tinker_cookbook import hyperparam_utils

    renderer, _ = _renderer(args.model)
    data = _datums(rows, renderer, args.effort, args.max_length)
    tune_data = _datums(tune, renderer, args.effort, args.max_length)[
        : args.batch_size * 4
    ]
    print(f"{len(data)} training sequences, {len(tune_data)} tune sequences")
    service = tinker.ServiceClient()
    client = service.create_lora_training_client(
        base_model=args.model, rank=args.rank
    ).result()
    lr = args.lr or hyperparam_utils.get_lr(args.model)
    per_epoch = max(1, -(-len(data) // args.batch_size))  # a partial last batch counts
    total = steps or per_epoch * args.epochs
    rng = random.Random(0)
    order: list = []
    for step in range(total):
        if not order:
            order = list(range(len(data)))
            rng.shuffle(order)
        idx, order = order[: args.batch_size], order[args.batch_size :]
        batch = [data[i] for i in idx]
        current = lr * max(0.0, 1.0 - step / total)
        fwd = client.forward_backward(batch, loss_fn="cross_entropy")
        opt = client.optim_step(
            tinker.AdamParams(learning_rate=current, beta1=0.9, beta2=0.95, eps=1e-8)
        )
        result = fwd.result()
        opt.result()
        line = {"step": step, "lr": current, "train_nll": _mean_nll(result, batch)}
        if tune_data and (step % args.eval_every == 0 or step == total - 1):
            ev = client.forward(tune_data, loss_fn="cross_entropy").result()
            line["tune_nll"] = _mean_nll(ev, tune_data)
        print(json.dumps(line))
        _append(os.path.join(args.out, "train_log.jsonl"), line)
    return client


def _parts(message: dict) -> tuple:
    """(thinking, text) of a parsed reply message."""
    content = message.get("content")
    if not isinstance(content, list):
        return "", content or ""
    thinking = "".join(
        p.get("thinking", "") for p in content if p.get("type") == "thinking"
    )
    text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return thinking, text


LEVELS = ("supported", "partial", "not_supported", "unclear")


def _level(value) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def reply_levels(text: str) -> dict:
    """Claim number -> support level from a JSON ``{"items": [...]}`` reply
    (code fences and words around the JSON are allowed); empty when the
    reply can't be read. A claim whose level is missing or not one of
    ``LEVELS`` is left out, so the reply can't match its labels."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    items = data.get("items") if isinstance(data, dict) else None
    levels = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        level = _level(item.get("support_level"))
        if level not in LEVELS:  # missing or made up: the claim has no verdict
            continue
        try:
            levels[int(item.get("claim_number"))] = level
        except (TypeError, ValueError):
            continue
    return levels


def _yes_no(levels: dict) -> dict:
    return {
        k: "yes" if v == "supported" else v if v == "unclear" else "no"
        for k, v in levels.items()
    }


def matches_gold(text: str, gold: dict, loose: bool = False) -> bool:
    """True when the reply gives every labelled claim its label, and no
    other claim numbers. ``loose`` counts partial and not supported as one
    answer, as the test score does."""
    got = reply_levels(text)
    return bool(gold) and (_yes_no(got) == _yes_no(gold) if loose else got == gold)


def row_key(messages: list) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[
        :16
    ]


def distilled_row(prompt: list, thinking: str, text: str, meta: dict) -> dict:
    """A training row whose reply holds the model's reasoning and answer."""
    parts = ([{"type": "thinking", "thinking": thinking}] if thinking else []) + [
        {"type": "text", "text": text}
    ]
    return {
        "messages": prompt + [{"role": "assistant", "content": parts}],
        "distill": meta,
    }


def _weighted(renderer, messages: list, effort: float) -> int:
    from tinker_cookbook.renderers import TrainOnWhat

    pairs = renderer.build_supervised_examples(
        messages, TrainOnWhat.LAST_ASSISTANT_MESSAGE, effort=effort
    )
    return int(sum(float(w.sum()) for _, w in pairs))


def cmd_distill(args) -> None:
    import tinker

    renderer, _ = _renderer(args.model)
    service = tinker.ServiceClient()
    if args.checkpoint in (None, "", "base"):
        sampler = _resolve(service.create_sampling_client(base_model=args.model))
    else:
        sampler = _resolve(service.create_sampling_client(model_path=args.checkpoint))
    params = tinker.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=renderer.get_stop_sequences(),
    )
    log_path = args.out + ".log.jsonl"
    tried = {r["key"] for r in _read(log_path)} if os.path.exists(log_path) else set()
    todo = []
    for row in _read(args.data):
        prompt, last = row["messages"][:-1], row["messages"][-1]
        if row_key(prompt) in tried:
            continue
        gold = reply_levels(last["content"] if isinstance(last["content"], str) else "")
        todo.append((prompt, gold, row.get("item_ids")))
    print(f"distilling {len(todo)} examples ({len(tried)} already in {log_path})")
    checked = os.path.exists(args.out)
    kept = total = 0
    pending: list = []
    queue = list(todo)
    while queue or pending:
        while queue and len(pending) < args.workers:
            prompt, gold, ids = queue.pop(0)
            model_input = renderer.build_generation_prompt(prompt, effort=args.effort)
            future = sampler.sample(
                prompt=model_input, num_samples=args.samples, sampling_params=params
            )
            pending.append((prompt, gold, ids, future, time.perf_counter()))
        prompt, gold, ids, future, start = pending.pop(0)
        resp = _resolve(future)
        total += 1
        log = {"key": row_key(prompt), "item_ids": ids, "gold": gold, "kept": None}
        lengths, unreadable = [], 0
        for i, seq in enumerate(resp.sequences):
            message, _ = renderer.parse_response(seq.tokens)
            thinking, text = _parts(message)
            lengths.append(len(seq.tokens))
            unreadable += not reply_levels(text)  # not JSON, or cut off
            if log["kept"] is None and matches_gold(text, gold, args.loose):
                log["kept"] = i
                meta = {"sample": i, "output_tokens": len(seq.tokens), "item_ids": ids}
                out = distilled_row(prompt, thinking, text, meta)
                if thinking and not checked:  # reasoning must carry weight
                    with_thinking = _weighted(renderer, out["messages"], args.effort)
                    plain = distilled_row(prompt, "", text, meta)["messages"]
                    without = _weighted(renderer, plain, args.effort)
                    print(
                        f"weighted tokens: {with_thinking} with reasoning, {without} without"
                    )
                    if with_thinking <= without:
                        sys.exit("the renderer gives the reasoning no weight; stopping")
                    checked = True
                _append(args.out, out)
                kept += 1
        log["output_tokens"], log["unreadable"] = lengths, unreadable
        log["seconds"] = round(time.perf_counter() - start, 1)
        _append(log_path, log)
        print(
            f"{total}/{len(todo)}: {'kept sample ' + str(log['kept']) if log['kept'] is not None else 'none matched'}"
            f" (labels {sorted(set(gold.values()))}; kept {kept} of {total})"
        )
    _distill_summary(log_path)


def _distill_summary(log_path: str) -> None:
    """Share of examples kept, per label they hold."""
    counts: dict = {}
    for r in _read(log_path):
        for level in set(r["gold"].values()) or {"(none)"}:
            k, n = counts.get(level, (0, 0))
            counts[level] = (k + (r["kept"] is not None), n + 1)
    for level, (k, n) in sorted(counts.items()):
        print(f"{level}: kept {k} of {n} examples holding it")
    rows = _read(log_path)
    bad = sum(r.get("unreadable", 0) for r in rows)
    answers = sum(len(r.get("output_tokens", [])) for r in rows)
    print(f"{bad} of {answers} answers had no readable JSON verdicts (text or cut off)")


def _sample_rows(sampler, renderer, prompts: list, args, out_path: str = "") -> list:
    import tinker

    params = tinker.SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0, stop=renderer.get_stop_sequences()
    )
    replies = []
    for row in prompts:
        prompt = renderer.build_generation_prompt(row["messages"], effort=args.effort)
        start = time.perf_counter()
        resp = sampler.sample(
            prompt=prompt, num_samples=1, sampling_params=params
        ).result()
        tokens = resp.sequences[0].tokens
        message, _ = renderer.parse_response(tokens)
        thinking, content = _parts(message)
        out = {
            "item_ids": row["item_ids"],
            "reply": content,
            "thinking": thinking,
            "output_tokens": len(tokens),
            "prompt_tokens": prompt.length,
            "seconds": round(time.perf_counter() - start, 2),
        }
        replies.append(out)
        if out_path:
            _append(out_path, out)
    return replies


def cmd_smoke(args) -> None:
    rows = _read(args.data)[: args.batch_size * 15]
    client = train_loop(args, rows, [], steps=15)
    renderer, _ = _renderer(args.model)
    sampler = _resolve(client.save_weights_and_get_sampling_client(name="smoke"))
    for r in _sample_rows(sampler, renderer, _read(args.prompts)[:3], args):
        print(
            f"--- {r['output_tokens']} output tokens, {r['seconds']} s\n{r['reply'][:1500]}"
        )
    saved = client.save_weights_external("smoke-external").result()
    print(f"external weights: {saved.path}")
    target = _download(saved.path, os.path.join(args.out, "smoke-weights"))
    _list_tensors(target)
    try:  # a PEFT folder that transformers/vLLM/llama.cpp converters read
        from tinker_cookbook import weights

        weights.build_lora_adapter(
            base_model=args.model.split(":", 1)[0],
            adapter_path=target,
            output_path=os.path.join(args.out, "smoke-peft"),
        )
        print("PEFT adapter built: the cookbook can export Inkling adapters")
    except Exception as exc:  # the cookbook lists no Inkling support; report it
        print(f"PEFT adapter build failed: {type(exc).__name__}: {exc}")


def _download(tinker_path: str, parent: str) -> str:
    """Download a checkpoint into ``parent``; the folder it landed in.

    Uses Tinker's own ``tinker checkpoint download`` (the documented way:
    ``save_weights_external()`` then this command; a Thinking Machines
    engineer confirmed on 25 Nov that Inkling 2 Mini LoRAs download this way,
    not yet on the ``:free`` ID word for word). It makes a folder named after
    the checkpoint inside ``parent``. Falls back to the cookbook's
    ``weights.download()`` if the command fails. Files ending ``.age`` are
    encrypted and need ``age -d`` first.
    """
    import subprocess

    os.makedirs(parent, exist_ok=True)
    before = set(os.listdir(parent))
    cmd = [
        sys.executable,
        "-m",
        "tinker.cli",
        "checkpoint",
        "download",
        tinker_path,
        "--output",
        parent,
        "--force",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    print(done.stdout[-2000:], done.stderr[-2000:], sep="\n")
    if done.returncode == 0:
        new = sorted(set(os.listdir(parent)) - before) or sorted(os.listdir(parent))
        folder = os.path.join(parent, new[-1]) if new else parent
        print(f"downloaded with `tinker checkpoint download` into {folder}")
    else:
        from tinker_cookbook import weights

        print("`tinker checkpoint download` failed; trying the cookbook's download")
        folder = os.path.join(parent, "cookbook")
        weights.download(tinker_path=tinker_path, output_dir=folder)
    encrypted = [f for _, _, fs in os.walk(folder) for f in fs if f.endswith(".age")]
    if encrypted:
        print(f"{len(encrypted)} files are age-encrypted: decrypt with `age -d` first")
    return folder


def _list_tensors(folder: str) -> None:
    from safetensors import safe_open

    for root, _, files in os.walk(folder):
        for name in sorted(files):
            path = os.path.join(root, name)
            print(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB)")
            if name.endswith(".json"):
                print(open(path).read()[:2000])
            if name.endswith(".safetensors"):
                with safe_open(path, "np") as f:
                    keys = list(f.keys())
                    print(f"  {len(keys)} tensors, e.g.")
                    for k in keys[:12]:
                        print(f"   {k} {f.get_slice(k).get_shape()}")


def cmd_train(args) -> None:
    client = train_loop(args, _read(args.data), _read(args.tune) if args.tune else [])
    state = client.save_state(args.name).result()
    sampler = client.save_weights_for_sampler(args.name).result()
    row = {
        "name": args.name,
        "state": state.path,
        "sampler": sampler.path,
        "model": args.model,
    }
    _append(os.path.join(args.out, "checkpoints.jsonl"), row)
    print(json.dumps(row))


def cmd_sample(args) -> None:
    import tinker

    renderer, _ = _renderer(args.model)
    service = tinker.ServiceClient()
    if args.checkpoint == "base":
        sampler = _resolve(service.create_sampling_client(base_model=args.model))
    else:
        sampler = _resolve(service.create_sampling_client(model_path=args.checkpoint))
    done = set()
    if os.path.exists(args.out):
        done = {tuple(r["item_ids"]) for r in _read(args.out)}
    prompts = [p for p in _read(args.prompts) if tuple(p["item_ids"]) not in done]
    print(f"sampling {len(prompts)} prompts ({len(done)} already in {args.out})")
    replies = _sample_rows(sampler, renderer, prompts, args, out_path=args.out)
    if replies:
        tokens = sum(r["output_tokens"] for r in replies) / len(replies)
        print(f"mean output tokens per prompt: {tokens:.0f}")


def cmd_download(args) -> None:
    """Download a checkpoint: ``tinker://.../external_weights/...`` from
    ``save_weights_external()``, or ``.../weights/...`` from ``save_state()``
    (Tinker exports that one through a temporary external checkpoint)."""
    _list_tensors(_download(args.checkpoint, args.out))


def main() -> None:
    if not os.getenv("TINKER_API_KEY"):
        sys.exit("set TINKER_API_KEY first")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["smoke", "train", "sample", "distill", "download"])
    p.add_argument("--model", default=FREE_MODEL)
    p.add_argument("--data")
    p.add_argument("--tune")
    p.add_argument("--prompts")
    p.add_argument("--checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="citecheck")
    p.add_argument("--effort", type=float, default=0.0)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0, help="0 = the cookbook's value")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--samples", type=int, default=4, help="distill: answers per example"
    )
    p.add_argument("--temperature", type=float, default=0.7, help="distill only")
    p.add_argument("--workers", type=int, default=8, help="distill: prompts at once")
    p.add_argument(
        "--loose", action="store_true", help="distill: partial = not supported"
    )
    args = p.parse_args()
    os.makedirs(
        args.out
        if args.step not in ("sample", "distill")
        else os.path.dirname(os.path.abspath(args.out)),
        exist_ok=True,
    )
    {
        "smoke": cmd_smoke,
        "train": cmd_train,
        "sample": cmd_sample,
        "distill": cmd_distill,
        "download": cmd_download,
    }[args.step](args)


if __name__ == "__main__":
    main()

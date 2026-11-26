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
  reasoning.
- ``train --data <train.jsonl> [--tune <tune.jsonl>] --out <dir>``: one or
  more epochs of LoRA training, loss on the tune file every
  ``--eval-every`` steps, a checkpoint at the end (path printed and saved in
  ``<dir>/checkpoints.jsonl``).
- ``sample --checkpoint <tinker://...> --prompts <file> --out <replies.jsonl>``:
  one reply per prompt at temperature 0, with seconds and token counts.
  ``--checkpoint base`` samples the untuned model.
- ``download --checkpoint <tinker://.../weights/...> --out <dir>``.

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
        content = message.get("content")
        if isinstance(content, list):  # thinking + text parts
            content = "".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        out = {
            "item_ids": row["item_ids"],
            "reply": content or "",
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
    from tinker_cookbook import weights

    target = os.path.join(args.out, "smoke-weights")
    weights.download(tinker_path=saved.path, output_dir=target)
    _list_tensors(target)
    try:
        weights.build_lora_adapter(
            base_model=args.model.split(":", 1)[0],
            adapter_path=target,
            output_path=os.path.join(args.out, "smoke-peft"),
        )
        print("PEFT adapter built: the cookbook can export Inkling adapters")
    except Exception as exc:  # the cookbook lists no Inkling support; report it
        print(f"PEFT adapter build failed: {type(exc).__name__}: {exc}")


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
    from tinker_cookbook import weights

    weights.download(tinker_path=args.checkpoint, output_dir=args.out)
    _list_tensors(args.out)


def main() -> None:
    if not os.getenv("TINKER_API_KEY"):
        sys.exit("set TINKER_API_KEY first")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["smoke", "train", "sample", "download"])
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
    args = p.parse_args()
    os.makedirs(
        args.out
        if args.step != "sample"
        else os.path.dirname(os.path.abspath(args.out)),
        exist_ok=True,
    )
    {
        "smoke": cmd_smoke,
        "train": cmd_train,
        "sample": cmd_sample,
        "download": cmd_download,
    }[args.step](args)


if __name__ == "__main__":
    main()

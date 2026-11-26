"""The training run's report card: one self-contained HTML page (inline CSS
and SVG, no scripts or outside files) that ZenML shows as the
``training_report`` artifact.

It holds numbers only: the two error rates with their 95% ranges, the same
rates for any baselines given, the loss curve, the rates per kind of claim
and the training settings. No claim, article or reply text goes in, since a
ZenML server may be shared or in the cloud.
"""

import html
from typing import Any, Dict, List, Optional, Sequence, Tuple

METRICS = (
    ("wrong_passed", "Wrong claims passed", "wrong claims called supported"),
    ("good_doubted", "Good claims doubted", "good claims not called supported"),
)

STYLE = """
.rc{color-scheme:light;--bg:#fff;--ink:#191919;--ink2:#666;--rule:#e6e6e6;
--track:#f1f1f1;--tuned:#257de0;--base:#9a9a9a;--train:#9bbfe9;--tune:#f47230;
--good:#087a1a;--good-bg:#e3f4e6;--bad:#b42d2e;--bad-bg:#fbe6e6;
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;
color:var(--ink);background:var(--bg);max-width:760px;margin:0 auto;padding:24px 20px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .rc{
color-scheme:dark;--bg:#191919;--ink:#efefef;--ink2:#9c9c9c;--rule:#333;
--track:#262626;--tuned:#338ced;--base:#7a7a7a;--train:#3f6a99;--tune:#e76624;--good:#3fcf55;
--good-bg:#12331a;--bad:#f07a7b;--bad-bg:#3d1a1a}}
:root[data-theme="dark"] .rc{color-scheme:dark;--bg:#191919;--ink:#efefef;
--ink2:#9c9c9c;--rule:#333;--track:#262626;--tuned:#338ced;--base:#7a7a7a;
--train:#3f6a99;--tune:#e76624;--good:#3fcf55;
--good-bg:#12331a;--bad:#f07a7b;--bad-bg:#3d1a1a}
.rc *{box-sizing:border-box}
.rc .eyebrow{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2)}
.rc h1{font-size:22px;margin:2px 0 4px;font-weight:600}
.rc h2{font-size:13px;font-weight:600;margin:28px 0 10px;color:var(--ink2);
text-transform:uppercase;letter-spacing:.05em}
.rc .meta{color:var(--ink2);margin:0}
.rc .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:20px}
.rc .tile{border:1px solid var(--rule);border-radius:10px;padding:14px 16px}
.rc .tile .label{color:var(--ink2);font-size:13px}
.rc .tile .big{font-size:34px;font-weight:600;line-height:1.1;margin:4px 0;font-variant-numeric:tabular-nums}
.rc .tile .sub{color:var(--ink2);font-size:12.5px;font-variant-numeric:tabular-nums}
.rc .note{color:var(--ink2);font-size:12.5px;margin:6px 0 0}
.rc svg{display:block;width:100%;height:auto;overflow:visible}
.rc svg text{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
.rc svg text.name{fill:var(--ink);font-size:12px}
.rc .legend{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin-bottom:4px}
.rc .legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px}
.rc table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
.rc th,.rc td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--rule)}
.rc th{font-weight:600;font-size:12px;color:var(--ink2)}
.rc td.n,.rc th.n{text-align:right}
.rc dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:0;font-size:12.5px}
.rc .tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.rc .tag.good{color:var(--good);background:var(--good-bg)}
.rc .tag.bad{color:var(--bad);background:var(--bad-bg)}
.rc dt{color:var(--ink2)} .rc dd{margin:0;font-variant-numeric:tabular-nums}
"""


def _e(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _pct(share: Optional[float]) -> str:
    return "n/a" if share is None else f"{100 * share:.1f}%"


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% range for ``k`` out of ``n`` (the same formula as ``citecheck score``)."""
    if not n:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def baseline_rates(
    baselines: Dict[str, Dict[str, Any]], report: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Each baseline as ``{name, wrong_passed, good_doubted}`` rates. A value is
    a count on the same test set (``36``) or ``[count, of]``."""
    rows = []
    for name, got in baselines.items():
        row: Dict[str, Any] = {"name": name}
        for key, _, _ in METRICS:
            value = got.get(key)
            if value is None:
                continue
            k, n = (
                value
                if isinstance(value, (list, tuple))
                else (value, report[key]["of"])
            )
            lo, hi = wilson(int(k), int(n))
            row[key] = {
                "count": int(k),
                "of": int(n),
                "share": k / n if n else None,
                "low": lo,
                "high": hi,
            }
        rows.append(row)
    return rows


def _tile(
    rate: Dict[str, Any], label: str, what: str, best: Optional[Dict[str, Any]]
) -> str:
    sub = f"{rate['count']} of {rate['of']} {what}"
    rng = f"95% range {_pct(rate['low'])} to {_pct(rate['high'])}"
    vs = ""
    if best is not None:
        vs = (
            f'<div class="sub">best baseline: {_e(best["name"])}, '
            f"{_pct(best['rate']['share'])} ({best['rate']['count']})</div>"
        )
    return (
        f'<div class="tile"><div class="label">{_e(label)}</div>'
        f'<div class="big">{_pct(rate["share"])}</div>'
        f'<div class="sub">{_e(sub)}</div><div class="sub">{rng}</div>{vs}</div>'
    )


def _ranges(rows: Sequence[Tuple[str, Dict[str, Any], bool]]) -> str:
    """One row per checker: a dot at its share and a line over its 95% range,
    on one 0 to max scale so the rows compare."""
    # a rate with no labelled claims (share None) is drawn as a name only
    top = max((r["high"] for _, r, _ in rows if r.get("share") is not None), default=0)
    top = min(1.0, max(0.1, round(top + 0.05, 1)))
    left, right, row_h, width = 190, 120, 26, 720
    plot = width - left - right
    x = lambda s: left + plot * s / top  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {row_h * len(rows) + 22}" role="img">']
    for t in range(0, 6):
        s = top * t / 5
        out.append(
            f'<line x1="{x(s):.1f}" x2="{x(s):.1f}" y1="0" y2="{row_h * len(rows)}" '
            'stroke="var(--rule)" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x(s):.1f}" y="{row_h * len(rows) + 15}" '
            f'text-anchor="middle">{100 * s:.0f}%</text>'
        )
    for i, (name, r, tuned) in enumerate(rows):
        y = i * row_h + row_h / 2
        colour = "var(--tuned)" if tuned else "var(--base)"
        weight = ' style="font-weight:600"' if tuned else ""
        out.append(
            f'<text class="name" x="0" y="{y + 4:.1f}"{weight}>{_e(name)}</text>'
        )
        if r.get("share") is None:
            continue
        tip = (
            f"{name}: {_pct(r['share'])} ({r['count']} of {r['of']}), "
            f"95% range {_pct(r['low'])} to {_pct(r['high'])}"
        )
        out.append(
            f"<g><title>{_e(tip)}</title>"
            f'<rect x="{left}" y="{y - row_h / 2:.1f}" width="{plot}" height="{row_h}" fill="transparent"/>'
            f'<line x1="{x(r["low"]):.1f}" x2="{x(r["high"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="2" stroke-linecap="round" opacity=".55"/>'
            f'<circle cx="{x(r["share"]):.1f}" cy="{y:.1f}" r="5" fill="{colour}" '
            f'stroke="var(--bg)" stroke-width="2"/></g>'
        )
        out.append(
            f'<text x="{width}" y="{y + 4:.1f}" text-anchor="end"{weight}>'
            f"{_pct(r['share'])}  ({r['count']}/{r['of']})</text>"
        )
    out.append("</svg>")
    return "".join(out)


def _loss_chart(log: List[Dict[str, Any]]) -> str:
    train = [(r["step"], r["train_nll"]) for r in log if r.get("train_nll") is not None]
    tune = [(r["step"], r["tune_nll"]) for r in log if r.get("tune_nll") is not None]
    points = train + tune
    if not points:
        return '<p class="note">No training log was recorded.</p>'
    width, height, left, bottom, top_pad = 720, 220, 40, 22, 8
    x0 = min(s for s, _ in points)
    x1 = max(max(s for s, _ in points), x0 + 1)
    y1 = max(v for _, v in points) * 1.08
    x = lambda s: left + (width - left - 8) * (s - x0) / (x1 - x0)  # noqa: E731
    y = lambda v: top_pad + (height - bottom - top_pad) * (1 - v / y1)  # noqa: E731
    out = [
        '<div class="legend"><span><i style="background:var(--train)"></i>train loss</span>'
        '<span><i style="background:var(--tune)"></i>tune loss (held out)</span></div>',
        f'<svg viewBox="0 0 {width} {height}" role="img">',
    ]
    for t in range(0, 5):
        v = y1 * t / 4
        out.append(
            f'<line x1="{left}" x2="{width - 8}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
            'stroke="var(--rule)" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>'
        )
    for s in (x0, (x0 + x1) // 2, x1):
        out.append(
            f'<text x="{x(s):.1f}" y="{height - 6}" text-anchor="middle">step {s}</text>'
        )

    def line(series: List[Tuple[int, float]], colour: str, w: float) -> str:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x(s):.1f},{y(v):.1f}"
            for i, (s, v) in enumerate(series)
        )
        return (
            f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="{w}" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    if train:
        out.append(line(train, "var(--train)", 1.5))
    if tune:
        out.append(line(tune, "var(--tune)", 2))
        for s, v in tune:
            out.append(
                f"<g><title>step {s}: tune loss {v:.3f}</title>"
                f'<circle cx="{x(s):.1f}" cy="{y(v):.1f}" r="8" fill="transparent"/>'
                f'<circle cx="{x(s):.1f}" cy="{y(v):.1f}" r="4" fill="var(--tune)" '
                'stroke="var(--bg)" stroke-width="2"/></g>'
            )
    out.append("</svg>")
    return "".join(out)


def _strata(report: Dict[str, Any]) -> str:
    rows = []
    for name, t in report.get("by_stratum", {}).items():
        cells = "".join(
            f'<td class="n">{_pct(t[k]["share"])} <span style="color:var(--ink2)">'
            f"({t[k]['count']}/{t[k]['of']})</span></td>"
            for k, _, _ in METRICS
        )
        rows.append(f"<tr><td>{_e(name)}</td>{cells}</tr>")
    head = "".join(f'<th class="n">{_e(label)}</th>' for _, label, _ in METRICS)
    return f"<table><tr><th>kind of claim</th>{head}</tr>{''.join(rows)}</table>"


def render(
    report: Dict[str, Any],
    loss_log: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    baselines: Optional[Dict[str, Dict[str, Any]]] = None,
    sampling: Optional[Dict[str, Any]] = None,
) -> str:
    """The report card as one HTML string."""
    name = cfg.get("name") or "citation checker"
    model = str(cfg.get("model", "")).split(":")[0]
    others = baseline_rates(baselines or {}, report)
    tiles = []
    for key, label, what in METRICS:
        scored = [
            {"name": b["name"], "rate": b[key]}
            for b in others
            if b.get(key, {}).get("share") is not None
        ]
        best = min(scored, key=lambda b: b["rate"]["share"]) if scored else None
        tiles.append(_tile(report[key], label, what, best))
    compare = []
    for key, label, _ in METRICS if others else ():
        rows = [(f"{name} (tuned)", report[key], True)]
        rows += [(b["name"], b[key], False) for b in others if key in b]
        compare.append(
            f'<p class="note" style="margin:14px 0 4px">{_e(label)}</p>{_ranges(rows)}'
        )
    steps = len(loss_log)
    tune = [r["tune_nll"] for r in loss_log if r.get("tune_nll") is not None]
    settings = {
        "model": cfg.get("model"),
        "LoRA rank": cfg.get("rank"),
        "learning rate": cfg.get("lr") or "cookbook default",
        "epochs": cfg.get("epochs"),
        "batch size": cfg.get("batch_size"),
        "thinking effort": cfg.get("effort"),
        "steps run": steps,
        "final tune loss": f"{tune[-1]:.3f}" if tune else "n/a",
        "best tune loss": f"{min(tune):.3f}" if tune else "n/a",
        "failed replies": report.get("errors", 0),
    }
    if sampling:
        settings["mean output tokens"] = f"{sampling.get('mean_output_tokens', 0):.0f}"
        settings["mean seconds per prompt"] = f"{sampling.get('mean_seconds', 0):.1f}"
    dl = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in settings.items())
    meta = (
        f"{_e(model)} · {steps} steps · {report['labelled']} hand-labelled test claims"
        f" ({report.get('unsure', 0)} unsure left out)"
    )
    return (
        f'<style>{STYLE}</style><div class="rc">'
        f'<div class="eyebrow">Citation checker · fine-tune report</div>'
        f'<h1>{_e(name)}</h1><p class="meta">{meta}</p>'
        f'<div class="tiles">{"".join(tiles)}</div>'
        '<p class="note">Lower is better for both. Claims come from articles the training never saw.</p>'
        + (f"<h2>Against the baselines</h2>{''.join(compare)}" if others else "")
        + f"<h2>Loss</h2>{_loss_chart(loss_log)}"
        f"<h2>By kind of claim</h2>{_strata(report)}"
        f"<h2>Run</h2><dl>{dl}</dl></div>"
    )


# ---- the Kitaru comparison ------------------------------------------------

# evaluation name -> (label, lower is better, how to print a mean)
KITARU_MEASURES = {
    "wrong_passed": ("Wrong claims passed", True, "pct"),
    "good_doubted": ("Good claims doubted", True, "pct"),
    "agrees": ("Agrees with the answer key", False, "pct"),
    "answered": ("Gave a readable verdict", False, "pct"),
    "seconds": ("Seconds per claim", True, "s"),
    "input_tokens": ("Input tokens per claim", True, "n"),
    "output_tokens": ("Output tokens per claim", True, "n"),
    "cost_usd": ("Cost per claim", True, "usd"),
}


def _fmt(value: Optional[float], kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return _pct(value)
    if kind == "s":
        return f"{value:.1f} s"
    if kind == "usd":
        return f"${value:.4f}"
    return f"{value:,.0f}"


def _pair_bars(base: Optional[float], tuned: Optional[float]) -> str:
    """Two thin bars on one scale: grey the baseline, blue the tuned model."""
    top = max(v for v in (base, tuned, 1e-12) if v is not None)
    bars = []
    for i, (v, colour) in enumerate(((base, "var(--base)"), (tuned, "var(--tuned)"))):
        w = 0 if v is None else max(2.0, 160 * v / top)
        bars.append(
            f'<rect x="0" y="{2 + i * 9}" width="{w:.1f}" height="6" rx="3" '
            f'fill="{colour}"/>'
        )
    return f'<svg viewBox="0 0 160 19" style="width:160px">{"".join(bars)}</svg>'


def render_kitaru(out: Dict[str, Any], experiment_name: str) -> str:
    """The Kitaru experiment as one HTML page: per evaluation, the baseline
    sessions' mean beside the tuned model's replay. Numbers and ids only."""
    order = list(KITARU_MEASURES)
    table = sorted(
        out.get("table") or [],
        key=lambda r: order.index(r["name"]) if r["name"] in order else len(order),
    )
    rows = []
    for r in table:
        label, lower, kind = KITARU_MEASURES.get(r["name"], (r["name"], True, "n"))
        base, tuned = r.get("baseline_mean"), r.get("result_mean")
        verdict = ""
        if base is not None and tuned is not None and base != tuned:
            better = (tuned < base) == lower
            # colour plus a mark and a word, so it never rests on colour alone
            verdict = (
                '<span class="tag good">✓ better</span>'
                if better
                else '<span class="tag bad">✕ worse</span>'
            )
        rows.append(
            f"<tr><td>{_e(label)}</td>"
            f'<td class="n">{_fmt(base, kind)}</td>'
            f'<td class="n" style="font-weight:600">{_fmt(tuned, kind)}</td>'
            f"<td>{_pair_bars(base, tuned)}</td>"
            f"<td>{verdict}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5">No evaluation results came back.</td></tr>')
    ids = {
        "experiment": experiment_name,
        "experiment id": out.get("experiment_id"),
        "run": out.get("run_id"),
        "status": out.get("status"),
        "sessions replayed": out.get("sessions"),
        "minutes": out.get("minutes"),
        "model override": out.get("model"),
    }
    dl = "".join(
        f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in ids.items() if v is not None
    )
    return (
        f'<style>{STYLE}</style><div class="rc">'
        '<div class="eyebrow">Citation checker · Kitaru comparison</div>'
        "<h1>Tuned checkpoint against the baseline</h1>"
        '<p class="meta">Every labelled test session replayed with the tuned '
        "checkpoint and scored by the same evaluator as the recorded baseline.</p>"
        '<div class="legend" style="margin-top:18px">'
        '<span><i style="background:var(--base)"></i>baseline sessions</span>'
        '<span><i style="background:var(--tuned)"></i>tuned replay</span></div>'
        '<table><tr><th>measure</th><th class="n">baseline</th>'
        '<th class="n">tuned</th><th></th><th></th></tr>'
        f"{''.join(rows)}</table>"
        f"<h2>Run</h2><dl>{dl}</dl></div>"
    )

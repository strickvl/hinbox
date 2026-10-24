"""Shared evaluation scoreboard: score any model on hand-labelled gold sets.

Run ``just eval --help`` for usage. Each run scores one task (relevance or
merge) with one model, writes per-item predictions, and appends one summary
row to ``{output_dir}/eval/scoreboard.jsonl`` so runs stay comparable.
"""

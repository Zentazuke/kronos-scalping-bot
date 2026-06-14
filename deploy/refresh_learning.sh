#!/usr/bin/env bash
# refresh_learning.sh — nightly learning refresh for the dashboard.
#
# 1) Label observations that have had time to resolve (replayed vs mainnet).
# 2) Walk-forward every model/dataset into walkforward.json, which the
#    dashboard reads on its next poll (the "Walk-forward verdict" panel).
#
# Every step is best-effort: a single failing run (e.g. <100 clean trades, or
# xgboost not installed) must never block the others. Triggered by
# kronos-learn.timer; safe to run by hand any time — it never touches the bot.
set -uo pipefail
cd /home/kronos/kronos-scalping-bot || exit 1
PY=/home/kronos/kronos-scalping-bot/.venv/bin/python

echo "=== refresh_learning $(date -u +%FT%TZ) ==="

# 1) Label resolved observations (mainnet candles, clean of phantom fills).
"$PY" label_observations.py --db observations.db \
  || echo "labeler: non-zero exit (fine if observations.db has no resolvable rows yet)"

# 2) Walk-forward each model x dataset. The learner refuses (exit 1) until it
#    has >=100 decided trades; --json only writes when a run actually produces
#    folds, so the panel simply keeps the last good verdict until then.
"$PY" learner.py walkforward --db journal.db      --model logistic --json walkforward.json || true
"$PY" learner.py walkforward --db journal.db      --model xgb      --json walkforward.json || true
"$PY" learner.py walkforward --db observations.db --model logistic --json walkforward.json || true
"$PY" learner.py walkforward --db observations.db --model xgb      --json walkforward.json || true

echo "=== refresh_learning done $(date -u +%FT%TZ) ==="

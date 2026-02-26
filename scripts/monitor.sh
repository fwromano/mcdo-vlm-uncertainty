#!/usr/bin/env bash
# monitor.sh — Live Phase 1 progress monitor for tmux
# Parses the experiment log and shows a compact status dashboard.
set -uo pipefail

LOGFILE="${1:-/private/tmp/claude-501/-Users-agc-Documents-GitHub-mcdo-vlm-uncertainty/tasks/b4e7f63.output}"

if [[ ! -f "$LOGFILE" ]]; then
  echo "Log not found: $LOGFILE"
  exit 1
fi

while true; do
  clear
  echo "╔══════════════════════════════════════════════════╗"
  echo "║       MC-Dropout Phase 1 — Live Monitor         ║"
  echo "╚══════════════════════════════════════════════════╝"
  echo ""

  # Current experiment & model
  LAST_EXP=$(grep -oE '\[Exp[0-9b]+\]' "$LOGFILE" | tail -1 | tr -d '[]')
  LAST_MODEL=$(grep -oE 'Loading model: \S+' "$LOGFILE" | tail -1 | sed 's/Loading model: //')
  LAST_TRIAL=$(grep -oE 'trial [0-9]+/[0-9]+' "$LOGFILE" | tail -1)
  LAST_T=$(grep -oE 'T=[0-9]+' "$LOGFILE" | tail -1)

  # Count completed trials per experiment
  EXP0_DONE=$(grep -c '^\[Exp0\].*trial.*complete\|Exp0.*saved\|Saving partial' "$LOGFILE" 2>/dev/null || echo 0)

  # Elapsed time (from file mod time)
  if [[ "$(uname)" == "Darwin" ]]; then
    START=$(stat -f %m "$LOGFILE" 2>/dev/null || echo 0)
  else
    START=$(stat -c %Y "$LOGFILE" 2>/dev/null || echo 0)
  fi
  NOW=$(date +%s)
  ELAPSED=$(( NOW - START ))
  # Use file creation time approximation — just show wall clock
  MINS=$(( ELAPSED / 60 ))
  SECS=$(( ELAPSED % 60 ))

  echo "  Experiment:  ${LAST_EXP:-starting...}"
  echo "  Model:       ${LAST_MODEL:-loading...}"
  echo "  Progress:    ${LAST_T:-...} | ${LAST_TRIAL:-...}"
  echo ""

  # Count completed experiment sections
  N_EXP0=$(grep -c 'Exp0.*summary\|exp0.*complete\|Exp0.*Saving' "$LOGFILE" 2>/dev/null || echo "?")
  N_EXP0B=$(grep -c 'Exp0b.*summary\|exp0b.*complete\|Exp0b.*Saving' "$LOGFILE" 2>/dev/null || echo "?")
  N_EXP4=$(grep -c 'Exp4.*summary\|exp4.*complete\|Exp4.*Saving' "$LOGFILE" 2>/dev/null || echo "?")
  N_EXP5=$(grep -c 'Exp5.*summary\|exp5.*complete\|Exp5.*Saving' "$LOGFILE" 2>/dev/null || echo "?")

  # Check if experiments are done
  exp_status() {
    local name="$1" pattern="$2"
    if grep -q "$pattern" "$LOGFILE" 2>/dev/null; then
      echo "DONE"
    elif grep -q "\[$name\]" "$LOGFILE" 2>/dev/null; then
      echo "running..."
    else
      echo "pending"
    fi
  }

  EXP0_S=$(exp_status "Exp0" "exp0_summary.json")
  EXP0B_S=$(exp_status "Exp0b" "exp0b_summary.json")
  EXP4_S=$(exp_status "Exp4" "exp4_subset_summary.json")
  EXP5_S=$(exp_status "Exp5" "exp5_subset.*summary.json")

  echo "  ┌────────────┬────────────┐"
  echo "  │ Experiment │ Status     │"
  echo "  ├────────────┼────────────┤"
  printf "  │ Exp 0      │ %-10s │\n" "$EXP0_S"
  printf "  │ Exp 0b     │ %-10s │\n" "$EXP0B_S"
  printf "  │ Exp 4      │ %-10s │\n" "$EXP4_S"
  printf "  │ Exp 5      │ %-10s │\n" "$EXP5_S"
  echo "  └────────────┴────────────┘"
  echo ""

  # Last 5 meaningful log lines (skip progress bars)
  echo "  Recent activity:"
  grep -v '^\s*$\||██\|it/s\]$\|it/s, ' "$LOGFILE" | grep -v 'UserWarning\|pin_memory\|QuickGELU' | tail -5 | while read -r line; do
    echo "    $line"
  done

  echo ""
  echo "  (refreshes every 15s — Ctrl-C or 'tmux kill-session -t mcdo-monitor' to stop)"
  sleep 15
done

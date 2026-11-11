#!/bin/zsh
# launchd wrapper: run the paper bot's catch-up scheduler; notify only when it did something.
cd "$(dirname "$0")"
out=$(~/quant_alpha/.venv/bin/python paper_bot.py auto 2>>paper/errors.log | tail -1)
[ -z "$out" ] && exit 0
echo "$(date '+%F %T') $out" >> paper/run.log
osascript -e "display notification \"${out//\"/}\" with title \"Paper bot\"" >/dev/null 2>&1

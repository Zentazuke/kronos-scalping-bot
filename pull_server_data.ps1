# pull_server_data.ps1 — copy the LIVE data from the server into this local Trading Bot folder.
# These files are gitignored (databases, caches, logs), so scp is the way to get them, not git.
#
# 1) set $SERVER to exactly what you SSH with (user@host or user@ip)
# 2) from this folder in PowerShell:   ./pull_server_data.ps1
#    (if blocked:  powershell -ExecutionPolicy Bypass -File .\pull_server_data.ps1)

$SERVER = "kronos@YOUR_SERVER_ADDRESS"      # <-- EDIT THIS (same as your ssh target)
$REMOTE = "kronos-scalping-bot"             # remote repo folder (relative to home)

$files = @(
  "tsm_live.db",        # live intraday trader (your closes + the bot's)
  "tsm_forward.db",     # the shadow forward-test scorecard
  "sleeve_forward.db",  # analyst-regime sleeves shadow test (added 2026-07-07)
  "sleeve.log",         # its cron log (commit/settle lines, MISSED warnings)
  "reconcile.log",      # daily live-vs-trial reconciliation verdicts
  "journal.db",         # Kronos trade journal
  "observations.db"     # harvested observations
)

Write-Host "Pulling live data from $SERVER ..."
foreach ($f in $files) {
  Write-Host "  $f"
  scp "${SERVER}:${REMOTE}/$f" .
}
Write-Host "  data_cache/ ..."
scp -r "${SERVER}:${REMOTE}/data_cache" .
# optional: the tail of the bot log for local inspection
# scp "${SERVER}:${REMOTE}/bot.log" .

Write-Host "Done. Pulled into $(Get-Location)."

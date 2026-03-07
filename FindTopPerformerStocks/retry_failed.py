# retry_failed.py
import json

# Load failed tickers from progress file
with open("output/progress.json") as f:
    progress = json.load(f)

# Reset failed tickers back to pending (removes them from "failed" set)
for stmt in progress:
    failed = list(progress[stmt]["failed"])
    progress[stmt]["failed"] = []   # clear failed
    # DON'T add to done — they'll be re-queued automatically

with open("output/progress.json", "w") as f:
    json.dump(progress, f, indent=2)

print(f"Reset failed tickers. Re-run fetch_quarterly_financials.py now.")
import json
from pathlib import Path
import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

EURO_COUNTRIES = {
    "England", "Germany", "Spain", "Italy", "France", "Portugal", "Netherlands",
    "Belgium", "Scotland", "Wales", "Northern Ireland"
}

def fetch_json(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_competitions():
    url = f"{BASE}/competitions.json"
    data = fetch_json(url)
    records = [c for c in data if c.get("country_name") in EURO_COUNTRIES]
    return pd.DataFrame(records)

def fetch_match_counts(df):
    rows = []
    for _, row in df.iterrows():
        comp_id = row["competition_id"]
        season_id = row["season_id"]
        matches_url = f"{BASE}/matches/{comp_id}/{season_id}.json"
        try:
            matches = fetch_json(matches_url)
            rows.append({
                "competition_id": comp_id,
                "season_id": season_id,
                "match_count": len(matches),
            })
        except requests.HTTPError:
            continue
    return pd.DataFrame(rows)

def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    comps = fetch_competitions()
    comps.to_csv(out_dir / "statsbomb_competitions.csv", index=False)
    counts = fetch_match_counts(comps)
    counts.to_csv(out_dir / "statsbomb_match_counts.csv", index=False)

if __name__ == "__main__":
    main()

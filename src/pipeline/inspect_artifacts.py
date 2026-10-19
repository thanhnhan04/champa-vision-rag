import pandas as pd

df = pd.read_csv("data/metadata_cleaned.csv")
print(f"Total rows: {len(df)}")
unique_arts = df.drop_duplicates(subset=["artifact_id"])
print(f"Unique artifacts: {len(unique_arts)}")

with open("data/artifacts_summary.txt", "w", encoding="utf-8") as f:
    for idx, (_, r) in enumerate(unique_arts.iterrows()):
        f.write(f"{idx+1}. [{r['artifact_id']}] Title: {r['title']} | Suggested: {r['name_vi_suggested']} | Era: {r['era']} | Mat: {r['material']} | Origin: {r['origin_site']} | Loc: {r['current_location']}\n")

print("Wrote summary to data/artifacts_summary.txt")


"""Step 1: Build an instruction-tuning dataset from Kaggle's netflix_titles.csv.
Usage: python 01_build_dataset.py netflix_titles.csv
Outputs: data/train.jsonl, data/val.jsonl, data/stats.json
"""
import sys, json, random, os
import pandas as pd

random.seed(42)
SYSTEM = "You are NetflixBot, a small assistant that answers questions about Netflix movies and TV shows."

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "netflix_titles.csv")
df = df.dropna(subset=["title", "description"]).drop_duplicates("title").reset_index(drop=True)
for c in ["director", "cast", "country", "rating", "duration", "listed_in", "release_year", "type"]:
    df[c] = df[c].fillna("").astype(str)

by_genre = {}
for _, r in df.iterrows():
    for g in [x.strip() for x in r["listed_in"].split(",") if x.strip()]:
        by_genre.setdefault((r["type"], g), []).append(r["title"])

def qa_for(r):
    t, out = r["title"], []
    out.append((f"What is {t} about?", r["description"]))
    out.append((f"Is {t} a movie or a TV show?", f"{t} is a {r['type']}." ))
    if r["director"]:
        out.append((f"Who directed {t}?", f"{t} was directed by {r['director']}."))
    if r["cast"]:
        out.append((f"Who stars in {t}?", f"The cast of {t} includes {r['cast']}."))
    if r["release_year"]:
        out.append((f"When was {t} released?", f"{t} was released in {r['release_year']}."))
    if r["rating"]:
        out.append((f"What is the maturity rating of {t}?", f"{t} is rated {r['rating']}."))
    if r["duration"]:
        out.append((f"How long is {t}?", f"{t} runs for {r['duration']}."))
    if r["country"]:
        out.append((f"Which country is {t} from?", f"{t} was produced in {r['country']}."))
    if r["listed_in"]:
        out.append((f"What genre is {t}?", f"{t} is listed under: {r['listed_in']}."))
    return out

titles = df["title"].tolist()
random.shuffle(titles)
cut = int(0.9 * len(titles))
val_titles = set(titles[cut:])

train, val = [], []
for _, r in df.iterrows():
    bucket = val if r["title"] in val_titles else train
    for q, a in qa_for(r):
        bucket.append({"system": SYSTEM, "question": q, "answer": a, "kind": q.split()[0]})

# Recommendation examples
for (typ, g), ts in by_genre.items():
    if len(ts) < 6:
        continue
    for _ in range(15):
        picks = random.sample(ts, 5)
        word = "movies" if typ == "Movie" else "TV shows"
        a = f"Here are some {g} {word} on Netflix: " + ", ".join(picks) + "."
        ex = {"system": SYSTEM, "question": f"Recommend some {g} {word}.", "answer": a, "kind": "Recommend"}
        (val if random.random() < 0.1 else train).append(ex)

# Out-of-scope refusals
for q in ["What is the weather today?", "Write me a Python script.", "Who won the cricket world cup?",
          "What is the capital of France?", "Tell me about Amazon Prime titles."]:
    train.append({"system": SYSTEM, "question": q,
                  "answer": "Sorry, I can only answer questions about Netflix movies and TV shows.", "kind": "OOS"})

random.shuffle(train); random.shuffle(val)
os.makedirs("data", exist_ok=True)
for name, rows in [("train", train), ("val", val)]:
    with open(f"data/{name}.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

stats = {"titles": len(df), "train_examples": len(train), "val_examples": len(val),
         "movies": int((df.type == "Movie").sum()), "tv_shows": int((df.type == "TV Show").sum()),
         "release_year_min": int(df.release_year.replace("", "0").astype(int).min()),
         "release_year_max": int(df.release_year.replace("", "0").astype(int).max())}
json.dump(stats, open("data/stats.json", "w"), indent=2)
print(stats)

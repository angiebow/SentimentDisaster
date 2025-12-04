import pandas as pd
import requests
import time
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

CSV_PATH = "labeled_sentiment_data_unified.csv"
OUTPUT = "articles_kepler.geojson"

# ---- Load a PUBLIC Indonesian NER model ----
ner_model = "indolem/indolem-ner"

tokenizer = AutoTokenizer.from_pretrained(ner_model)
model = AutoModelForTokenClassification.from_pretrained(ner_model)

ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# ---- Clean place names ----
def clean_location_name(loc):
    return (
        loc.replace("Kab.", "Kabupaten")
           .replace("Kota ", "")
           .replace("Prop.", "Provinsi")
           .strip()
    )

# ---- Geocode ----
def geocode_place(place):
    place = clean_location_name(place)
    url = f"https://nominatim.openstreetmap.org/search?q={place}&format=json&countrycodes=id&limit=1"
    time.sleep(1)  # API rate limit
    r = requests.get(url, headers={"User-Agent": "KeplerMapper/1.0"})
    data = r.json()
    if len(data) == 0:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])

# ---- Load your CSV ----
df = pd.read_csv(CSV_PATH)

features = []

for idx, row in df.iterrows():
    text = row["text"]
    sentiment = row["sentiment_label"]

    # Extract NER entities
    ents = ner(text)

    # Only location entities
    locs = [e["word"] for e in ents if e["entity_group"] == "LOC"]

    if len(locs) == 0:
        continue

    for loc in locs:
        coords = geocode_place(loc)
        if coords is None:
            continue

        lat, lon = coords

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "article_text": text[:200] + "...",
                "location": loc,
                "sentiment": sentiment
            }
        }

        features.append(feature)

# ---- Save GeoJSON ----
geojson = {"type": "FeatureCollection", "features": features}

with open(OUTPUT, "w") as f:
    json.dump(geojson, f, indent=2)

print("Saved to", OUTPUT)

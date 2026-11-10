"""Build data/businesses.csv and data/businesses_photos from the Yelp Open Dataset.

    python scripts/prepare_yelp.py --yelp PATH/yelp_dataset --photos PATH/yelp_photos

Keeps the businesses that have a review and a captioned photo: one review
and one photo per business.
"""
import argparse
import json
import os
import shutil

import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yelp", required=True, help="folder with yelp_academic_dataset_*.json")
    ap.add_argument("--photos", required=True, help="folder with photos.json and photos/")
    ap.add_argument("--limit", type=int, default=None, help="number of businesses to keep (default: all)")
    args = ap.parse_args()

    review = {}
    with open(os.path.join(args.yelp, "yelp_academic_dataset_review.json"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            review.setdefault(r["business_id"], r["text"])
    photo = {}
    with open(os.path.join(args.photos, "photos.json"), encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("caption"):
                photo.setdefault(p["business_id"], p)

    rows = []
    out = os.path.join(DATA, "businesses_photos")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(args.yelp, "yelp_academic_dataset_business.json"), encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            bid = b["business_id"]
            if bid not in review or bid not in photo:
                continue
            p = photo[bid]
            src = os.path.join(args.photos, "photos", p["photo_id"] + ".jpg")
            if not os.path.exists(src):
                continue
            shutil.copy(src, os.path.join(out, p["photo_id"] + ".jpg"))
            rows.append({"business_id": bid, "name": b.get("name", ""), "address": b.get("address", ""),
                         "city": b.get("city", ""), "state": b.get("state", ""), "stars": b.get("stars", ""),
                         "hours": b.get("hours") or {}, "photo_id": p["photo_id"],
                         "photo_caption": p.get("caption", ""), "photo_label": p.get("label", ""),
                         "review": review[bid]})
            if args.limit and len(rows) >= args.limit:
                break
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "businesses.csv"), index=False, encoding="utf-8")
    print(f"{len(rows)} businesses written to data/businesses.csv")


if __name__ == "__main__":
    main()

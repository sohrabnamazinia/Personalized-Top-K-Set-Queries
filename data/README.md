# Datasets

The datasets are not part of this repository. Download them from their
sources and place the files in this folder under the names below.

| Dataset | Source | Files expected here |
|---|---|---|
| Hotels | https://www.kaggle.com/dsv/7542286 | `hotels.csv` |
| Movies (Wikipedia Movie Plots) | https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots | `wiki_movie_plots_deduped.csv` |
| Yelp Businesses (Yelp Open Dataset, with photos) | https://business.yelp.com/data/resources/open-dataset/ | `businesses.csv`, `businesses_photos/` |

The two Yelp files are built from the downloaded dataset with

```
python scripts/prepare_yelp.py --yelp PATH/yelp_dataset --photos PATH/yelp_photos
```

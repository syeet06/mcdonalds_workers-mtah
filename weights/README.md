# Trained weights

The trained files the submission loads, committed so nobody has to retrain.

The ball detector's output on our match (`ball_video.npz`) is not committed: it is frame-level data
derived from the match video. Build it (~11.6 h) or email [chi.tsang.25@ucl.ac.uk](mailto:chi.tsang.25@ucl.ac.uk?subject=Request%20for%20ball_video.npz%20%28McDonald%27s%20Workers%2C%20More%20than%20a%20Hack%202027%29&body=Hi%2C%0A%0AI%27m%20reproducing%20the%20McDonald%27s%20Workers%20submission%20for%20More%20than%20a%20Hack%202027.%20Could%20you%20please%20send%20me%20your%20copy%20of%20ball_video.npz%2C%20the%20ball%20detector%27s%20output%20on%20the%20match%20video%2C%20so%20I%20can%20run%20the%20pipeline%20without%20building%20it%3F%0A%0AThanks%2C%0A) or [sui.lim.25@ucl.ac.uk](mailto:sui.lim.25@ucl.ac.uk?subject=Request%20for%20ball_video.npz%20%28McDonald%27s%20Workers%2C%20More%20than%20a%20Hack%202027%29&body=Hi%2C%0A%0AI%27m%20reproducing%20the%20McDonald%27s%20Workers%20submission%20for%20More%20than%20a%20Hack%202027.%20Could%20you%20please%20send%20me%20your%20copy%20of%20ball_video.npz%2C%20the%20ball%20detector%27s%20output%20on%20the%20match%20video%2C%20so%20I%20can%20run%20the%20pipeline%20without%20building%20it%3F%0A%0AThanks%2C%0A)
for our copy and put it here; see "The video ball" in the top-level README.

| file | what it is | made by |
|---|---|---|
| `xgb_both_all7.json` | XGBoost possession model, trained on clean and degraded copies of the seven public IDSSE matches | `python -m pipeline.model.train_six_tune_seventh --tune --refit` (writes `pipeline/model/models/`; copy the file here) |
| `ball_video_finetuned.pt` | WASB ball detector fine-tuned on this match's tactical video, with algorithm-made labels only | `pipeline/ball_video_train.py` (writes `pipeline/cache/ball_video/finetuned.pt`; copy the file here) |

The code reads them through `WEIGHTS_DIR` in `pipeline/common.py`.

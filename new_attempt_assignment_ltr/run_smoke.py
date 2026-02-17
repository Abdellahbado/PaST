from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset_builder import BuildConfig, build_ranking_dataset
from .train_ltr import TrainConfig, train_xgb_ranker
from .eval_ltr import evaluate_ranker_predictions


def main(out_dir: str = "new_attempt_runs/smoke"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds_path = build_ranking_dataset(out_dir=str(out / "data"), cfg=BuildConfig(n_instances=50, pool_size=30))

    model_path = train_xgb_ranker(TrainConfig(dataset_npz=ds_path, out_dir=str(out / "model")))

    # quick eval on full dataset (not a proper test, just sanity)
    import xgboost as xgb

    data = np.load(ds_path)
    X, y, qid = data["X"], data["y"], data["qid"]
    booster = xgb.Booster()
    booster.load_model(model_path)
    pred = booster.predict(xgb.DMatrix(X))

    stats = evaluate_ranker_predictions(qid=qid, y_true=y, y_pred=pred)

    with open(out / "smoke_eval.json", "w") as f:
        json.dump(stats.__dict__, f, indent=2)

    print(stats)


if __name__ == "__main__":
    main()

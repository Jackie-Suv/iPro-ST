#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from datetime import datetime

import pandas as pd
from autogluon.tabular import TabularPredictor


def load_topk_features(mrmr_csv: str, topk: int = 368, feature_col: str = "Feature"):
    df_rank = pd.read_csv(mrmr_csv)
    if feature_col not in df_rank.columns:
        raise ValueError(f"[ERROR] '{mrmr_csv}' missing column '{feature_col}'. "
                         f"Available columns: {list(df_rank.columns)}")
    feats = df_rank[feature_col].astype(str).tolist()[:topk]
    # 去重但保持顺序
    seen = set()
    feats = [f for f in feats if not (f in seen or seen.add(f))]
    return feats


def ensure_features_exist(df: pd.DataFrame, features: list):
    exist = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    return exist, missing


def train_and_save(
    train_csv: str,
    mrmr_csv: str,
    outdir: str,
    label: str = "y1",
    topk: int = 368,
    presets: str = "best_quality",
    time_limit: int | None = None,
    random_seed: int = 42,
):
    # 1) load data
    train_df = pd.read_csv(train_csv)
    if label not in train_df.columns:
        raise ValueError(f"[ERROR] label column '{label}' not found in train_csv: {train_csv}")

    # 2) load top-k features
    top_feats = load_topk_features(mrmr_csv, topk=topk, feature_col="Feature")
    top_feats_exist, missing = ensure_features_exist(train_df, top_feats)
    if len(top_feats_exist) == 0:
        raise ValueError("[ERROR] None of the top-k features exist in training data columns.")

    if missing:
        print(f"[WARN] {len(missing)} features in mRMR list not found in train data. "
              f"Example: {missing[:10]}")

    use_cols = top_feats_exist + [label]
    train_df = train_df[use_cols].copy()
    train_df = train_df.dropna(subset=[label])  # 防止label缺失

    # 3) make model path
    os.makedirs(outdir, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(outdir, f"autogluon_iProST_{run_tag}")

    # 4) train
    predictor = TabularPredictor(
        label=label,
        path=model_path,
        eval_metric="auc",   # 也可换成 "mcc" 或你论文里主指标
    ).fit(
        train_data=train_df,
        presets=presets,
        time_limit=time_limit,
        random_seed=random_seed,
    )

    # 5) choose final model name
    all_models = predictor.get_model_names()
    best_model = predictor.get_model_best()
    final_model = "WeightedEnsemble_L2" if "WeightedEnsemble_L2" in all_models else best_model

    print(f"[INFO] Best model: {best_model}")
    print(f"[INFO] Final model (used for inference): {final_model}")

    # 6) delete unnecessary models to reduce size (keep final + dependencies)
    #   - 如果 final_model 是 WeightedEnsemble_L2，会自动保留其依赖基模型
    predictor.delete_models(models_to_keep=[final_model], dry_run=False)
    predictor.save()

    # 7) save metadata (features list, final_model name, etc.)
    meta = {
        "label": label,
        "topk": topk,
        "mrmr_csv": os.path.abspath(mrmr_csv),
        "train_csv": os.path.abspath(train_csv),
        "model_path": os.path.abspath(model_path),
        "final_model": final_model,
        "best_model_before_prune": best_model,
        "n_features_used": len(top_feats_exist),
        "missing_features_in_train": missing,
    }
    with open(os.path.join(model_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    pd.Series(top_feats_exist, name="Feature").to_csv(
        os.path.join(model_path, "top_features_used.csv"), index=False
    )

    print(f"[OK] Saved predictor to: {model_path}")
    return model_path, final_model


def predict_with_saved(
    model_path: str,
    test_csv: str,
    mrmr_csv: str,
    out_csv: str,
    label: str = "y1",
    topk: int = 368,
    model_name: str = "WeightedEnsemble_L2",
):
    predictor = TabularPredictor.load(model_path)

    test_df = pd.read_csv(test_csv)

    top_feats = load_topk_features(mrmr_csv, topk=topk, feature_col="Feature")
    top_feats_exist, missing = ensure_features_exist(test_df, top_feats)
    if len(top_feats_exist) == 0:
        raise ValueError("[ERROR] None of the top-k features exist in test data columns.")

    if missing:
        print(f"[WARN] {len(missing)} features in mRMR list not found in test data. "
              f"Example: {missing[:10]}")

    X = test_df[top_feats_exist].copy()

    # 若模型不存在该名字，则自动用 best
    all_models = predictor.get_model_names()
    if model_name not in all_models:
        best = predictor.get_model_best()
        print(f"[WARN] '{model_name}' not found in loaded predictor. Use best='{best}' instead.")
        model_name = best

    y_pred = predictor.predict(X, model=model_name)

    # 对二分类 / 多分类都兼容
    proba = predictor.predict_proba(X, model=model_name)

    out = test_df.copy()
    out["pred"] = y_pred

    if isinstance(proba, pd.DataFrame):
        # 多分类：每一列为一个类别概率
        for c in proba.columns:
            out[f"proba_{c}"] = proba[c].values
    else:
        # 二分类：Series通常为正类概率（具体取决于label编码）
        out["proba"] = proba.values

    out.to_csv(out_csv, index=False)
    print(f"[OK] Prediction saved to: {out_csv}")

    # 若测试集含真实标签，顺便评估一下
    if label in test_df.columns:
        eval_res = predictor.evaluate(test_df[top_feats_exist + [label]], model=model_name, silent=True)
        print("[INFO] Evaluation on test set:", eval_res)


def main():
    parser = argparse.ArgumentParser(
        description="Train & save AutoGluon model using top-368 features from mRMR-rank.csv, "
                    "prefer WeightedEnsemble_L2 as final model; optionally predict on test set."
    )
    parser.add_argument("--train_csv", type=str, required=True, help="Training CSV file path.")
    parser.add_argument("--test_csv", type=str, default=None, help="(Optional) Test CSV file path.")
    parser.add_argument("--mrmr_csv", type=str, default="mRMR-rank.csv", help="mRMR rank CSV path.")
    parser.add_argument("--label", type=str, default="y1", help="Label column name.")
    parser.add_argument("--topk", type=int, default=368, help="Top-K features to use.")
    parser.add_argument("--outdir", type=str, default="./models/", help="Directory to save models.")
    parser.add_argument("--presets", type=str, default="best_quality", help="AutoGluon presets.")
    parser.add_argument("--time_limit", type=int, default=None, help="Time limit (seconds) for AutoGluon fit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--pred_out", type=str, default="predictions.csv", help="Output CSV for predictions.")
    args = parser.parse_args()

    model_path, final_model = train_and_save(
        train_csv=args.train_csv,
        mrmr_csv=args.mrmr_csv,
        outdir=args.outdir,
        label=args.label,
        topk=args.topk,
        presets=args.presets,
        time_limit=args.time_limit,
        random_seed=args.seed,
    )

    if args.test_csv is not None:
        predict_with_saved(
            model_path=model_path,
            test_csv=args.test_csv,
            mrmr_csv=args.mrmr_csv,
            out_csv=args.pred_out,
            label=args.label,
            topk=args.topk,
            model_name=final_model,  # 训练时确认过的 final model
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.rag_encoder import HeterogeneousFeatureEncoder
from dataset_provider.dataset4LLM_utils import read_dataset


def _load_dataset(dataset_name: str, split: str = "train") -> pd.DataFrame:
    supported = ("MOVER_EPIC", "MOVER_SIS", "INSPIRE", "Multimodal_SDP", "OursSDP")
    if dataset_name not in supported:
        raise ValueError(f"暂只支持 {supported}，收到: {dataset_name}")

    if split not in ("train", "val", "all"):
        raise ValueError(f"split 仅支持 train/val/all，收到: {split}")

    root_path = "/home/likx/time_series_forecasting/SDP_DATASET"
    if dataset_name != "OursSDP" :
        root_path = os.path.join(
            "/home/likx/time_series_forecasting/SDP_DATASET", dataset_name
        )

    def _read_pkl(sp: str) -> pd.DataFrame:
        path = os.path.join(root_path, f"{sp}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到数据文件: {path}")
        return read_dataset(path)

    if split == "all":
        df_train = _read_pkl("train")
        df_val = _read_pkl("val")
        return pd.concat([df_train, df_val], ignore_index=True)

    return _read_pkl(split)


def _build_feature_slices(
    encoder: HeterogeneousFeatureEncoder, encoded_features: Dict[str, np.ndarray]
) -> Tuple[Dict[str, Tuple[int, int]], List[str], np.ndarray]:
    feature_slices: Dict[str, Tuple[int, int]] = {}
    feature_names_in_order: List[str] = []

    cat_slices: Dict[str, Tuple[int, int]] = {}
    cur = 0
    for cat, arr in encoded_features.items():
        dim = arr.shape[1]
        cat_slices[cat] = (cur, cur + dim)
        cur += dim

    for cat, features in encoder.active_feature_categories.items():
        if cat not in encoded_features:
            continue
        if not features:
            continue
        cat_start, cat_end = cat_slices[cat]
        cat_dim = cat_end - cat_start
        used_features = [f for f in features if f in features]
        if not used_features:
            continue

        per_feat_dim = cat_dim // len(used_features)
        start = cat_start
        for i, feat in enumerate(used_features):
            if i == len(used_features) - 1:
                end = cat_end
            else:
                end = start + per_feat_dim
            feature_slices[feat] = (start, end)
            feature_names_in_order.append(feat)
            start = end

    full_matrix = []
    for cat in encoder.active_feature_categories.keys():
        if cat in encoded_features:
            full_matrix.append(encoded_features[cat])
    X = np.concatenate(full_matrix, axis=1) if full_matrix else np.empty((0, 0))

    return feature_slices, feature_names_in_order, X


def compute_pca_feature_importance(
    dataset_name: str,
    split: str = "train",
    n_components: int = 16,
    use_grouped_encoding: bool = False,
) -> Dict[str, float]:
    """
    对指定数据集做 PCA 分析，返回 {特征名: 重要性分数}。
    """
    df = _load_dataset(dataset_name, split=split)


    if dataset_name == "OursSDP":
        if "手麻系统中手术状态时间历史记录唯一标识" not in df.columns:
            raise KeyError(f"{dataset_name} 中找不到 'duration' 列，无法做 PCA-重要性分析。")
        df = df.dropna(subset=["手麻系统中手术状态时间历史记录唯一标识"]).reset_index(drop=True)
    else:
        if "duration" not in df.columns:
            raise KeyError(f"{dataset_name} 中找不到 'duration' 列，无法做 PCA-重要性分析。")
        df = df.dropna(subset=["duration"]).reset_index(drop=True)

    encoder = HeterogeneousFeatureEncoder(
        dataset_name=dataset_name, use_grouped_encoding=use_grouped_encoding
    )
    encoded_features = encoder.fit_transform(df)

    feature_slices, feature_names_order, X = _build_feature_slices(encoder, encoded_features)
    if X.size == 0:
        raise RuntimeError("编码后特征矩阵为空，无法做 PCA 分析。")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_components = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=max_components, random_state=42)
    _ = pca.fit_transform(X_scaled)

    loadings = pca.components_
    var_ratio = pca.explained_variance_ratio_

    importance: Dict[str, float] = {}
    for feat, (start, end) in feature_slices.items():
        if start >= end:
            continue
        feat_loadings = np.abs(loadings[:, start:end])
        mean_per_comp = feat_loadings.mean(axis=1)
        score = float(np.sum(mean_per_comp * var_ratio))
        importance[feat] = score

    total = sum(importance.values())
    if total > 0:
        for k in list(importance.keys()):
            importance[k] = float(importance[k] / total)

    importance_sorted = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )
    return importance_sorted


def main():
    parser = argparse.ArgumentParser(
        description="使用 PCA 对 MOVER_EPIC / MOVER_SIS 的编码特征做重要性分析（仅分析，不改动现有 RAG 框架）。"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["MOVER_EPIC", "MOVER_SIS", "INSPIRE", "Multimodal_SDP", "OursSDP"],
        help="要分析的数据集名称。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "all"],
        help="使用哪个划分做 PCA（默认 train）。",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=50,
        help="PCA 主成分个数上限（会自动截断到不超过特征维度）。",
    )
    parser.add_argument(
        "--use_grouped_encoding",
        action="store_true",
        help="是否使用分组编码（默认关闭；如果你想和 RAG 完全对齐，也可以打开）。",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="可选：将特征重要性结果保存为 JSON 文件路径。",
    )
    args = parser.parse_args()

    importance = compute_pca_feature_importance(
        dataset_name=args.dataset_name,
        split=args.split,
        n_components=args.n_components,
        use_grouped_encoding=args.use_grouped_encoding,
    )

    print(f"\n=== PCA 特征重要性（{args.dataset_name}, split={args.split}）===")
    for feat, score in importance.items():
        print(f"{feat:20s} : {score:.4f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(importance, f, ensure_ascii=False, indent=2)
        print(f"\n已将结果保存到: {args.output_json}")


if __name__ == "__main__":
    main()


"""
MOVER_EPIC 结构化特征编码器：用于个体表示中的非文本部分（基本人口统计学 + ASA + 时间）。

约定：
- procedure_name / diagnosis_names 等文本字段已在 text_representation 中编码，不在此处重复编码。
- 仅输出 continuous（float 矩阵），供 structured_branch 使用。
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder


MOVER_EPIC_NUMERICAL = [
    "patient_age",
    "patient_height",
    # 有些版本 EPIC 也包含体重；缺失时会自动填 0，但为保持维度一致显式列出
    "patient_weight",
    "surgery_start_time",
]

MOVER_EPIC_ORDINAL = [
    "assessment_asa",
]

MOVER_EPIC_CATEGORICAL_LOW = [
    "patient_gender",
]


def _to_scalar(x):
    if x is None:
        return None
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return None
        x = x[0]
    return x


def _date_to_numeric(val):
    if val is None:
        return 0.0
    try:
        ts = pd.to_datetime(str(val), errors="coerce")
        if pd.isna(ts):
            return 0.0
        return float(ts.toordinal())
    except Exception:
        return 0.0


class MOVER_EPICStructuredEncoder:
    def __init__(self):
        self.scaler_numerical = StandardScaler()
        self.encoder_ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.encoder_categorical_low = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._cat_low_cols = []
        self._n_cat_low_out = 1
        self.is_fitted = False

    def _ensure_scalar_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            out[col] = out[col].map(lambda v: _to_scalar(v) if isinstance(v, (list, tuple, np.ndarray)) else v)
        return out

    def fit(self, data):
        df = self._ensure_scalar_cells(data if isinstance(data, pd.DataFrame) else pd.DataFrame(data))

        # numerical
        X_num = np.zeros((len(df), len(MOVER_EPIC_NUMERICAL)), dtype=np.float64)
        for j, c in enumerate(MOVER_EPIC_NUMERICAL):
            if c not in df.columns:
                continue
            if c == "surgery_start_time":
                X_num[:, j] = df[c].apply(_date_to_numeric).values
            else:
                X_num[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0).values
        self.scaler_numerical.fit(X_num)

        # ordinal
        X_ord = df.reindex(columns=MOVER_EPIC_ORDINAL).fillna("UNK")
        self.encoder_ordinal.fit(X_ord)

        # categorical_low
        cat_low_cols = [c for c in MOVER_EPIC_CATEGORICAL_LOW if c in df.columns]
        self._cat_low_cols = cat_low_cols
        if cat_low_cols:
            X_cat = df[cat_low_cols].fillna("UNK")
            self.encoder_categorical_low.fit(X_cat)
            if hasattr(self.encoder_categorical_low, "categories_"):
                self._n_cat_low_out = int(sum(len(cats) for cats in self.encoder_categorical_low.categories_))
            else:
                self._n_cat_low_out = 1
        else:
            self._n_cat_low_out = 1

        self.is_fitted = True
        return self

    def transform(self, data, return_dict: bool = False):
        if not self.is_fitted:
            raise ValueError("MOVER_EPICStructuredEncoder 尚未 fit")
        df = self._ensure_scalar_cells(data if isinstance(data, pd.DataFrame) else pd.DataFrame(data))
        N = len(df)

        parts = []
        # numerical
        X_num = np.zeros((N, len(MOVER_EPIC_NUMERICAL)), dtype=np.float64)
        for j, c in enumerate(MOVER_EPIC_NUMERICAL):
            if c not in df.columns:
                continue
            if c == "surgery_start_time":
                X_num[:, j] = df[c].apply(_date_to_numeric).values
            else:
                X_num[:, j] = pd.to_numeric(df[c], errors="coerce").fillna(0).values
        parts.append(self.scaler_numerical.transform(X_num))

        # ordinal
        X_ord = df.reindex(columns=MOVER_EPIC_ORDINAL).fillna("UNK")
        parts.append(self.encoder_ordinal.transform(X_ord))

        # categorical_low
        if self._cat_low_cols:
            X_cat = df.reindex(columns=self._cat_low_cols).fillna("UNK")
            parts.append(self.encoder_categorical_low.transform(X_cat))
        else:
            parts.append(np.zeros((N, self._n_cat_low_out), dtype=np.float64))

        continuous = np.hstack(parts).astype(np.float32, copy=False)
        if return_dict:
            return {"continuous": continuous}
        return continuous

    def get_continuous_dim(self) -> int:
        if not self.is_fitted:
            raise ValueError("尚未 fit")
        return len(MOVER_EPIC_NUMERICAL) + len(MOVER_EPIC_ORDINAL) + self._n_cat_low_out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "scaler_numerical": self.scaler_numerical,
                    "encoder_ordinal": self.encoder_ordinal,
                    "encoder_categorical_low": self.encoder_categorical_low,
                    "_cat_low_cols": self._cat_low_cols,
                    "_n_cat_low_out": self._n_cat_low_out,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.scaler_numerical = d["scaler_numerical"]
        self.encoder_ordinal = d["encoder_ordinal"]
        self.encoder_categorical_low = d["encoder_categorical_low"]
        self._cat_low_cols = d.get("_cat_low_cols", [])
        self._n_cat_low_out = int(d.get("_n_cat_low_out", 1))
        self.is_fitted = bool(d.get("is_fitted", True))
        return self


"""
MMSDP 结构化特征编码器：用于个体表示中的非文本部分（数值、序数、类别、布尔、诊断码等）。

- 仅对 surgeons、anesthesiologists 做 embedding（以索引形式输出）；department 当作 categorical_low。
- 诊断码（ICD）转为固定 3 维向量后参与编码。
- procedures、diagnoses、position 由 text_representation 的 BERT 统一编码，不在此处单独编码。
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder


MMSDP_NUMERICAL = [
    # 基本信息
    'patient_age', 'patient_height', 'patient_weight',
    # 实验检查
    'diagnostic_CBC_RBC', 'diagnostic_CBC_WBC', 'diagnostic_CBC_HGB', 'diagnostic_CBC_PLT',
    # 手术相关日期（格式示例：2021/8/26），将被解析为数值时间轴
    'date',
]
MMSDP_ORDINAL = ['level', 'assessment_asa']
MMSDP_CATEGORICAL_LOW = ['patient_gender', 'type', 'degree_of_mouth_opening', 'neck_activity', 'department']
MMSDP_BOOLEAN = [
    'diagnostic_electrolyte', 'diagnostic_ECG', 'diagnostic_liver', 'diagnostic_kidney',
    'disease_smoking', 'disease_alcoholism', 'disease_allergies', 'disease_respiratory',
    'disease_nerve', 'disease_circulatory'
]


def _to_scalar(x):
    if x is None:
        return None
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return None
        x = x[0]
    return x


def _normalize_boolean_mmsdp(val):
    if val is None:
        return 'false'
    s = str(val).lower().strip()
    if s in ('true', 't', '1', '是'):
        return 'true'
    return 'false'


def _diagnosis_code_to_fixed_dim(icd_val):
    """
    将诊断码（ICD）转为固定 3 维向量。
    - 若已是长度为 3 的 list/array：直接转为 float
    - 若为单个数字或可解析字符串：按层级拆成 3 维并归一化
    - 否则返回 [0, 0, 0]
    """
    if icd_val is None or (isinstance(icd_val, float) and np.isnan(icd_val)):
        return [0.0, 0.0, 0.0]
    if isinstance(icd_val, (list, np.ndarray)):
        arr = np.asarray(icd_val, dtype=float).flatten()
        if arr.size >= 3:
            return [float(arr[0]), float(arr[1]), float(arr[2])]
        if arr.size == 1:
            v = float(arr[0])
            return [v / 100.0, (v % 100) / 100.0, (v % 10) / 10.0]
        pad = [0.0] * (3 - arr.size)
        return [float(x) for x in arr] + pad
    try:
        v = float(icd_val)
    except (TypeError, ValueError):
        s = str(icd_val).strip()
        try:
            v = float(s)
        except ValueError:
            return [0.0, 0.0, 0.0]
    v = max(0, min(1000, v))
    return [v / 100.0, (v % 100) / 100.0, (v % 10) / 10.0]


def _date_to_numeric(val):
    """
    将 'YYYY/M/D' 之类的日期字符串转为一个稳定的一维数值特征（单位：天）。
    解析失败时返回 0。
    """
    if val is None:
        return 0.0
    try:
        ts = pd.to_datetime(str(val), errors='coerce')
        if pd.isna(ts):
            return 0.0
        # 使用到天的分辨率，避免过细的秒级噪声
        return float(ts.toordinal())
    except Exception:
        return 0.0


class MMSDPStructuredEncoder:
    """
    MMSDP 结构化特征编码器（输出 continuous；历史上曾包含医生索引，但现已不再使用）。
    """

    def __init__(self):
        self.scaler_numerical = StandardScaler()
        self.encoder_ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.encoder_categorical_low = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder_boolean = LabelEncoder()
        self.surgeon_to_idx = {'__unk__': 0}
        self.anesthesiologist_to_idx = {'__unk__': 0}
        self._cat_low_cols = []
        self._n_cat_low_out = 1
        self.is_fitted = False

    def _ensure_scalar_cells(self, df):
        out = df.copy()
        for col in out.columns:
            out[col] = out[col].map(lambda x: _to_scalar(x) if isinstance(x, (list, tuple, np.ndarray)) else x)
        return out

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            df = self._ensure_scalar_cells(data)
        else:
            df = pd.DataFrame(data)
            df = self._ensure_scalar_cells(df)

        # numerical
        X_num = np.zeros((len(df), len(MMSDP_NUMERICAL)))
        for j, c in enumerate(MMSDP_NUMERICAL):
            if c in df.columns:
                if c == 'date':
                    X_num[:, j] = df[c].apply(_date_to_numeric).values
                else:
                    X_num[:, j] = pd.to_numeric(df[c], errors='coerce').fillna(0).values
        self.scaler_numerical.fit(X_num)

        # ordinal
        X_ord = df.reindex(columns=MMSDP_ORDINAL).fillna('无')
        self.encoder_ordinal.fit(X_ord)

        # categorical_low
        cat_low_cols = [c for c in MMSDP_CATEGORICAL_LOW if c in df.columns]
        self._cat_low_cols = cat_low_cols
        if cat_low_cols:
            X_cat = df[cat_low_cols].fillna('无')
            self.encoder_categorical_low.fit(X_cat)
            # sklearn 版本兼容：不同版本 OneHotEncoder 的输出维度属性不同
            if hasattr(self.encoder_categorical_low, "categories_"):
                self._n_cat_low_out = int(sum(len(c) for c in self.encoder_categorical_low.categories_))
            else:
                # 兜底：至少不为 0，避免后续拼接出错
                self._n_cat_low_out = 1
        else:
            self._n_cat_low_out = 1

        # boolean
        bool_vals = []
        for c in MMSDP_BOOLEAN:
            if c in df.columns:
                bool_vals.extend(_normalize_boolean_mmsdp(v) for v in df[c].tolist())
        if bool_vals:
            self.encoder_boolean.fit(np.array(['false', 'true'] + list(set(bool_vals))))
        else:
            self.encoder_boolean.fit(np.array(['false', 'true']))

        # high-cardinality id maps
        if 'surgeons' in df.columns:
            for v in df['surgeons'].tolist():
                v = _to_scalar(v)
                if v is not None and str(v).strip() and str(v) not in self.surgeon_to_idx:
                    self.surgeon_to_idx[str(v)] = len(self.surgeon_to_idx)
        if 'anesthesiologists' in df.columns:
            for v in df['anesthesiologists'].tolist():
                v = _to_scalar(v)
                if v is not None and str(v).strip() and str(v) not in self.anesthesiologist_to_idx:
                    self.anesthesiologist_to_idx[str(v)] = len(self.anesthesiologist_to_idx)

        self.is_fitted = True
        return self

    def transform(self, data, return_dict=False):
        if not self.is_fitted:
            raise ValueError("MMSDPStructuredEncoder 尚未 fit")
        if isinstance(data, pd.DataFrame):
            df = self._ensure_scalar_cells(data)
        else:
            df = pd.DataFrame(data)
            df = self._ensure_scalar_cells(df)
        N = len(df)

        parts = []
        # numerical
        X_num = np.zeros((N, len(MMSDP_NUMERICAL)))
        for j, c in enumerate(MMSDP_NUMERICAL):
            if c in df.columns:
                if c == 'date':
                    X_num[:, j] = df[c].apply(_date_to_numeric).values
                else:
                    X_num[:, j] = pd.to_numeric(df[c], errors='coerce').fillna(0).values
        parts.append(self.scaler_numerical.transform(X_num))

        # ordinal
        X_ord = df.reindex(columns=MMSDP_ORDINAL).fillna('无')
        parts.append(self.encoder_ordinal.transform(X_ord))

        # categorical_low
        if self._cat_low_cols:
            X_cat = df.reindex(columns=self._cat_low_cols).fillna('无')
            parts.append(self.encoder_categorical_low.transform(X_cat))
        else:
            parts.append(np.zeros((N, self._n_cat_low_out)))

        # boolean
        bool_cols = [c for c in MMSDP_BOOLEAN if c in df.columns]
        if bool_cols:
            bool_encoded = []
            for c in bool_cols:
                vals = [_normalize_boolean_mmsdp(v) for v in df[c].tolist()]
                enc = self.encoder_boolean.transform(vals)
                bool_encoded.append(enc.reshape(-1, 1))
            parts.append(np.hstack(bool_encoded))
        else:
            parts.append(np.zeros((N, len(MMSDP_BOOLEAN))))

        # ICD fixed dim
        if 'ICD' in df.columns:
            icd_arr = np.array([_diagnosis_code_to_fixed_dim(v) for v in df['ICD'].tolist()], dtype=np.float64)
            parts.append(icd_arr)
        else:
            parts.append(np.zeros((N, 3)))

        continuous = np.hstack(parts)

        if return_dict:
            return {
                'continuous': continuous,
            }
        # 当前项目中只在 return_dict=True 的路径下使用该编码器，
        # 因此非字典返回只保留 continuous，便于后续维护。
        return continuous

    def get_continuous_dim(self):
        if not self.is_fitted:
            raise ValueError("尚未 fit")
        return len(MMSDP_NUMERICAL) + len(MMSDP_ORDINAL) + self._n_cat_low_out + len(MMSDP_BOOLEAN) + 3

    def get_surgeon_vocab_size(self):
        return len(self.surgeon_to_idx) if self.surgeon_to_idx else 1

    def get_anesthesiologist_vocab_size(self):
        return len(self.anesthesiologist_to_idx) if self.anesthesiologist_to_idx else 1

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler_numerical': self.scaler_numerical,
                'encoder_ordinal': self.encoder_ordinal,
                'encoder_categorical_low': self.encoder_categorical_low,
                'encoder_boolean': self.encoder_boolean,
                'surgeon_to_idx': self.surgeon_to_idx,
                'anesthesiologist_to_idx': self.anesthesiologist_to_idx,
                '_cat_low_cols': self._cat_low_cols,
                '_n_cat_low_out': self._n_cat_low_out,
                'is_fitted': self.is_fitted,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.scaler_numerical = d['scaler_numerical']
        self.encoder_ordinal = d['encoder_ordinal']
        self.encoder_categorical_low = d['encoder_categorical_low']
        self.encoder_boolean = d['encoder_boolean']
        self.surgeon_to_idx = d.get('surgeon_to_idx') or {'__unk__': 0}
        self.anesthesiologist_to_idx = d.get('anesthesiologist_to_idx') or {'__unk__': 0}
        self._cat_low_cols = d.get('_cat_low_cols', [])
        self._n_cat_low_out = d.get('_n_cat_low_out', 1)
        self.is_fitted = d.get('is_fitted', True)
        return self


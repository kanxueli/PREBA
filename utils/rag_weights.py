#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG权重配置模块
统一管理特征权重和类别权重配置。

运行时通过 set_weight_scheme() / --rag_weight_scheme 选择权重来源：
  - paper_pca: 论文中写死的 PCA 权重（默认 FAISS 路径，兼容旧索引）
  - mutual_info / pca / uniform / ridge_coef / permutation / rf_mdi / shap:
      从 feature_weighting/results/<dataset>/<scheme>/weights.json 加载
  - 也可 --rag_weight_json 指向任意 weights.json
"""

import json
import os

# 消融：True 时各特征标量权重一律为 1.0 (用于消融实验，启用PCA则置为False)。
ABLATION_UNIFORM_FEATURE_WEIGHTS = False

WEIGHT_SCHEMES = (
    "paper_pca",
    "pca",
    "uniform",
    "mutual_info",
    "ridge_coef",
    "permutation",
    "rf_mdi",
    "shap",
)

_DEFAULT_SCHEME = "mutual_info"
_ACTIVE_SCHEME = os.environ.get("RAG_WEIGHT_SCHEME", _DEFAULT_SCHEME)
_WEIGHT_JSON_OVERRIDE = os.environ.get("RAG_WEIGHT_JSON") or None
_LOGGED_RESOLVE_KEY = None
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WEIGHTS_ROOT = os.path.join(_PROJECT_ROOT, "feature_weighting", "results")

_DATASET_DIR_ALIASES = {
    "OurSDP": "OurSDP",
    "OursSDP": "OurSDP",
    "Multimodal_SDP": "Multimodal_SDP",
    "MMSDP": "Multimodal_SDP",
    "INSPIRE": "INSPIRE",
    "MOVER_EPIC": "MOVER_EPIC",
    "MOVER_SIS": "MOVER_SIS",
}


def set_weight_scheme(scheme, weight_json=None):
    """在构建/查询 RAG 之前调用，使 get_feature_weights_* 返回对应权重。"""
    global _ACTIVE_SCHEME, _WEIGHT_JSON_OVERRIDE, _LOGGED_RESOLVE_KEY
    if scheme:
        scheme = str(scheme).strip()
        if scheme not in WEIGHT_SCHEMES:
            raise ValueError(
                "未知 rag_weight_scheme=%r，可选: %s" % (scheme, ", ".join(WEIGHT_SCHEMES))
            )
        _ACTIVE_SCHEME = scheme
    _LOGGED_RESOLVE_KEY = None
    if weight_json:
        _WEIGHT_JSON_OVERRIDE = os.path.abspath(weight_json)
    elif weight_json == "":
        _WEIGHT_JSON_OVERRIDE = None


def get_weight_scheme():
    return _ACTIVE_SCHEME


def get_weight_json_override():
    return _WEIGHT_JSON_OVERRIDE


def faiss_weight_suffix(scheme=None, weight_json=None):
    """paper_pca 沿用旧 FAISS 目录；其它方案加后缀，避免索引混用。"""
    scheme = scheme or _ACTIVE_SCHEME
    json_path = weight_json if weight_json is not None else _WEIGHT_JSON_OVERRIDE
    if json_path:
        stem = os.path.splitext(os.path.basename(json_path))[0]
        return "_w_%s_%s" % (scheme, stem)
    if scheme in (None, "", "paper_pca"):
        return ""
    return "_w_%s" % scheme


def _maybe_uniform_feature_weights(weights: dict) -> dict:
    if not ABLATION_UNIFORM_FEATURE_WEIGHTS:
        return dict(weights)
    return {k: 1.0 for k in weights}


def _dataset_dir_name(dataset_name):
    if not dataset_name:
        return "OurSDP"
    return _DATASET_DIR_ALIASES.get(dataset_name, dataset_name)


def _paper_weights_oursdp():
    return {
        "患者住院科室": 0.40,
        "拟行手术名称": 0.32,
        "手术等级": 0.15,
        "主要诊断": 0.30,
        "手术器械清点单手术名称": 0.32,
        "总体评估时的ASA分级": 0.25,
        "麻醉计划风险评估时患者心功能分级(New York)": 0.15,
        "患者详情中对应的手术类型": 0.15,
        "手术排班的麻醉方法": 0.25,
        "拟选用麻醉药": 0.25,
        "麻醉前访视-手术史详情": 0.05,
        "麻醉前访视病史治疗情况": 0.05,
        "麻醉前访视麻醉史": 0.25,
        "麻醉前访视-过敏史详情": 0.05,
        "患者年龄": 0.12,
        "患者性别": 0.10,
        "麻醉前访视吸烟史标识": 0.06,
        "麻醉前访视酗酒史": 0.06,
        "术前检查中肺功能是否异常": 0.06,
        "术前检查中血气是否异常": 0.06,
        "术前检查中超声心动图是否异常": 0.06,
        "麻醉前访视呼吸系统病史": 0.05,
        "麻醉前访视神经系统病史": 0.05,
        "麻醉前访视消化系统病史": 0.05,
        "麻醉前访视脊柱四肢病史": 0.05,
        "麻醉前访视泌尿系统病史": 0.05,
        "麻醉前访视免疫系统病史": 0.05,
        "麻醉前访视其它系统病史异常情况": 0.05,
        "手术医生": 0.15,
        "麻醉医生姓名": 0.15,
        "手术排班的洗手护士": 0.05,
        "手术排班的巡回护士": 0.05,
        "手术排班的手术房间id": 0.05,
        "手术注意事项": 0.06,
        "麻醉前访视查体-一般情况": 0.05,
        "麻醉前访视查体-心肺听诊详情": 0.05,
        "麻醉前访视中的特殊情况记录": 0.05,
        "术前检查中放射结果详情描述": 0.06,
        "术前检查中心电图检查详细情况": 0.05,
        "术前检查中凝血筛选详细情况": 0.05,
        "手术排班的排班台次": 0.02,
        "患者入手术室时间": 0.05,
    }


def _paper_weights_mmsdp():
    return {
        "department": 0.089,
        "procedures": 0.082,
        "diagnoses": 0.078,
        "level": 0.085,
        "ICD": 0.089,
        "assessment_asa": 0.029,
        "position": 0.083,
        "type": 0.005,
        "patient_gender": 0.016,
        "anesthesiologists": 0.064,
        "surgeons": 0.082,
        "diagnostic_CBC_RBC": 0.003,
        "diagnostic_CBC_WBC": 0.026,
        "diagnostic_CBC_HGB": 0.026,
        "diagnostic_CBC_PLT": 0.033,
        "patient_age": 0.047,
        "degree_of_mouth_opening": 0.006,
        "neck_activity": 0.017,
        "patient_height": 0.016,
        "patient_weight": 0.024,
        "disease_smoking": 0.025,
        "disease_alcoholism": 0.018,
        "disease_allergies": 0.007,
        "diagnostic_electrolyte": 0.0,
        "diagnostic_ECG": 0.0,
        "diagnostic_liver": 0.0,
        "diagnostic_kidney": 0.0,
        "disease_respiratory": 0.015,
        "disease_nerve": 0.017,
        "disease_circulatory": 0.018,
    }


def _paper_weights_inspire():
    return {
        "department.1": 0.158,
        "diagnoses_text": 0.140,
        "assessment_asa": 0.069,
        "ICD": 0.135,
        "preop_labs_latest_json": 0.140,
        "anesthesia": 0.089,
        "patient_age": 0.079,
        "patient_gender": 0.090,
        "patient_weight": 0.034,
        "patient_height": 0.006,
        "emop": 0.066,
    }


def _paper_weights_mover_epic():
    return {
        "procedure_name": 0.30,
        "diagnosis_names": 0.25,
        "assessment_asa": 0.13,
        "patient_age": 0.15,
        "patient_gender": 0.10,
        "patient_height": 0.06,
    }


def _paper_weights_mover_sis():
    return {
        "procedure_name": 0.30,
        "patient_gender": 0.06,
        "patient_height": 0.08,
        "patient_age": 0.14,
        "patient_weight": 0.06,
        "preop_medications": 0.30,
    }


def _paper_weights_for_dataset(dataset_name):
    key = _dataset_dir_name(dataset_name)
    if key == "Multimodal_SDP":
        return _paper_weights_mmsdp()
    if key == "INSPIRE":
        return _paper_weights_inspire()
    if key == "MOVER_EPIC":
        return _paper_weights_mover_epic()
    if key == "MOVER_SIS":
        return _paper_weights_mover_sis()
    return _paper_weights_oursdp()


def _load_weights_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("权重文件为空或格式不是 dict: %s" % path)
    return {str(k): float(v) for k, v in data.items()}


def _scheme_weights_path(dataset_name, scheme):
    return os.path.join(_WEIGHTS_ROOT, _dataset_dir_name(dataset_name), scheme, "weights.json")


def _log_resolve_once(message, key):
    global _LOGGED_RESOLVE_KEY
    if _LOGGED_RESOLVE_KEY != key:
        print(message)
        _LOGGED_RESOLVE_KEY = key


def resolve_feature_weights(dataset_name="OurSDP"):
    """按当前 scheme 返回特征标量权重。database / retriever / encoder 仍调用原来的 getter。"""
    paper = _paper_weights_for_dataset(dataset_name)
    scheme = _ACTIVE_SCHEME
    json_path = _WEIGHT_JSON_OVERRIDE
    ds = _dataset_dir_name(dataset_name)

    if json_path:
        if not os.path.isfile(json_path):
            raise FileNotFoundError("指定的权重 JSON 不存在: %s" % json_path)
        loaded = _load_weights_json(json_path)
        _log_resolve_once("RAG 特征权重: scheme=%s json=%s" % (scheme, json_path), json_path)
        return _maybe_uniform_feature_weights(loaded)

    if scheme == "paper_pca":
        _log_resolve_once("RAG 特征权重: scheme=paper_pca dataset=%s (内置 PCA)" % ds, ("paper_pca", ds))
        return _maybe_uniform_feature_weights(paper)

    if scheme == "uniform":
        _log_resolve_once("RAG 特征权重: scheme=uniform dataset=%s" % ds, ("uniform", ds))
        return {k: 1.0 for k in paper}

    path = _scheme_weights_path(dataset_name, scheme)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "未找到 %s/%s 权重文件: %s\n请先运行 feature_weighting 估计权重，"
            "或改用 --rag_weight_scheme paper_pca / --rag_weight_json" % (
                ds, scheme, path
            )
        )
    loaded = _load_weights_json(path)
    _log_resolve_once("RAG 特征权重: scheme=%s path=%s" % (scheme, path), path)
    return loaded


def get_feature_weights():
    """OurSDP 特征权重（兼容旧调用）。"""
    return resolve_feature_weights("OurSDP")


def get_feature_weights_MMSDP():
    return resolve_feature_weights("Multimodal_SDP")


def get_feature_weights_INSPIRE():
    return resolve_feature_weights("INSPIRE")


def get_feature_weights_MOVER_EPIC():
    return resolve_feature_weights("MOVER_EPIC")


def get_feature_weights_MOVER_SIS():
    return resolve_feature_weights("MOVER_SIS")


def get_category_base_weights():
    """获取类别基础权重配置"""
    return {
        "numerical": 1.0,
        "ordinal": 1.0,
        "categorical_low": 1.0,
        "boolean": 1.0,
        "categorical_high": 1.0,
        "text": 1.0,
    }


def calculate_category_weight(category, encoded_vector_length):
    """根据实际编码长度动态计算类别权重"""
    import numpy as np

    base_weights = get_category_base_weights()

    if category in ["categorical_high", "text"]:
        length_factor = 1.0 / np.sqrt(np.sqrt(encoded_vector_length))
    else:
        length_factor = 1.0 / np.sqrt(encoded_vector_length)

    return base_weights[category] * length_factor

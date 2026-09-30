# PREBA: Surgical Duration Prediction via PCA-Weighted Retrieval-Augmented LLMs and Bayesian Averaging Aggregation

PREBA predicts the duration of a surgical case by grounding an off-the-shelf LLM in the hospital's own historical records. No LLM fine-tuning is required: the model is prompted with clinically similar past cases and with statistical priors computed from the institution's data, and the resulting multi-path predictions are consolidated by Bayesian averaging.

The framework has three components:

1. **Heterogeneous biomedical feature embedding** — numerical, categorical, ordinal, Boolean and free-text fields are encoded with type-specific encoders and concatenated into a single representation.
2. **PCA-weighted retrieval-augmented generation** — retrieval weights are derived from the covariance structure of the institution's historical cases, so that similarity is driven by the variation shared across cases rather than by case-specific fluctuations. Retrieved candidates are then filtered by clinical post-processing (department / procedure / surgery-level matching and IQR outlier removal).
3. **Bayesian averaging aggregation** — the LLM is sampled several times under different decoding temperatures, and the candidates are calibrated against institution-specific statistical priors.

---

## 1. Installation

```bash
git clone https://github.com/kanxueli/PREBA.git
cd PREBA

conda create -n preba python=3.10 -y
conda activate preba

pip install -r requirements.txt
# FAISS is not installed from PyPI under the name "faiss"; install one of:
pip install faiss-cpu==1.9.0     # CPU-only
# conda install -c pytorch faiss-gpu=1.9.0   # GPU

# vLLM is used to serve the LLM backbones locally
pip install vllm
```

---

## 2. Repository layout

```
PREBA/
├── models/
│   └── LLM4SDP.py                  # main entry: retrieval → prompting → multi-path sampling → aggregation
├── utils/
│   ├── rag_encoder.py              # heterogeneous feature embedding (numerical/categorical/ordinal/Boolean/text)
│   ├── text_encoder.py             # BERT-family text encoders for free-text fields
│   ├── *_structured_encoder.py     # per-dataset structured-field encoders (MMSDP / INSPIRE / MOVER-EPIC)
│   ├── rag_weights.py              # retrieval weighting schemes (PCA, uniform, ridge, MI, permutation, RF-MDI, SHAP)
│   ├── rag_database.py             # FAISS index construction over weighted embeddings
│   ├── rag_retriever.py            # cosine-similarity retrieval
│   ├── rag_post_processor.py       # clinical post-processing of the candidate pool
│   ├── prior_hint_generator.py     # statistical priors from matched clinical strata
│   ├── prompt_constructor.py       # prompt assembly (system role, demonstrations, priors, query)
│   ├── ensemble_strategies.py      # aggregation strategies incl. Bayesian averaging
│   ├── calculate_model_performance.py, run_metrics.py   # MAE / RMSE / R² / MAPE
├── dataset_provider/
│   ├── data_preprocessing.py       # raw EHR cleaning and field normalization
│   └── dataset4LLM_utils.py
├── prompts/                        # prompt templates per dataset
├── pca_analysis_results/           # pre-computed PCA retrieval weights per dataset (.json)
├── scripts/                        # vLLM serving scripts for each backbone
├── run_LLM4SDP.sh                  # CLI wrapper around models/LLM4SDP.py
├── run_pca_ananlysis.sh            # recompute PCA retrieval weights
└── requirements.txt
```

---

## 3. Data preparation

| Dataset | Access | Notes |
|---|---|---|
| MMSDP | Request from the authors of [Li et al., 2024] | Chinese, 88,538 cases, 30 preoperative features |
| INSPIRE | https://physionet.org/content/inspire/ | Seoul National University Hospital, ~117K cases |
| MOVER (EPIC) | https://mover.ics.uci.edu/ | UC Irvine Medical Center, 59,075 cases |

Each dataset is expected as three pickled dataframes in one directory:

```
<DATA_ROOT>/<DATASET_NAME>/
├── train_set.pkl
├── val_set.pkl
└── test_set.pkl
```

Raw EHR exports can be cleaned with `dataset_provider/data_preprocessing.py`. The training split is the only source of the retrieval corpus, the PCA weights and the statistical priors; validation is used for hyperparameter selection, and the test split is used only for the final evaluation.

**Paths to set before the first run** (currently hardcoded):

| Location | What to set |
|---|---|
| `models/LLM4SDP.py`, `dataset_load(...)` calls | `<DATA_ROOT>` for each dataset |
| `models/LLM4SDP.py`, `base_path` | directory where FAISS indexes are cached |
| `models/LLM4SDP.py`, `*_model_id` variables | local paths of the LLM checkpoints |
| `scripts/run_*.sh` | checkpoint path and serving port |

---

## 4. Quick start

**Step 1 — serve a backbone with vLLM**

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3-8B --port 60000 --reasoning-parser deepseek_r1
```

Ready-made scripts for Qwen3-4B/8B/14B/32B are in `scripts/`. DeepSeek-R1-Distill-Llama-8B and HuatuoGPT-o1-7B are served the same way.

**Step 2 — build the retrieval index and run PREBA**

```bash
bash run_LLM4SDP.sh \
  --model qwen3 --port 60000 --gpu 0 \
  --dataset_name MMSDP \
  --use_rag --rag_k 8 --index_type Flat \
  --rag_weight_scheme pca \
  --enable_prior_hint \
  --force_rebuild_rag \
  --max_workers 10
```

`--force_rebuild_rag` is only needed the first time, or after the encoder, the weighting scheme or the corpus changes.

**Step 3 — evaluate**

```bash
python utils/calculate_model_performance.py --results <path/to/results.json>
```

This reports MAE, RMSE, R² and MAPE, and can also re-aggregate the stored multi-path candidates under the alternative strategies in `utils/ensemble_strategies.py`.

---

## 5. Citation

```bibtex
@article{wu2025preba,
  title   = {PREBA: Surgical Duration Prediction via PCA-Weighted Retrieval-Augmented
             LLMs and Bayesian Averaging Aggregation},
  author  = {Wu, Wanyin and Li, Kanxue and Yu, Baosheng and Zhao, Haoyun and
             Zhan, Yibing and Tao, Dapeng and Jin, Hua},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2025},
  note    = {Under review}
}
```

## 6. License

Released for academic research. See `LICENSE` for details.
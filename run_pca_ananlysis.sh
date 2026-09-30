#!/bin/bash

# 对 MOVER_SIS 的 all(train+val) 做 PCA 分析
# CUDA_VISIBLE_DEVICES=2 python pca_feature_analysis_mover.py \
#   --dataset_name MOVER_SIS \
#   --split all \
#   --n_components 16 \
#   --output_json pca_mover_sis_importance.json

# MOVER_EPIC 的 all(train+val) 做 PCA 分析
# CUDA_VISIBLE_DEVICES=2 python pca_feature_analysis_mover.py \
#   --dataset_name MOVER_EPIC \
#   --split all \
#   --n_components 50 \
#   --output_json pca_mover_epic_importance.json


# # INSPIRE 的 all(train+val) 做 PCA 分析
# CUDA_VISIBLE_DEVICES=2 python pca_feature_analysis_mover.py \
#   --dataset_name INSPIRE \
#   --split all \
#   --n_components 50 \
#   --output_json pca_inspire_importance.json

# # Multimodal_SDP 的 all(train+val) 做 PCA 分析
# CUDA_VISIBLE_DEVICES=2 python pca_feature_analysis_mover.py \
#   --dataset_name Multimodal_SDP \
#   --split all \
#   --n_components 100 \
#   --output_json pca_mmsdp_importance.json

# # INSPIRE 的 all(train+val) 做 PCA 分析
CUDA_VISIBLE_DEVICES=2 python pca_feature_analysis_mover.py \
  --dataset_name OursSDP \
  --split all \
  --n_components 50 \
  --output_json pca_ourssdp_importance.json

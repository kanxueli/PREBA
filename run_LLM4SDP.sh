#!/bin/bash

# 默认参数
PORT=8000
MODEL="llama3"
API_KEY=""
MAX_WORKERS=4
USE_RAG=""
RAG_K=5
FORCE_REBUILD_RAG=""
FORCE_REENCODE=""
INDEX_TYPE="Flat"
GPU_ID=0
Dataset_Name=""
RAG_WEIGHT_SCHEME="mutual_info"
RAG_WEIGHT_JSON=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            shift 2
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --use_rag)
            USE_RAG="--use_rag"
            shift
            ;;
        --rag_k)
            RAG_K="$2"
            shift 2
            ;;
        --dataset_name)
            Dataset_Name="$2"
            shift 2
            ;;
        --enable_prior_hint)
            ENABLE_PRIOR_HINT="--enable_prior_hint"
            shift
            ;;
        --force_rebuild_rag)
            FORCE_REBUILD_RAG="--force_rebuild_rag"
            shift
            ;;
        --force_reencode)
            FORCE_REENCODE="--force_reencode"
            shift
            ;;
        --index_type)
            INDEX_TYPE="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --ran_path)
            RAN_PATH="$2"
            shift 2
            ;;
        --use_grouped_encoding)
            USE_GROUPED_ENCODING="--use_grouped_encoding"
            shift
            ;;
        --rag_weight_scheme)
            RAG_WEIGHT_SCHEME="$2"
            shift 2
            ;;
        --rag_weight_json)
            RAG_WEIGHT_JSON="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --port PORT      API服务器端口号 (默认: 8000)"
            echo "  --model MODEL    模型名称: llama3 或 qwen3 (默认: llama3)"
            echo "  --api_key KEY    API密钥 (默认: 从环境变量获取)"
            echo "  --max_workers N  并发线程数 (默认: 4)"
            echo "  --use_rag        启用RAG检索增强模式"
            echo "  --rag_k N        RAG检索案例数量 (默认: 5)"
            echo "  --force_rebuild_rag 强制重建RAG数据库"
            echo "  --force_reencode 强制重新编码特征，忽略缓存"
            echo "  --index_type INDEX_TYPE 索引类型: IVFFlat 或 Flat (默认: IVFFlat)"
            echo "  --gpu ID         指定GPU ID (例如: 5)"
            echo "  --ran_path PATH  RAN模型路径（可选）"
            echo "  --use_grouped_encoding  使用分组编码（按权重分组编码BERT特征，默认: 独立编码）"
            echo "  --rag_weight_scheme NAME  RAG特征权重: mutual_info(默认)/paper_pca/pca/uniform/ridge_coef/permutation/rf_mdi/shap"
            echo "  --rag_weight_json PATH    可选，直接指定 weights.json"
            echo "  -h, --help       显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --port 8001 --model qwen3 --max_workers 8"
            echo "  $0 --port 8000 --model llama3 --api_key your_key --max_workers 6"
            echo "  $0 --port 8000 --model llama3 --use_rag --rag_k 5"
            echo "  $0 --port 8000 --model llama3 --use_rag --force_rebuild_rag"
            echo "  $0 --port 8000 --model llama3 --use_rag --gpu 5"
            echo "  $0 --use_rag --dataset_name INSPIRE --rag_weight_scheme mutual_info --force_rebuild_rag"
            echo "  $0 --use_rag --dataset_name INSPIRE --rag_weight_scheme paper_pca"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 构建命令
if [ -n "$GPU_ID" ]; then
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python models/LLM4SDP.py --port $PORT --model $MODEL --max_workers $MAX_WORKERS"
else
    CMD="python models/LLM4SDP.py --port $PORT --model $MODEL --max_workers $MAX_WORKERS"
fi
#
if [ -n "$API_KEY" ]; then
    CMD="$CMD --api_key $API_KEY"
fi

# RAG相关初始化
if [ -n "$USE_RAG" ]; then
    CMD="$CMD $USE_RAG --rag_k $RAG_K --index_type $INDEX_TYPE"
fi

# 数据集
if [ -n "$Dataset_Name" ]; then
    CMD="$CMD --dataset_name $Dataset_Name"
fi

# 先验提示相关初始化
if [ -n "$ENABLE_PRIOR_HINT" ]; then
    CMD="$CMD --enable_prior_hint"
fi

# 强制重建RAG数据库
if [ -n "$FORCE_REBUILD_RAG" ]; then
    CMD="$CMD $FORCE_REBUILD_RAG"
fi

# 强制重新编码特征，忽略缓存
if [ -n "$FORCE_REENCODE" ]; then
    CMD="$CMD $FORCE_REENCODE"
fi

# RAN模型路径
if [ -n "$RAN_PATH" ]; then
    CMD="$CMD --ran_path $RAN_PATH"
fi

# 分组编码选项
if [ -n "$USE_GROUPED_ENCODING" ]; then
    CMD="$CMD $USE_GROUPED_ENCODING"
fi

if [ -n "$RAG_WEIGHT_SCHEME" ]; then
    CMD="$CMD --rag_weight_scheme $RAG_WEIGHT_SCHEME"
fi
if [ -n "$RAG_WEIGHT_JSON" ]; then
    CMD="$CMD --rag_weight_json $RAG_WEIGHT_JSON"
fi

# 执行命令
eval $CMD
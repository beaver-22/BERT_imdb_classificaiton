CONFIG_PATH="configs/modern_bert.yaml"

export CUDA_VISIBLE_DEVICES=0

python src/main.py --config $CONFIG_PATH
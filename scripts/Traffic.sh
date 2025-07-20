root_path_name=./data/traffic/
data_path_name=traffic.csv
model_id_name=traffic
data_name=traffic

seed=2024

seq_len=96
# export CUDA_VISIBLE_DEVICES=2
for pred_len in 96 192 336 720
do
  python -u main.py \
    --seed $seed \
    --data $root_path_name$data_path_name \
    --feature_type M \
    --target OT \
    --checkpoint_dir ../checkpoints \
    --name $model_id_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --moe_k 2 \
    --patch_config [(16, 8), (24, 12), (32, 16), (48, 24)] \
    --loss_coef 0.01 \
    --norm True \
    --layernorm False \
    --dropout 0.1 \
    --train_epochs 10 \
    --batch_size 2 \
    --learning_rate 0.00008 \
    --result_path result.csv
done

#!/bin/bash

tasks=("acos" "asqp")
for task_name in "${tasks[@]}"; do
    if [ "$task_name" = "acos" ]; then
        datasets=("laptop16" "rest16")
    else
        datasets=("rest15" "rest16")
    fi

    for dataset in "${datasets[@]}"; do
        for ratio in 0.01 0.05 0.1; do
          for top_k in $(seq 5 5); do  # 修改这里以直接使用 5
              for seed in 3407; do
                  echo "Current task: $dataset"

                  export per_device_train_batch_size=16
                  export per_device_eval_batch_size=16
                  export epochs=200
                  export save_steps=80
                  export eval_steps=80
                  export learning_rate=1e-4
                  export alpha=0.5
                  export constraint_decoding=True
                  export model_name_or_path=/mnt/sda3/ztj/pretrained-models/t5-base
                  export cls_model_name_or_path=/mnt/sda3/ztj/pretrained-models/bert-base-uncased
                  export save_strategy=steps
                  export evaluation_strategy=steps
                  export shot_ratio_index="-1[+]${ratio}[+]0"
                  export warmup_ratio=0
                  export load_best_model_at_end=True
                  export combined=True
                  export virtual_token=True
                  export implicit_token=True
                  export ctrl_token=front
                  export CUDA_VISIBLE_DEVICES=1
                  export output_dir=./outputs/${task_name}/${dataset}_seed${seed}_epoch${epochs}_bts${per_device_train_batch_size}_lr${learning_rate}_topk${top_k}_ratio${ratio}
                  export HF_DATASETS_OFFLINE=1
                  export TRANSFORMERS_OFFLINE=1

                  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python ./main.py \
                      --do_train True \
                      --do_predict True \
                      --predict_with_generate \
                      --overwrite_output_dir \
                      --model_name_or_path=${model_name_or_path} \
                      --cls_model_name_or_path=${cls_model_name_or_path} \
                      --task_name=${task_name} \
                      --dataset=${dataset} \
                      --seed=${seed} \
                      --output_dir=${output_dir} \
                      --per_device_train_batch_size=${per_device_train_batch_size} \
                      --per_device_eval_batch_size=${per_device_eval_batch_size} \
                      --learning_rate=${learning_rate} \
                      --num_train_epochs=${epochs} \
                      --save_strategy=${save_strategy} \
                      --save_steps=${save_steps} \
                      --lr_scheduler_type=linear \
                      --constraint_decoding ${constraint_decoding} \
                      --alpha ${alpha} \
                      --use_fast_tokenizer \
                      --evaluation_strategy=${evaluation_strategy} \
                      --eval_steps=${eval_steps} \
                      --load_best_model_at_end=${load_best_model_at_end} \
                      --metric_for_best_model eval_f1_score \
                      --save_total_limit 1 \
                      --warmup_ratio=${warmup_ratio} \
                      --shot_ratio_index=${shot_ratio_index} \
                      --lowercase True \
                      --sort_label True \
                      --max_length 256 \
                      --topk=${top_k} \
                      --ctrl_token ${ctrl_token} \
                      --combined ${combined} \
                      --virtual_token ${virtual_token} \
                      --implicit_token ${implicit_token} \
              done
          done
        done
    done
done

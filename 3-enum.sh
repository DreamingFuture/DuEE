CUDA_VISIBLE_DEVICES=6 python run_cls.py \
    --dataset DuEE-Fin \
    --event_type enum \
    --max_len 400 \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --per_gpu_eval_batch_size 12 \
    --learning_rate 1e-5 \
    --linear_learning_rate 5e-5 \
    --early_stop 2
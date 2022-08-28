CUDA_VISIBLE_DEVICES=1 python predict_ner.py \
--dataset DuEE1.0 \
--event_type role \
--max_len 250 \
--per_gpu_eval_batch_size 128 \
--model_name_or_path /data/qingyang/data/chinese-roberta-wwm-ext \
--fine_tunning_model_path ./output/DuEE1.0/role/best_model.pkl \
--test_json ./data/DuEE1.0/subor_news.json
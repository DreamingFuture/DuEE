cuda=5
model=role
# 预测trigger和role
CUDA_VISIBLE_DEVICES=$cuda python predict_ner.py \
--dataset DuEE1.0 \
--event_type role \
--max_len 250 \
--per_gpu_eval_batch_size 128 \
--model_name_or_path /data/qingyang/data/chinese-roberta-wwm-ext \
--fine_tunning_model_path ./output/DuEE1.0/${model}/best_model.pkl \
--test_json ./output/DuEE1.0/DataPredictionRes/surbot_news.json \
--predict_save_path ./output/DuEE1.0/${model}/test_result.json

# 合并预测结果，输出预测文件
python duee_1_my_postprocess.py \
--trigger_file ./output/DuEE1.0/${model}/test_result.json \
--role_file ./output/DuEE1.0/${model}/test_result.json \
--schema_file ./conf/DuEE1.0/event_schema.json \
--save_path ./output/DuEE1.0/surbot_news_together.json
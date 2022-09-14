# 更改测试数据，输入test文件名，输入是 素问接口文件 输出是 id-text 的json文件
# 输入输出文件名变成arg
input_name=news_1w.jsonl
test_name=duee_surbot.json
# echo "********开始数据预处理********"
# python /data/qingyang/event_extration/DuEE_merge/data/DuEE1.0/our_test.py \
# 	--input_name $input_name \
# 	--output_name $test_name

# 运行数据预处理程序
# 生成conf下的dict文件（trigger和role的都要是最新类别数
# 生成data下的test/train/dev.csv文件，是BIO标注的数据，作为输入数据输入到模型中
# test的输入（ourtest的输出文件）变成arg
echo "********开始处理tag.dict 和 tsv文件********"
python duee_1_data_prepare.py \
	--test_inputName $test_name

echo "********开始预测trigger和role********"
cuda=6
model=role
# 预测trigger和role
CUDA_VISIBLE_DEVICES=$cuda python predict_ner.py \
--dataset DuEE1.0 \
--event_type role \
--max_len 250 \
--per_gpu_eval_batch_size 128 \
--model_name_or_path /data/qingyang/data/chinese-roberta-wwm-ext \
--fine_tunning_model_path ./output/DuEE1.0/${model}/best_model.pkl \
--test_json ./data/DuEE1.0/${test_name} \
--predict_save_path ./output/DuEE1.0/${model}/test_result.json

# 合并预测结果，输出预测文件
echo "********合并预测结果，输出预测文件********"
python duee_1_my_postprocess.py \
--trigger_file ./output/DuEE1.0/${model}/test_result.json \
--role_file ./output/DuEE1.0/${model}/test_result.json \
--schema_file ./conf/DuEE1.0/event_schema.json \
--save_path ./output/DuEE1.0/duee_surbot.json

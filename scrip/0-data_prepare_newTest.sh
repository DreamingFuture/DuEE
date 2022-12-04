# 更改测试数据，输入test文件名，输入是 素问接口文件 输出是 id-text 的json文件
# 输入输出文件名变成arg

echo "开始数据预处理"
input_name=news_5w.jsonl
test_name=subor_news.json
# python /data/qingyang/event_extration/DuEE_merge/data/DuEE1.0/our_test.py \
# 	--input_name $input_name \
# 	--output_name $test_name

# 运行数据预处理程序
# 生成conf下的dict文件（trigger和role的都要是最新类别数
# 生成data下的test/train/dev.csv文件，是BIO标注的数据，作为输入数据输入到模型中
# test的输入（ourtest的输出文件）变成arg
echo "开始处理tag.dict 和 tsv文件"
python ../duee_1_data_prepare.py \
	--test_inputName $test_name

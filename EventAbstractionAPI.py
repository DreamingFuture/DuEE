"""
1. 先进行数据的预处理
2. 在进行数据的预测
3. 最后进行数据的后处理
"""

import os
from data.data_utils import schema_process, data_process
from utils.utils import write_by_lines
import argparse
import json
import re
import jiagu
import jsonlines
import predict_ner
import duee_1_my_postprocess


def transform_data_2_need_type(input_path:str, use_content:bool):
    print("\n=================Transform Data to Need Type==============")
    print("\n=================use_content:{}==============".format(use_content))
    res = []
    id1 = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            content = json.loads(content)
            res.append({'text':content['title'], 'id':f'id-{id1}-0', 'id_new':content['news_id']})
            if use_content:
                try:
                    sentences = jiagu.summarize(content['content'].replace('<br>', ''), 2)
                except IndexError:
                    sentences = []
                for i, sentence in enumerate(sentences):
                    res.append({'text':sentence, 'id':f'id-{id1}-{i+1}', 'id_new':content['news_id']})

            content = f.readline()
            id1 += 1
            if id1 % 500 == 499:
                print('id:',id1,' done!')
    print('id:',id1,' done!')

    import jsonlines
    with jsonlines.open(input_path, 'w') as f:
        for item in res:
            f.write(item)
    print("=================end transformation process==============")   


def process_labels(labels_list):
    for i in range(len(labels_list)):
        labels_list[i] = labels_list[i].split('\t')[1].split('\002')
    return labels_list


def data_prepare(input_path:str):
    print("\n=================DUEE 1.0 DATASET==============")
    conf_dir = "./conf/DuEE1.0"
    schema_path = "{}/event_schema.json".format(conf_dir)
    tags_trigger_path = "{}/trigger_tag.dict".format(conf_dir)
    tags_role_path = "{}/role_tag.dict".format(conf_dir)
    print("\n=================start schema process==============")
    print('input path {}'.format(schema_path))
    tags_trigger = schema_process(schema_path, "trigger")[:-1] # 去掉 'O'
    
    tags_role = schema_process(schema_path, "role", trigger_num=len(tags_trigger))
    # 把trigger的类型和role的拼接在一起，
    tags_role = tags_trigger + tags_role
    # 然后写入到trigger和role的dict中
    # write_by_lines(tags_trigger_path, tags_trigger)
    # print("save trigger tag {} at {}".format(len(tags_trigger), tags_trigger_path))
    write_by_lines(tags_role_path, tags_role)
    print("save trigger tag {} at {}".format(len(tags_role), tags_role_path))
    print("=================end schema process===============")

    # data process
    data_dir = "./data/DuEE1.0"
    trigger_save_dir = "{}/trigger".format(data_dir)
    role_save_dir = "{}/role".format(data_dir)

    print("\n=================start sentence process==============")
    if not os.path.exists(trigger_save_dir):
        os.makedirs(trigger_save_dir)
    if not os.path.exists(role_save_dir):
        os.makedirs(role_save_dir)

    print("\n----trigger------for dir {} to {}".format(data_dir, trigger_save_dir))
    train_tri = data_process("{}/duee_train.json".format(data_dir), "trigger", type="duee1")
    write_by_lines("{}/train.tsv".format(trigger_save_dir), train_tri)
    dev_tri = data_process("{}/duee_dev.json".format(data_dir), "trigger", type="duee1")
    write_by_lines("{}/dev.tsv".format(trigger_save_dir), dev_tri)
    # 测试集的数据改成tsv要改这里
    test_tri = data_process(input_path, "trigger", type="duee1")
    write_by_lines("{}/test.tsv".format(trigger_save_dir), test_tri)
    print("train {} dev {} test {}".format(
        len(train_tri), len(dev_tri), len(test_tri)))

    # 训练集和测试集的标注结果，会输入到role中，在trigger标注的基础上继续标注role
    train_tri = process_labels(train_tri)[1:]
    dev_tri = process_labels(dev_tri)[1:]


    print("\n----role------for dir {} to {}".format(data_dir, role_save_dir))
    # 传入trigger的训练集的标注结果
    train_role = data_process("{}/duee_train.json".format(data_dir), "role", type="duee1", labels_list=train_tri)
    write_by_lines("{}/train.tsv".format(role_save_dir), train_role)
    # 传入role的测试集的标注结果
    dev_role = data_process("{}/duee_dev.json".format(data_dir), "role", type="duee1", labels_list=dev_tri)
    write_by_lines("{}/dev.tsv".format(role_save_dir), dev_role)
    # 测试集的数据改成tsv要改这里
    test_role = data_process(input_path, "role", type="duee1", is_predict=True)
    write_by_lines("{}/test.tsv".format(role_save_dir), test_role)
    print("train {} dev {} test {}".format(len(train_role), len(dev_role), len(test_role)))
    print("=================end schema process==============")

    
def event_abstraction_API(input_path:str, output_path:str, use_content:bool = True) -> bool:
    """ 
    1. 先进行数据的预处理
    2. 在进行数据的预测
    3. 最后进行数据的后处理
    """
    predict_save_path = "./output/DuEE1.0/role/test_result.json"
    # 摘要提取
    # transform_data_2_need_type(input_path=input_path, use_content=use_content)

    # 数据预处理
    # data_prepare(input_path)

    #进行数据的预测
    predict_ner.main(input_path=input_path, predict_save_path=predict_save_path)

    # 预测结果合并
    duee_1_my_postprocess.main(predict_save_path=predict_save_path, output_path=output_path)


if __name__ == "__main__":
    input_path = "API_TEST/test.jsonl"
    output_path = "API_TEST/res.jsonl"
    use_content = True
    res = event_abstraction_API(input_path=input_path, output_path=output_path, use_content=use_content)
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""duee 1.0 dataset process"""
import os
from data.data_utils import schema_process, data_process
from utils.utils import write_by_lines
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_inputName', type=str, default='surbot_news.json')

args = parser.parse_args()

def process_labels(labels_list):
    for i in range(len(labels_list)):
        labels_list[i] = labels_list[i].split('\t')[1].split('\002')
    return labels_list

if __name__ == "__main__":
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
    test_tri = data_process("{}/{}".format(data_dir, args.test_inputName), "trigger", type="duee1")
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
    test_role = data_process("{}/{}".format(data_dir, args.test_inputName), "role", type="duee1", is_predict=True)
    write_by_lines("{}/test.tsv".format(role_save_dir), test_role)
    print("train {} dev {} test {}".format(len(train_role), len(dev_role), len(test_role)))
    print("=================end schema process==============")

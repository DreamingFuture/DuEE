import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast

from dataset.dataset import collate_fn, DuEEEventDataset
from model.model import DuEEEvent_model
# from utils.finetuning_argparse import get_argparse
import argparse

from utils.utils import init_logger, seed_everything, logger, read_by_lines, write_by_lines, cnt_time

@cnt_time 
def main(input_path:str = None, predict_save_path:str = None):
    parser = argparse.ArgumentParser()
    # 数据
    parser.add_argument("--dataset", type=str, default="DuEE1.0", help="train data")
    parser.add_argument("--event_type", type=str, default="role", help="dev data")
    parser.add_argument("--max_len", default=250, type=int, help="最大长度")
    parser.add_argument("--stride", type=int, default=100, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--overwrite_cache",  default=False, help="")

    #
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="训练Batch size的大小")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="训练Batch size的大小")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int, help="验证Batch size的大小")

    # 训练的参数
    parser.add_argument("--do_distri_train", default=False, help="是否用两个卡并行训练")
    parser.add_argument("--model_name_or_path", default="/data/qingyang/data/chinese-roberta-wwm-ext", type=str, help="预训练模型的路径")
    parser.add_argument("--num_train_epochs", default=50.0, type=float, help="训练轮数")
    parser.add_argument("--early_stop", default=8, type=int, help="早停")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="transformer层学习率")
    parser.add_argument("--linear_learning_rate", default=1e-3, type=float, help="linear层学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")
    parser.add_argument("--output_dir", default="./output", type=str, help="保存模型的路径")
    parser.add_argument("--fine_tunning_model_path",
                        type=str,
                        default="./output/DuEE1.0/role/best_model.pkl",
                        help="fine_tuning model path")
    if input_path:
        parser.add_argument("--test_json",
                        type=str,
                        default=f"{input_path}",
                        help="test json path")
    else:
        parser.add_argument("--test_json",
                        type=str,
                        required=True,
                        help="test json path")
    parser.add_argument("--predict_save_path",
                        type=str,
                        default=f"{predict_save_path}" if predict_save_path else "",
                        help="prediction输出位置"
                        )
    args = parser.parse_args()

    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    init_logger(log_file="./log/{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)

    # args.output_model_path = os.path.join(args.output_dir, args.dataset, args.event_type, "best_model.pkl")
    # # 设置保存目录
    # if not os.path.exists(os.path.dirname(args.output_model_path)):
    #     os.makedirs(os.path.dirname(args.output_model_path))

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)

    # dataset & dataloader
    args.test_data = "./data/{}/{}/test.tsv".format(args.dataset, args.event_type)
    args.tag_path = "./conf/{}/{}_tag.dict".format(args.dataset, args.event_type)
    test_dataset = DuEEEventDataset(args,
                                   args.test_data,
                                   args.tag_path,
                                   tokenizer)
    logger.info("The nums of the test_dataset features is {}".format(len(test_dataset)))
    test_iter = DataLoader(test_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=collate_fn,
                           num_workers=1)

    # load data from predict file
    sentences = read_by_lines(args.test_json)  # origin data format
    sentences = [json.loads(sent) for sent in sentences]

    # 用于evaluate
    args.label2it = test_dataset.label_vocab
    args.id2label = {val: key for key, val in args.label2it.items()}
    args.num_classes = len(args.id2label)

    #
    model = DuEEEvent_model(args.model_name_or_path, num_classes=args.num_classes)
    model.to(args.device)
    model.load_state_dict(torch.load(args.fine_tunning_model_path))

    results = []
    model.eval()
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_iter), total=len(test_iter)):
            for key in batch.keys():
                batch[key] = batch[key].to(args.device)
            logits = model(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )
            probs = F.softmax(logits, dim=-1).cpu()
            probs_ids = torch.argmax(probs, -1).numpy()
            probs = probs.numpy()
            seq_lens = batch["all_seq_lens"]
            for p_list, p_ids, seq_len in zip(probs.tolist(), probs_ids.tolist(), seq_lens.tolist()):
                prob_one = [p_list[index][pid] for index, pid in enumerate(p_ids[1: seq_len - 1])]
                label_one = [args.id2label[pid] for pid in p_ids[1: seq_len - 1]]
                results.append({"probs": prob_one, "labels": label_one})
    print(len(results))
    print(len(sentences))
    assert len(results) == len(sentences)
    for sent, ret in zip(sentences, results):
        sent["pred"] = ret
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    if not args.predict_save_path:
        args.predict_save_path = os.path.join("./output", args.dataset, args.event_type, "test_result.json")
    print("saving data {} to {}".format(len(sentences), args.predict_save_path))
    write_by_lines(args.predict_save_path, sentences)


if __name__ == '__main__':
    main()

# 该py用于串联整个工作流: finetune --> inference --> BLEU4 --> Pass Rate

import os
import json
import time
import logging
import argparse
import subprocess
import basic
import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import joblib
import ast

from utils import create_directory_if_not_exists

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def is_tmux_running(tmux_name):
    """
    Summary:该函数用于检查tmux是否存在,存在则关闭
    """
    try:
        result = subprocess.run(["tmux", "ls"], capture_output=True, text=True, check=True)
        return tmux_name in result.stdout
    except subprocess.CalledProcessError:
        print("没有找到 tmux 会话。")
        return False


def finetune(args):
    """
    Summary:该函数用于设置微调环境与开启微调
    """

    # 若finetune会话存在，则首先关闭
    if is_tmux_running(f"{args.lora_name}"):
        subprocess.run(["tmux", "kill-session", "-t", f"{args.lora_name}"])

    # 创建名为finetune的tmux会话
    tmux_command = f"tmux new-session -d -s {args.lora_name}"
    subprocess.run(tmux_command, shell=True)
    logging.info(f"已经创建名为{args.lora_name}的tmux窗口")

    # 设定微调命令
    nccl_command = "export NCCL_P2P_LEVEL=NVL"  #
    gpu_command = f"export CUDA_VISIBLE_DEVICES={args.gpu}"  #
    path_command = "cd ./Kcatllamafactory"
    activate_command = "conda activate KcatLF"
    finetune_command = f"llamafactory-cli train ./finetune/{args.lora_name}/finetune.yaml"
    full_command = f"{nccl_command} && {gpu_command} && {path_command} && {activate_command} && {finetune_command} && echo 'Training complete' && exit"

    tmux_exec_command = f"tmux send-keys -t {args.lora_name} '{full_command}' C-m"
    logging.info("使用llamafactory开启微调")
    start_time = time.time()
    subprocess.run(tmux_exec_command, shell=True)

    while is_tmux_running(f"{args.lora_name}"):
        time.sleep(10)
    end_time = time.time()
    logging.info(f"微调结束，用时:{end_time - start_time}s")


def inference(args):
    """
    Summary:该函数用于完成inference与计算PCC分数
    """
    input_file = "./Kcatllamafactory/data/kcat_km_test.json"
    output_file = f"./results_llamafactory/{args.lora_name}/generated_predictions.jsonl"
    output_vector = f"./results_llamafactory/{args.lora_name}/feature_vectors.npy"
    model_path = "./model/gemma-2-9b-it"
    peft_model = f"./Kcatllamafactory/saves/{args.lora_name}"
    gpu = "cuda:0"
    start_time = time.time()
    basic.inference.inference(input_file, output_file, model_path, peft_model, gpu, output_vector, logging)
    end_time = time.time()
    logging.info(f"推理结束，用时:{end_time - start_time}s")


def read_predictions(inference_file):
    kcat_true = []
    km_true = []
    kcat_pred = []
    km_pred = []

    with open(inference_file, "r", encoding="utf-8") as file:
        line_number = 0  # 追踪行号，帮助定位问题
        for line in file:
            line_number += 1
            try:
                data = json.loads(line)
                predict = ast.literal_eval(data["predict"])
                label = ast.literal_eval(data["label"])

                kcat_pred.append(float(predict[0]))
                km_pred.append(float(predict[1]))
                kcat_true.append(float(label[0]))
                km_true.append(float(label[1]))
            except json.JSONDecodeError as e:
                logging.error(f"JSON 解析错误在文件 {inference_file} 的第 {line_number} 行: {e}")
                continue
    return kcat_pred, km_pred, kcat_true, km_true


def calPCC_score(args):
    """
    该函数用于计算LLM输出与真实答案之间的皮尔逊相关系数
    """
    input_file = f"./results_llamafactory/{args.lora_name}/generated_predictions.jsonl"
    kcat_pred, km_pred, kcat_true, km_true = read_predictions(input_file)

    pearson_kcat = pearsonr(kcat_pred, kcat_true)
    pearson_km = pearsonr(km_pred, km_true)
    logging.info(f"计算PCC结束, 不逆变换情况下,kcat皮尔逊相关系数为: {pearson_kcat[0]}, p-value为:{pearson_kcat[1]}")
    logging.info(f"计算PCC结束, 不逆变换情况下,km皮尔逊相关系数为: {pearson_km[0]}, p-value为: {pearson_km[1]}")

    # 逆变换
    scaler = joblib.load('/home/zhangyangyu/Kcat_predict/dataset/kcat_km_scaler.save')  # 改
    predictions = np.column_stack((kcat_pred, km_pred))
    trues = np.column_stack((kcat_true, km_true))
    predictions_inverse = 10 ** (scaler.inverse_transform(predictions))
    trues_inverse = 10 ** (scaler.inverse_transform(trues))

    pearson_kcat_inverse = pearsonr(predictions_inverse[:, 0], trues_inverse[:, 0])
    pearson_km_inverse = pearsonr(predictions_inverse[:, 1], trues_inverse[:, 1])
    logging.info(
        f"计算PCC结束, 逆变换后,kcat皮尔逊相关系数为: {pearson_kcat_inverse[0]}, p-value为:{pearson_kcat_inverse[1]}")
    logging.info(
        f"计算PCC结束, 逆变换后,km皮尔逊相关系数为: {pearson_km_inverse[0]}, p-value为: {pearson_km_inverse[1]}")


def main(args):
    """
    Summary:该函数用于使用llamafactory完成finetune
    """

    # 工作流开始
    logging.info("工作流begin!")
    logging.info("\n")

    # # 是否开启finetune
    # if not args.finetune_flag:
    #     logging.info("Part: Finetune")
    #     finetune(args)
    #     logging.info("\n")
    # else:
    #     logging.info("Part: Finetune")
    #     logging.info("不执行微调")
    #     logging.info("\n")

    # 首先进行最终checkpoint的推理
    logging.info("Part: Final Inference")
    inference(args)
    logging.info("\n")

    # 计算PCC分数
    logging.info("Part: PCC")
    calPCC_score(args)
    logging.info("\n")

    # 工作流结束
    logging.info("工作流over!")


def load_llm_train_test_dataset():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    create_directory_if_not_exists(os.path.join(path, "dataset"))
    df = pd.read_csv(os.path.join(args.input_dir, "dataset/input.tsv"), sep="\t", encoding="utf-8")


def load_llm_inference_dataset():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    create_directory_if_not_exists(os.path.join(path, "dataset"))
    df = pd.read_csv(os.path.join(args.input_dir, "dataset/input.tsv"), sep="\t", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="predict/Synechocystis sp")  # .
    parser.add_argument("--finetune_flag", action="store_true")
    parser.add_argument("--lora_name", type=str, default="kcat_km_epoch100_gemma")
    parser.add_argument("--gpu", type=str, default="0,1,2,3")

    args = parser.parse_args()

    basic.file_directory.create_directory_if_not_exists(f"./results_llamafactory/{args.lora_name}/record.log")
    logging.basicConfig(
        filename=f"./results_llamafactory/{args.lora_name}/record.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    main(args)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--finetune_flag", action="store_true")
#     parser.add_argument("--finetune_gpu", type=str, default="0,1,2,3,4,5,6,7")
#     parser.add_argument("--inference_gpu", type=str, default="cuda:0")
#     parser.add_argument("--lora_name", type=str, default="kcat_km_epoch1000_all_llama3")

#     for checkpoint in [1000,2000,3000,4000,5000,6000,8000,10000,12000,14000,16000]:
#        # 更新 lora_name 参数
#         lora_name = f"kcat_km_epoch1000_all_llama3/checkpoint-{checkpoint}"
#         args = parser.parse_args(args=["--lora_name", lora_name])

#         basic.file_directory.create_directory_if_not_exists(f"./results_llamafactory/{args.lora_name}/record.log")
#         # 移除所有处理器以允许重新配置 logging
#         for handler in logging.root.handlers[:]:
#             logging.root.removeHandler(handler)

#         logging.basicConfig(
#             filename=f"./results_llamafactory/{args.lora_name}/record.log",
#             level=logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(message)s",
#             filemode="w"
#         )

#         main(args)

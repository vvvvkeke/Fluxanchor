import argparse
import os
import pickle
import re
import subprocess
import sys
sys.path.append("./")
sys.path.append("../")
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import json
import time
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from basic.inference import inference
from utils import create_directory_if_not_exists, inverse_transformation
import logging
import ast
def dataframe_to_json(df, json_file_path, is_dataset=False):
    # and km   ;and the KM units are mol/L;  I need two values: the first for kcat and the second for km.
    data = []
    count = 0
    for index, row in df.iterrows():
        count = count + 1
        instruction = "You are a world-renowned biochemistry expert with over 20 years of research experience in enzyme " \
                      "kinetics and protein engineering. Please predict the kinetic parameters for the enzyme based on the information I will provide. " \
                      "Note: The values of kcat and km provided here are scaled, the original kcat units are s^{-1} and the KM units are mol/L . " \
                      "They have been logarithmically transformed and normalized. I need two values: the first for kcat and the second for km. " \
                      "Please provide the predictions in this scaled and transformed format."
        input_data = {
            # "The EC_number": row["EC_number"],
            # "sequence": row["sequence"],
            # "substrate":row["substrate"],
            # "product":row["product"],
            "reaction": row["reaction"],
            "type": row["type"],
            "ph": row["ph"],
            "temperature": row["temperature"],
            "organism": row["organisms"],
            # "smiles_substrate": row["smiles_substrate"],
            # "smiles_product": row["smiles_product"],
            "smiles_reaction": row["smiles_reaction"],
        }
        # 将字典转换为 JSON 字符串
        input_data_str = json.dumps(input_data)
        if is_dataset:
            output_data = [
                row["kcat"],
                row["km"]
            ]
            output_data_str = json.dumps(output_data)
            data.append({
                "instruction": instruction,
                "input": input_data_str,
                "output": output_data_str,
                "uniprot_id": row["uniprot_id"],
                "index": index
            })
        else:
            data.append({
                "instruction": instruction,
                "input": input_data_str,
                "uniprot_id": row["uniprot_id"],
                "index": index
            })
    # 写入 JSON 文件
    with open(json_file_path, mode='w', encoding='utf-8') as jsonf:
        json.dump(data, jsonf, indent=4)

    print(f"JSON dataset has been created successfully in {json_file_path}.")
    print(f"一共有{count}条数据")

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
    # activate_command = "conda activate KcatLF"
    finetune_command = f"llamafactory-cli train ./finetune/{args.lora_name}/finetune.yaml"
    # full_command = f"{nccl_command} && {gpu_command} && {path_command} && {activate_command} && {finetune_command} && echo 'Training complete' && exit"
    full_command = f"{nccl_command} && {gpu_command} && {path_command} && {finetune_command} && echo 'Training complete' && exit"

    tmux_exec_command = f"tmux send-keys -t {args.lora_name} '{full_command}' C-m"
    logging.info("使用llamafactory开启微调")
    start_time = time.time()
    subprocess.run(tmux_exec_command, shell=True)

    while is_tmux_running(f"{args.lora_name}"):
        time.sleep(10)
    end_time = time.time()
    logging.info(f"微调结束，用时:{end_time - start_time}s")

def inferences(args, name):
    """
    Summary:该函数用于完成inference与计算PCC分数
    """
    # input_file = "../Kcatllamafactory/data/kcat_km_test.json"
    input_file = f"{args.json_output}/Kcatllamafactory/data/{name}.json"
    output_file = f"{args.json_output}/Kcatllamafactory/results_llamafactory/{args.lora_name}/generated_predictions.jsonl"
    output_vector = f"{args.json_output}/Kcatllamafactory/results_llamafactory/{args.lora_name}/feature_vectors.npy"
    model_path = f"{args.json_output}/Kcatllamafactory/model/gemma-2-9b-it"
    peft_model = f"{args.json_output}/Kcatllamafactory/saves/{args.lora_name}"
    gpu = "cuda:0"
    start_time = time.time()
    print(args.is_dataset)
    inference(input_file, output_file, model_path, peft_model, gpu, output_vector, logging, args.is_dataset)
    end_time = time.time()
    logging.info(f"推理结束，用时:{end_time-start_time}s")

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
                label =  ast.literal_eval(data["label"])

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
    input_file = f"../Kcatllamafactory/results_llamafactory/{args.lora_name}/generated_predictions.jsonl"
    kcat_pred, km_pred, kcat_true, km_true = read_predictions(input_file)

    pearson_kcat = pearsonr(kcat_pred, kcat_true)
    pearson_km = pearsonr(km_pred, km_true)
    logging.info(f"计算PCC结束, 不逆变换情况下,kcat皮尔逊相关系数为: {pearson_kcat[0]}, p-value为:{pearson_kcat[1]}")
    logging.info(f"计算PCC结束, 不逆变换情况下,km皮尔逊相关系数为: {pearson_km[0]}, p-value为: {pearson_km[1]}")

    # 逆变换
    with open(os.path.join("./dataset", 'kcat_scaler.pkl'), 'rb') as file:
        kcat_scaler = pickle.load(file)
    with open(os.path.join("./dataset", 'km_scaler.pkl'), 'rb') as file:
        km_scaler = pickle.load(file)
    true_kcats, true_kms, pred_kcats, pred_kms = inverse_transformation(kcat_scaler, km_scaler, kcat_true, km_true, kcat_pred, km_pred)
    # predictions = np.column_stack((kcat_pred, km_pred))
    # trues = np.column_stack((kcat_true, km_true))
    # predictions_inverse = 10 ** (scaler.inverse_transform(predictions))
    # trues_inverse = 10 ** (scaler.inverse_transform(trues))

    pearson_kcat_inverse = pearsonr(pred_kcats, true_kcats)
    pearson_km_inverse = pearsonr(pred_kms, true_kms)
    logging.info(
        f"计算PCC结束, 逆变换后,kcat皮尔逊相关系数为: {pearson_kcat_inverse[0]}, p-value为:{pearson_kcat_inverse[1]}")
    logging.info(
        f"计算PCC结束, 逆变换后,km皮尔逊相关系数为: {pearson_km_inverse[0]}, p-value为: {pearson_km_inverse[1]}")


def get_response(system_content, information, model, tokenizer, gpu):
    """
    Summary:该函数用于向LLM传入query, 返回LLM的response, 作一般的inference使用
    """

    messages = [
        # {
        #     "role": "system",
        #     "content": system_content
        # },
        {
            "role": "user",
            "content": system_content + information
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(gpu)

    with torch.no_grad():
        # 获取输入的隐藏状态
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # 最后一层的隐藏状态

        # 平均输入序列的特征向量，得到整体表示
        last_hidden_state = last_hidden_state.to(torch.float32)
        input_feature_vector = last_hidden_state.mean(dim=1).cpu().numpy()

        # 生成文本响应
        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=4096,
            temperature=0.6,
            top_p=0.9,
        )

    response_generated_ids = []

    for input_ids, output_ids in zip(inputs.input_ids, generated_ids):
        generated_tokens = output_ids[len(input_ids):]
        response_generated_ids.append(generated_tokens)
    response = tokenizer.batch_decode(response_generated_ids, skip_special_tokens=True)[0]
    torch.cuda.empty_cache()
    return response, input_feature_vector

def extract_kcat_km(index,system_content, user_content, model, tokenizer, gpu, i):
    """
    使用正则表达式从模型输出中提取kcat和km的值。
    """
    pattern = r'\[(\d+\.\d+),\s*(\d+\.\d+)\]'
    j = 0
    while j < 10: # 如果未找到匹配项并且尝试次数少于10次，则重试
        response,input_feature_vector = get_response(system_content, user_content, model, tokenizer, gpu)
        match = re.search(pattern, response)
        if match:
            return response,input_feature_vector
        j += 1
        print(f"Retry times: {j} | Response: {response}")
        logging.info(f"第{i}条序列,index为{index}:Retry times: {j} | Response: {response}")
    return str([0, 0]),input_feature_vector

def get_tensor(dataset ,save_dir, lora_name, is_dataset, json_output, resume=False):
    """
    提取LLM特征向量
    Args:
        dataset: 数据集路径
        save_dir: 保存目录
        lora_name: LoRA模型名称
        is_dataset: 是否为训练数据集
        json_output: JSON输出路径
        resume: 是否继续之前的生成（跳过已存在的tensor文件）
    """
    model_path = f"{json_output}/Kcatllamafactory/model/gemma-2-9b-it"
    peft_model = f"{json_output}/Kcatllamafactory/saves/{lora_name}"
    gpu = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=gpu, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if peft_model != "None":
        model = PeftModel.from_pretrained(model, peft_model)
        print(f"加载Lora checkpoint: {peft_model}")
    else:
        raise ValueError("无Lora checkpoint加载")
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    output_json_path = os.path.join(save_dir, f"responses.json")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # 读取JSON文件为列表
    with open(dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 如果是resume模式，统计已存在的tensor数量
    if resume:
        existing_tensors = []
        for i in range(len(dataset)):
            tensor_path = os.path.join(save_dir, f"{i}.tensor")
            if os.path.exists(tensor_path):
                existing_tensors.append(i)
        print(f"Resume模式：发现 {len(existing_tensors)} 个已存在的tensor文件，将跳过这些文件")
        logging.info(f"Resume模式：发现 {len(existing_tensors)} 个已存在的tensor文件，将跳过这些文件")

    # 判断是追加模式还是覆盖模式
    file_mode = "a" if resume else "w"

    with open(output_json_path, file_mode, encoding="utf-8") as output_file:
        processed_count = 0
        skipped_count = 0

        for i, row in enumerate(dataset):
            index = i
            save_path = os.path.join(save_dir, f"{index}.tensor")

            # 检查tensor文件是否已存在
            if resume and os.path.exists(save_path):
                skipped_count += 1
                print(f"跳过第{i + 1}条序列 (tensor文件已存在): {save_path}")
                logging.info(f"跳过第{i + 1}条序列 (tensor文件已存在): {save_path}")
                continue

            print(f"正在处理第{i + 1}条序列 (总共{len(dataset)}条)")
            start = time.time()

            system_content = row["instruction"]
            information = row["input"]
            response, input_feature_vector = extract_kcat_km(index, system_content, information, model, tokenizer,
                                                             gpu, i)
            print(f"Process: {i + 1}/{len(dataset)} | Response: {response}")

            # 保存输入序列的特征向量
            torch.save(torch.tensor(input_feature_vector), save_path)

            if is_dataset:
                # 保存响应数据到JSON文件
                response_data = {
                    "uniprot_id": row['uniprot_id'],
                    "index": row['index'],
                    "label": row['output'],
                    "predict": response
                }
            else:
                # 保存响应数据到JSON文件
                response_data = {
                    "uniprot_id": row['uniprot_id'],
                    "index": row['index'],
                    "predict": response
                }
            json_line = json.dumps(response_data, ensure_ascii=False)
            output_file.write(json_line + "\n")
            output_file.flush()  # 立即写入磁盘，防止丢失

            end = time.time()
            processed_count += 1
            print(f'第{i + 1}条序列处理完成，用时：{end - start:.2f}秒')
            logging.info(f'第{i + 1}条序列处理完成，用时：{end - start:.2f}秒')

    print(f"\n完成！共处理 {processed_count} 条数据，跳过 {skipped_count} 条数据")
    logging.info(f"完成！共处理 {processed_count} 条数据，跳过 {skipped_count} 条数据")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ../predict/rubisco_km
    parser.add_argument("--input_dir", type=str, default="../predict/rubisco_km")  # ../predict/Synechocystis sp
    parser.add_argument("--lora_name", type=str, default="kcat_km_epoch100_gemma/checkpoint-1600")
    parser.add_argument("--gpu", type=str, default="0,1,2,3")
    parser.add_argument("--is_train", type=str, default=False)
    # parser.add_argument("--is_dataset", type=bool, default=False)
    parser.add_argument("--json_output", type=str, default="..")
    parser.add_argument("--resume", action="store_true", help="是否继续之前的生成（跳过已存在的tensor文件）")
    args = parser.parse_args()
    name = args.input_dir.split("/")[-1]
    df = pd.read_csv(os.path.join(args.input_dir, "dataset/input.tsv"), sep="\t", encoding="utf-8")
    if name == "..":
        df['kcat'] = np.log10(df['kcat'])
        df['km'] = np.log10(df['km'])
        dataset_split = np.load('w_dataset_split.npz')
        train_json_file_path = '../Kcatllamafactory/data/kcat_km_train.json'
        val_json_file_path = '../Kcatllamafactory/data/kcat_km_val.json'
        test_json_file_path = '../Kcatllamafactory/data/kcat_km_test.json'
        with open(os.path.join("./dataset", 'kcat_scaler.pkl'), 'rb') as file:
            kcat_scaler = pickle.load(file)
        with open(os.path.join("./dataset", 'km_scaler.pkl'), 'rb') as file:
            km_scaler = pickle.load(file)
        df['kcat'] = kcat_scaler.transform(df[['kcat']])
        df['km'] = km_scaler.transform(df[['km']])
        train_index = dataset_split['train_index']
        train_dataset = df.loc[train_index]
        val_index = dataset_split['val_index']
        val_dataset = df.loc[val_index]
        test_index = dataset_split['test_inedx']
        test_dataset = df.loc[test_index]
        dataframe_to_json(train_dataset, train_json_file_path)
        dataframe_to_json(val_dataset, val_json_file_path)
        dataframe_to_json(test_dataset, test_json_file_path)
        is_dataset = True
    else:
        is_dataset = False
        json_output = f"{args.json_output}/Kcatllamafactory/data/" + name + ".json"
        dataframe_to_json(df, json_output, is_dataset=is_dataset)
    LLM_output = os.path.join(args.input_dir, "dataset/LLM_tensor")
    create_directory_if_not_exists(os.path.join(args.input_dir, "dataset/LLM_tensor"))

    create_directory_if_not_exists(f"{args.json_output}/Kcatllamafactory/results_llamafactory/{args.lora_name}")
    logging.basicConfig(
        filename=f"{args.json_output}/Kcatllamafactory/results_llamafactory/{args.lora_name}/record.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    if args.is_train:
        finetune(args)
    if not os.path.exists(f"{args.json_output}/Kcatllamafactory/results_llamafactory/{args.lora_name}/feature_vectors.npy"):
        inferences(args, name)
    get_tensor(json_output, LLM_output, args.lora_name, is_dataset, args.json_output, args.resume)
    if is_dataset:
        calPCC_score(args)
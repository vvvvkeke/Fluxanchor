import argparse

from Bio.Emboss.Applications import NeedleCommandline
import pandas as pd
import numpy as np
from os.path import join, exists
import os
from tqdm import tqdm  # 导入 tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed  # 导入并行处理模块
import subprocess

from utils import create_directory_if_not_exists


def calculate_identity(fasta_file_1, fasta_file_2):
    needle_cline = NeedleCommandline(asequence=fasta_file_1, bsequence=fasta_file_2,
                                     gapopen=10, gapextend=0.5, filter=True)
    # command = [
    #     "needle",
    #     "-asequence", fasta_file_1,
    #     "-bsequence", fasta_file_2,
    #     "-gapopen", "10",
    #     "-gapextend", "0.5",
    # ]

    try:
        # 使用 subprocess 运行 needle 命令
        result = subprocess.run(str(needle_cline).split(), capture_output=True, text=True, check=True)
        # result = subprocess.run(command, capture_output=True, text=True, check=True)
        stdout = result.stdout  # 0
        out = stdout[stdout.find("Identity"):]
        out = out[:out.find("\n")]
        percent = float(out[out.find("(") + 1:out.find(")") - 1].replace(" ", ""))
        return percent
    except subprocess.CalledProcessError as e:
        print(f"Error running needle: {e}")
        return None  # 返回 None 表示出错


# 存储相似性得分
result_dir = "analysis/split_protein_sequence_identity/rubisco/similarity"
os.makedirs(result_dir, exist_ok=True)


# 计算每个测试序列与训练集中所有序列的相似性
def process_test_sequence(i):
    test_fasta_path = join("analysis/split_protein_sequence_identity/dataset/test_fasta", f"seq_{i}.fasta")
    max_identity = 0

    for j in tqdm(train_indices, desc=f"Processing train sequences for test seq {i}", leave=False):
        train_fasta_path = join("analysis/split_protein_sequence_identity/dataset/train_fasta", f"seq_{j}.fasta")

        # 计算相似性
        ident = calculate_identity(test_fasta_path, train_fasta_path)
        if ident is not None:
            if ident > max_identity:
                max_identity = ident

                # 将最大相似性写入结果文件
    result_file_path = join(result_dir, f"test_seq_{i}.txt")
    with open(result_file_path, "w") as ofile:
        ofile.write(str(max_identity))
    print(f"seq_{i}的最大相似性是{max_identity}")
    return max_identity


def write_fasta(sequence, filename, header="sequence"):
    """
    Write an amino acid sequence to a FASTA file.

    Parameters:
    sequence (str): The amino acid sequence to write.
    filename (str): The name of the output FASTA file.
    header (str): The header for the FASTA file (default is "sequence").
    """
    with open(filename, 'w') as f:
        f.write(f">{header}\n")  # Write the header
        # Write the sequence, splitting it into lines of 60 characters
        f.write(sequence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 计算平均相似性: 45.2650671785029
    parser.add_argument("--output_dir", type=str, default="analysis/split_protein_sequence_identity/dataset")
    parser.add_argument("--test_data", type=str, default="test_data")
    parser.add_argument("--train_data", type=str, default="train_data")
    args = parser.parse_args()

    create_directory_if_not_exists(args.output_dir)
    create_directory_if_not_exists(os.path.join(args.output_dir, args.test_data))
    create_directory_if_not_exists(os.path.join(args.output_dir, args.train_data))
    create_directory_if_not_exists(result_dir)

    # 加载原始数据集
    data_path = 'dataset/input.tsv'
    data_set = pd.read_csv(data_path, sep='\t')
    # data_set = data_set.dropna(subset=['kcat'])
    # data_set = data_set.sample(frac=1, random_state=42)
    # 加载索引
    indices = np.load('dataset/w_dataset_split.npz')
    train_indices = indices['train_index']
    test_indices = indices['test_index']

    train_sequence = data_set['sequence'].iloc[train_indices].tolist()
    test_sequence = data_set['sequence'].iloc[test_indices].tolist()

    # for key in train_sequence:
    #     write_fasta(train_sequence[key],
    #                 f"{args.output_dir}/train_fasta/seq_{key}.fasta",
    #                 header=f"seq_train_{key}")
    #
    # for key in test_sequence:
    #     write_fasta(test_sequence[key],
    #                 f"{args.output_dir}/test_fasta/seq_{key}.fasta",
    #                 header=f"seq_test_{key}")

    # # 使用 ProcessPoolExecutor 进行并行处理
    # with ProcessPoolExecutor() as executor:
    #     futures = {executor.submit(process_test_sequence, i): i for i in test_indices}
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Processing test sequences"):
    #         i = futures[future]
    #         try:
    #             result = future.result()  # 获取结果
    #         except Exception as e:
    #             print(f"Error processing test seq_{i}: {e}")

    print("相似性计算完成，结果已保存！")
    similarity_path = os.path.join(args.output_dir, "similarity")
    count = 0
    for f in os.listdir(similarity_path):
        with open(os.path.join(similarity_path, f), "r") as file:
            for line in file:
                count = count + float(line)
    print(f"计算平均相似性: {count / len(os.listdir(similarity_path))}")

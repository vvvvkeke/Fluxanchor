# Load model directly
from huggingface_hub import login
import sys
sys.path.append("./")
sys.path.append("../")
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig, SamplingConfig
import torch
import pandas as pd
import time
import argparse
from utils import create_directory_if_not_exists
import os


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if __name__ == '__main__':
    print(sys.path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../predict/rubisco_km")  # ..
    # parser.add_argument("--output_dir", type=str, default="../predict/Synechocystis sp/")
    args = parser.parse_args()
    # Prepare data
    df = pd.read_csv(os.path.join(args.input_dir, "dataset/input.tsv"), sep="\t", encoding="utf-8")

    # torch.cuda.set_device(0)  # 明确设置使用 GPU 1
    model = ESM3.from_pretrained("esm3-open").to("cuda")

    path = os.path.join(args.input_dir, "dataset/ESM3")
    create_directory_if_not_exists(path)

    for i, row in df.iterrows():
        start = time.time()
        uniprot_id = row['uniprot_id']

        save_path = os.path.join(path, f"{uniprot_id}.tensor")
        if os.path.exists(save_path):
            continue

        protein = ESMProtein(sequence=(row['sequence']))
        protein_tensor = model.encode(protein)
        output = model.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
        )
        torch.save(output.per_residue_embedding[1:-1], save_path)
        end = time.time()
        print(f'第{i + 1}条序列({uniprot_id})处理完成，用时：{end - start:.2f}秒')

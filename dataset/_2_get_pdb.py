import pandas as pd
from chai_lab.chai1 import run_inference
from pathlib import Path
import os
import sys
import argparse
import logging
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO
sys.path.append("./")
sys.path.append("../")
from utils import create_directory_if_not_exists


def int_to_chain(i, base=62):
    """
    int_to_chain(int,int) -> str
    Converts a positive integer to a chain ID. Chain IDs include uppercase
    characters, numbers, and optionally lowercase letters.
    i = a positive integer to convert
    base = the alphabet size to include. Typically 36 or 62.
    """
    if i < 0:
        raise ValueError("positive integers only")
    if base < 0 or 62 < base:
        raise ValueError("Invalid base")

    quot = int(i) // base
    rem = i % base
    if rem < 26:
        letter = chr(ord("A") + rem)
    elif rem < 36:
        letter = str(rem - 26)
    else:
        letter = chr(ord("a") + rem - 36)
    if quot == 0:
        return letter
    else:
        return int_to_chain(quot - 1, base) + letter


class OutOfChainsError(Exception):
    pass


def rename_chains(structure):
    """Renames chains to be one-letter chains

    Existing one-letter chains will be kept. Multi-letter chains will be truncated
    or renamed to the next available letter of the alphabet.

    If more than 62 chains are present in the structure, raises an OutOfChainsError

    Returns a map between new and old chain IDs, as well as modifying the input structure
    """
    next_chain = 0  #
    # single-letters stay the same
    chainmap = {c.id: c.id for c in structure.get_chains() if len(c.id) == 1}
    for o in structure.get_chains():
        if len(o.id) != 1:
            if o.id[0] not in chainmap:
                chainmap[o.id[0]] = o.id
                o.id = o.id[0]
            else:
                c = int_to_chain(next_chain)
                while c in chainmap:
                    next_chain += 1
                    c = int_to_chain(next_chain)
                    if next_chain >= 62:
                        raise OutOfChainsError()
                chainmap[c] = o.id
                o.id = c
    return chainmap

def add_cryst1_header(pdb_file, output_file):
    # Define a default CRYST1 line (adjust values as needed)
    # cryst1_line = "CRYST1   90.000   90.000   90.000  90.00  90.00  90.00 P 1           1\n"
    line = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
    # line = "PARENT N/A\n"
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    # Check if CRYST1 line is present
    is_atom = any(line.startswith('ATOM') for line in lines)

    with open(output_file, 'w') as file:
        if is_atom:
            # Add CRYST1 line at the beginning if not present
            file.write(line)
        file.writelines(lines)

if __name__ == "__main__":
    # HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 python _2_get_pdb.py
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="../predict/Synechocystis sp")  # ../predict/Synechocystis sp
    # parser.add_argument("--input_dir", type=str, default="../predict/Synechocystis sp/dataset/"
    #                                                      "input.tsv")

    # parser.add_argument("--output_dir", type=str, default="../predict/Synechocystis sp/dataset/cif_outputs")
    # parser.add_argument("--ciffile", default="dataset/cif_outputs/outputs")
    # parser.add_argument("--pdbfile", default="dataset/pdb_outputs")
    args = parser.parse_args()

    chai_dir = os.path.join(args.input_dir, "dataset/cif_outputs")
    fasta_input_dir = os.path.join(chai_dir, "inputs")
    cif_output_dir = os.path.join(chai_dir, "outputs")
    pdbfile = os.path.join(args.input_dir, "dataset/pdb_outputs")

    create_directory_if_not_exists(chai_dir)
    create_directory_if_not_exists(fasta_input_dir)
    create_directory_if_not_exists(cif_output_dir)
    create_directory_if_not_exists(pdbfile)

    # 设置目录路径
    input_dir = os.path.join(args.input_dir, "dataset/input.tsv")
    df = pd.read_csv(input_dir, sep="\t", encoding="utf-8")
    tsv_file = "val_protein_record.tsv"
    if os.path.exists(tsv_file):
        print("存在tsv文件")
        record = pd.read_csv(tsv_file)
    else:
        print("不存在tsv文件")
        record = pd.DataFrame(columns=['id', 'sequence'])
    # 遍历目录下的所有文件
    for index, row in df.iterrows():
        id = row.loc["uniprot_id"]
        protein = row.loc["sequence"]
        output_dir = Path(os.path.join(cif_output_dir, f"{id}"))
        if os.path.exists(output_dir):
            print(f"{output_dir}已存在")
            continue
        print(f"{output_dir}不存在，开始生成")
        example_fasta = f"""
        >protein|name=protein
        {protein}
        """.strip()
        if len(protein) > 2048:
            print("chai-lab不支持长度大于2048的氨基酸序列")
            record.loc[record.shape[0]] = [id, protein]
            record.to_csv(tsv_file, sep=",", index=False)
            continue
        fasta_path = Path(os.path.join(fasta_input_dir, f"{id}.fasta"))

        fasta_path.write_text(example_fasta)
        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=output_dir,
            # 'default' setup
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            seed=42,
            device="cuda:0",
            use_esm_embeddings=True,
            # See example .aligned.pqt files in this directory
            msa_directory=Path(__file__).parent,
            # Exclusive with msa_directory; can be used for MMseqs2 server MSA generation
            use_msa_server=False,
        )
        cif_paths = candidates.cif_paths
        scores = [rd.aggregate_score for rd in candidates.ranking_data]
        # pLDDTs = candidates.plddt.max()
        # pTM = scores["pTM"]
        # candidates.ranking_data
        scores_data = pd.DataFrame(columns=['id', 'pLDDTs', 'pTM'])
        for i in range(len(candidates.ranking_data)):
            print(f"pLDDTs={candidates.ranking_data[i].plddt_scores.complex_plddt}")
            print(f"pTM={candidates.ranking_data[i].ptm_scores.complex_ptm}")
            scores_data.loc[i] = [i, candidates.ranking_data[i].plddt_scores.complex_plddt,
                                  candidates.ranking_data[i].ptm_scores.complex_ptm]
            scores_data.to_csv(os.path.join(output_dir, "score_data.tsv"), sep=",", index=False)


    # Not sure why biopython needs this to read a cif file
    strucid = "1xxx"
    for root, dirs, files in os.walk(cif_output_dir):
        best_scores = -100
        f = '0'
        uniprot_id = root.split("/")[-1]
        if os.path.exists(os.path.join(pdbfile, uniprot_id + ".pdb")):
            print(f"{uniprot_id} was pdb")
            continue
        for file in files:
            # 构造完整的文件路径
            file_path = os.path.join(root, file)
            # 处理文件
            subfix = file_path[-3:]

            if subfix == "npz":
                scores = np.load(file_path)
                # scores['aggregate_score']
                ptm = scores['ptm'][0]

                if ptm > best_scores:
                    best_scores = ptm
                    f = file[-5]
        # Read file
        if os.path.exists(os.path.join(root, "pred.model_idx_" + f + ".cif")):
            parser = MMCIFParser()
            structure = parser.get_structure(strucid, os.path.join(root, "pred.model_idx_" + f + ".cif"))

            # rename long chains
            try:
                chainmap = rename_chains(structure)
            except OutOfChainsError:
                logging.error("Too many chains to represent in PDB format")
                sys.exit(1)

            if True:
                for new, old in chainmap.items():
                    if new != old:
                        logging.info("Renaming chain {0} to {1}".format(old, new))

            # Write PDB
            io = PDBIO()
            io.set_structure(structure)
            io.save(os.path.join(pdbfile, uniprot_id + ".pdb"))
            # 行头添加说明支持dssp运行
            add_cryst1_header(os.path.join(pdbfile, uniprot_id + ".pdb"), os.path.join(pdbfile, uniprot_id + ".pdb"))
            print(f"{uniprot_id} transfer to pdb")

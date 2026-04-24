import argparse
import os
import subprocess
import sys
sys.path.append("./")
sys.path.append("../")
from utils import create_directory_if_not_exists

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # predict/rubisco_km

    parser.add_argument("--input_dir", type=str, default="../predict/rubisco_km")  # ../predict/Synechocystis sp
    parser.add_argument("--sh_path", type=str, default="..")
    args = parser.parse_args()
    path = args.input_dir
    PDB_DIR = os.path.join(path, "dataset/pdb_outputs")

    OUT_DIR = os.path.join(path, "dataset/dssp_file")

    create_directory_if_not_exists(OUT_DIR)

    script_path = f'{args.sh_path}/get_dssp.sh'

    args = [PDB_DIR, OUT_DIR]
    subprocess.run([script_path] + args)
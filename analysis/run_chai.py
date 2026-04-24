import argparse
import os
from pathlib import Path

import pandas as pd
from chai_lab.chai1 import run_inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    fasta_path = Path(args.fasta_path)
    output_dir = Path(args.output_dir)
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        # 'default' setup
        num_trunk_recycles=3,
        num_diffn_timesteps=100,
        seed=42,
        device="cuda:0",
        use_esm_embeddings=True,
        # See example .aligned.pqt files in this directory
        msa_directory=Path(__file__).parent,
        # Exclusive with msa_directory; can be used for MMseqs2 server MSA generation
        use_msa_server=False,
    )
    # cif_paths = candidates.cif_paths
    # scores = [rd.aggregate_score for rd in candidates.ranking_data]
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
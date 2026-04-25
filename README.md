FluxAnchor: A Multimodal Deep Learning Framework Bridging In Vitro and In Vivo Enzyme Kinetics for Metabolic Modeling🧬


[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

### Installation
```bash
conda create --name 3dgraph python=3.11
conda activate 3dgraph
pip install torch==2.3.0 transformers==4.42.3 datasets==2.19.2 accelerate==0.30.1 peft==0.11.1 trl==0.9.4 deepspeed==0.14.0 vllm==0.4.3
pip install chai_lab==0.5.0  
pip install esm
pip install unimol_tools --upgrade
pip install e3nn

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install torch_geometric

pip install cobra openpyxl requests pebble xlsxwriter Bio Require quest scikit-learn  RDKit seaborn pubchempy bioservices==1.10.4 pyprobar xmltodict plotly kaleido nbformat jupyterlab ipykernel xgboost matplotlib wandb
pip install escher
 ```

By the way, the environment used by TurNUP(using esm2) is not compatible with ours. To reproduce the baseline/TurNUP in our code repository, we need to install the following Anaconda environment:

```bash
conda create -n ESM-2 python==3.9
conda activate ESM-2
pip install torch torchvision torchaudio
pip install fair-esm
pip install xgboost
pip install pandas
pip install rdkit
pip install drfp
pip install numpy==1.24.3
pip install joblib
pip install scikit-learn
 ```

Similarly, catpred also uses esm2, so the following environment also needs to be installed:
```bash
conda create -n catpred python=3.9 -y
conda activate catpred
pip install -y 
matplotlib=3.1.3 
numpy=1.18.1 
pandas=1.0.3 
pytorch=1.11.0 
scikit-learn=0.22.2 
scipy=1.4.1 
tensorboardX=2.0 
torchvision=0.5.0 
tqdm=4.45.0 
seaborn 
-c pytorch

pip install 
pandas-flavor==0.2.0 
faiss-cpu 
pytorch-scatter 
torch-geometric 
ipdb 
fair-esm 
progres 
rdkit-pypi 
transformers 
descriptastorus 
sentencepiece 
rotary_embedding_torch==0.6.5 
typed-argument-parser==1.6.1
 ```

### Quick Start ⚡⚡⚡

**The file directory is as follows (only important files are displayed):**
```
FluxAnchor
│
├── analysis
    ├──analysis_13C.py
    ├──analysis_proteomics.py
    └──analysis_vivo_vitro_kcat.py
│
├── baseline 
    ├──DLKcat
    ├──CatPred-1.0.1
    ├──UniKP
    ├──TurNUP
    └──DeepEnzyme
│
├── basic
│
├── dataset 
    ├──input.tsv
    ├──_1_get_ESM_tensor.py
    ├──_2_get_pdb.py
    ├──_3_get_dssp.py
    ├──_4_get_uni_mol_tensor.py
    ├──_5_get_llm_tensor.py preprocess.py
    └──preprocess.py
│
├── models
│
├── predict
    ├──data_file 
    ├──iECDH1ME8569_1439
    └──rubisco
│
├── result
│
├── results_extend_reactions
│
├── Kcatllamafactory 
│
├── script 
│
├── _1_get_ecGEM_dataset.py
├── _2_get_ecGEM_by_KinLLM.py
├── _2baseline_get_ecGEM_by_AutoPACMEN.py
├── _2baseline_get_ecGEM_by_Catpred.py
├── _2baseline_get_ecGEM_by_DLkcat.py
├── _2baseline_get_ecGEM_by_TurNup.py
├── _2baseline_get_ecGEM_by_UniKP.py
├── _3_make_13C_data_with_TF_regulation_v2.py
├── _4_fluxanchor_by_bayesian.py
├── _5_test_fluxanchor.py
├── predict_kcat_km.py
├── train_test_model.py
├── ecGEM_utils.py
├── utils.py
└── README.md
```


**Validation of KinLLM Model**
```bash
python train_test_model.py --task test
```

**Generate a specified deep learning dataset from a metabolic network and build ecGEM using FluxAnchor**
```bash
python _1_get_ecGEM_dataset.py
python _2_get_ecGEM_by_KinLLM.py 
python _4_fluxanchor_by_bayesian.py
```

**Generate a specified deep learning dataset from a metabolic network and build ecGEM using KinLLM**
```bash
python _1_get_ecGEM_dataset.py
python _2_get_ecGEM_by_KinLLM.py  
```

**Building ecGEM using other baseline models (requires extracting the baseline folder in advance):**

AutoPACMEN:
```bash
python _2baseline_get_ecGEM_by_AutoPACMEN.py
```

Catpred:
```bash
python _1_get_ecGEM_dataset.py
python _2baseline_get_ecGEM_by_Catpred.py
```

DLkcat:
```bash
python _1_get_ecGEM_dataset.py
python _2baseline_get_ecGEM_by_DLkcat.py
```

TurNUP:
```bash
python _1_get_ecGEM_dataset.py
python _2baseline_get_ecGEM_by_TurNup.py
```

UniKP:
```bash
python _1_get_ecGEM_dataset.py
python _2baseline_get_ecGEM_by_UniKP.py
```

**Downstream metabolic network analysis**
```bash
cd analysis
python analysis_13C.py
python analysis_vivo_vitro_kcat.py
python analysis_proteomics.py
 ```


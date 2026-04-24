import argparse
import cobra
import datetime
import pandas as pd
import subprocess
import re
import sys

sys.path.append(r'./script/')

from script.ECMpy_function import *


parser = argparse.ArgumentParser()


def kcat_mw_calculation(ouputdf, metdf):
    '''
    Make the kcat_mw input file for ecGEM construction.

    Arguments:
    * DLouputdf: pandas.DataFrame - inputdf from DLKCAT.
    * metdf: pandas.DataFrame - DataFrame containing metabolite, reaction, gene, similes, prosequence, mass information.
    '''

    ouputdf.reset_index(drop=True, inplace=True)
    metdf.reset_index(drop=True, inplace=True)

    if 'reac_id' not in ouputdf.columns:
        raise ValueError("ouputdf必须包含reac_id列")
    if 'reactions' not in metdf.columns:
        raise ValueError("metdf必须包含reactions列")
    # ouputdf['join_key'] = ouputdf['reac_id']
    # metdf['join_key'] = metdf['reactions']
    # Combine DLouputdf and metdf based on index
    # outputdf_rex = pd.concat([metdf, ouputdf['pred_kcats_inverse_10']], axis=1)
    outputdf_rex = pd.merge(
        left=metdf,
        right=ouputdf[['reac_id', 'pred_kcats_inverse_10']],
        left_on='reactions',  
        right_on='reac_id',  
        how='left'  
    )
    # Remove rows with 'None' in Kcat value
    outputdf_rex = outputdf_rex[outputdf_rex['pred_kcats_inverse_10'] != 'None']
    outputdf_rex = outputdf_rex[outputdf_rex['pred_kcats_inverse_10'] != np.nan]
    # Convert Kcat value to float
    outputdf_rex['pred_kcats_inverse_10'] = outputdf_rex['pred_kcats_inverse_10'].astype(float)
    outputdf_rex.dropna()
    # Sort by Kcat value and keep only the first occurrence of each reaction
    outputdf_rex = outputdf_rex.sort_values('pred_kcats_inverse_10', ascending=False).drop_duplicates(
        subset=['reactions'], keep='first')

    # Calculate kcat_mw
    outputdf_rex['kcat_mw'] = outputdf_rex['pred_kcats_inverse_10'] * 3600 * 1000 / outputdf_rex['totalmass']

    # Prepare reaction_kcat_mw DataFrame
    reaction_kcat_mw = pd.DataFrame()
    reaction_kcat_mw['reactions'] = outputdf_rex['reactions']
    reaction_kcat_mw['data_type'] = 'ours'
    reaction_kcat_mw['kcat'] = outputdf_rex['pred_kcats_inverse_10']
    reaction_kcat_mw['MW'] = outputdf_rex['mass']
    reaction_kcat_mw['kcat_MW'] = outputdf_rex['kcat_mw']
    reaction_kcat_mw.reset_index(drop=True, inplace=True)

    print('reaction_kcat_mw generated')
    return reaction_kcat_mw

def main(args):
    method = "Catpred"
    kcat_folder = f"{args.path}/analysis/get_kcat_mw_by_{method}/"
    dataset_path = f"{args.path}/dataset/"
    create_file(kcat_folder)
    create_file(dataset_path)
    sbml_path = f"{args.path}/network.xml"
    comdf_file = f'{dataset_path}comdf.csv'
    ouputdf_file = f'{kcat_folder}kcat_ablation_trainvalModel_expseqemb36_attn6_esm_ens10_Pretrained_egnnFeats.csv'
    reaction_kcat_mw_file = f'{kcat_folder}reaction_kcat_MW.csv'
    input_file = f'{dataset_path}input.tsv'

    input_df = pd.read_csv(input_file, sep="\t")
    input_df['reactant_smiles'] = input_df['smiles_substrate']
    pdbpath = []
    for i, row in input_df.iterrows():
        pdbpath.append(f"{dataset_path}pdb_outputs" + row['uniprot_id'] + ".pdb")
    input_df['pdbpath'] = pdbpath
    input_df['uniprot'] = input_df['uniprot_id']
    input_df.to_csv(f"{kcat_folder}kcat-random_test.csv", index=False)

    # Step 0: read GEM
    if re.search('\.xml', sbml_path):
        model = cobra.io.read_sbml_model(sbml_path)
    elif re.search('\.json', sbml_path):
        model = cobra.io.json.load_json_model(sbml_path)

    ###################################################################################
    starttime = datetime.datetime.now()
    # Step 1: use Kcat method to calculate kcat
    path_text = args.path
    path_text = path_text.split("/")[1]
    # path_text = path_text.replace(" ", "\\ ")  
    print("Starting to Use Kcat method calculate kcat...")
    sh_file = "./baseline/CatPred-1.0.1/catpred_pipeline/catpred/reproduce_get_ecGEM.sh"
    if not os.path.exists(sh_file):
        print(f"Error: Script not found at {sh_file}")
        return False
    subprocess.run(['bash', f'{sh_file}', 'prediction', f'{path_text}'], text=True)
    print("Kcat method done!")
    print()
    endtime = datetime.datetime.now()
    print(endtime - starttime)
    ##################################################################################

    starttime = datetime.datetime.now()
    # Step 2: get the kcat_mw file
    print("Starting to get reaction kcat_mw for model......")
    ouputdf = pd.read_csv(ouputdf_file, sep=',')
    ouputdf['pred_kcats_inverse_10'] = np.power(10, ouputdf['log10kcat_max'])
    ouputdf.to_csv(ouputdf_file, index=False)
    comdf = pd.read_csv(comdf_file)
    reaction_kcat_mw = kcat_mw_calculation(ouputdf, comdf)
    reaction_kcat_mw = reaction_kcat_mw.dropna()
    reaction_kcat_mw.to_csv(reaction_kcat_mw_file, index=False)
    print("Reaction kcat_mw done!")
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 3: get ecGEM
    print("Starting to get reaction kcat_mw for model......")
    method = "Catpred"  # DLkcat AutoPACMEN EkiLLm
    ecGEM_path = f"{args.path}/analysis/get_kcat_mw_by_{method}/ecGEM"
    create_file(ecGEM_path)

  
    taxonom_id = 83333
    reaction_kcat_MW_file = f"{args.path}/analysis/get_kcat_mw_by_{method}/reaction_kcat_MW.csv"

    gene_abundance_colname = 'abundance'
    gene_abundance_file = './predict/data_file/gene_abundance.csv'  # downolad from https://pax-db.org/download
    ecModel_output_file = f"{ecGEM_path}/ecGEM_irr_enz_constraint.json"
    f = 0.4  # The enzyme mass fraction 
    ptot = 0.56  # The total protein fraction in cell. 
    sigma = 0.5  # The approximated saturation of enzyme.e.g.,0.5/1
    lowerbound = 0  # Lowerbound  of enzyme concentration constraint.
    if "iECDH1ME8569_1439" in args.path:
        upperbound = 0.35 
    else:
        upperbound = 0.5  # round(ptot * f * sigma, 3)  # total enzyme
    normal_model = cobra.io.read_sbml_model(sbml_path)
    if args.is_auto:
        obj = 'BIOMASS_Ec_SynAuto_1'  
        model.objective = obj
        normal_model = set_auto_model(normal_model)
    # else:
    #     obj = 'BIOMASS_Ecoli_core_w_GAM'
    # normal_model.objective = obj

    trans_model2enz_json_model_split_isoenzyme(ecGEM_path, sbml_path, reaction_kcat_MW_file, f, ptot, sigma, lowerbound,
                                               upperbound,
                                               ecModel_output_file, synauto=args.is_auto, comdf=comdf)
    enz_model = get_enzyme_constraint_model(ecModel_output_file)

    # reaction_ids = [rxn.id for rxn in enz_model.reactions]
    print(f"ecGEM saved in {ecModel_output_file}")
    endtime = datetime.datetime.now()
    print(endtime - starttime)


if __name__ == '__main__':
    # predict/km_predict
    # predict/Synechocystis sp
    # predict/rubisco
    # predict/E.coli
    # predict/iECDH1ME8569_1439
    # predict/Bacillus subtilis
    parser.add_argument("--path", type=str,
                        default="predict/Bacillus subtilis")

    parser.add_argument("--is_auto", type=bool, default=False)


    args = parser.parse_args()
    main(args=args)

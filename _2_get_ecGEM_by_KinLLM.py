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

def kcat_km_mw_calculation(ouputdf, metdf):
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
    # # 创建映射关系
    # ouputdf['join_key'] = ouputdf['reac_id']
    # metdf['join_key'] = metdf['reactions']
    # Combine DLouputdf and metdf based on index
    # outputdf_rex = pd.concat([metdf, ouputdf['pred_kcats_inverse_10']], axis=1)
    outputdf_rex = pd.merge(
        left=metdf,
        right=ouputdf[['reac_id', 'pred_kcats_inverse_10', 'pred_kms_inverse_10']],
        left_on='reactions',  
        right_on='reac_id',  
        how='left' 
    )
    # Convert prediction values to numeric and keep valid rows only
    outputdf_rex['pred_kcats_inverse_10'] = pd.to_numeric(
        outputdf_rex['pred_kcats_inverse_10'], errors='coerce'
    )
    outputdf_rex['pred_kms_inverse_10'] = pd.to_numeric(
        outputdf_rex['pred_kms_inverse_10'], errors='coerce'
    )
    outputdf_rex = outputdf_rex.dropna(subset=['pred_kcats_inverse_10', 'pred_kms_inverse_10'])

    # Ensure MW is usable; fallback to monomer mass when totalmass is missing
    outputdf_rex['mass'] = pd.to_numeric(outputdf_rex['mass'], errors='coerce')
    outputdf_rex['totalmass'] = pd.to_numeric(outputdf_rex['totalmass'], errors='coerce')
    outputdf_rex.loc[outputdf_rex['totalmass'] <= 0, 'totalmass'] = np.nan
    outputdf_rex['totalmass'] = outputdf_rex['totalmass'].fillna(outputdf_rex['mass'])
    outputdf_rex = outputdf_rex.dropna(subset=['totalmass'])
    outputdf_rex['catalytic_efficiency'] = outputdf_rex['pred_kcats_inverse_10'] / outputdf_rex['pred_kms_inverse_10']
    # outputdf_rex already filtered for required values
    # Sort by Kcat value and keep only the first occurrence of each reaction
    outputdf_rex = outputdf_rex.sort_values('pred_kcats_inverse_10', ascending=False).drop_duplicates(
        subset=['reactions'], keep='first')

    # Calculate kcat_mw
    outputdf_rex['efficiency_mw'] = outputdf_rex['catalytic_efficiency'] * 1000 / outputdf_rex['totalmass']
    outputdf_rex['kcat_mw'] = outputdf_rex['pred_kcats_inverse_10'] * 3600 * 1000 / outputdf_rex['totalmass']
    # outputdf_rex['catalytic_efficiency'] = outputdf_rex['kcat_mw'] / outputdf_rex['pred_kms_inverse_10'] / 1000
    # Prepare reaction_kcat_mw DataFrame
    reaction_kcat_mw = pd.DataFrame()
    reaction_kcat_mw['reactions'] = outputdf_rex['reactions']
    reaction_kcat_mw['data_type'] = 'predict'
    reaction_kcat_mw['kcat'] = outputdf_rex['pred_kcats_inverse_10']
    reaction_kcat_mw['km'] = outputdf_rex['pred_kms_inverse_10']  
    reaction_kcat_mw['MW'] = outputdf_rex['mass']
    reaction_kcat_mw['kcat_MW'] = outputdf_rex['kcat_mw']
    reaction_kcat_mw['catalytic_efficiency'] = outputdf_rex['catalytic_efficiency']  
    reaction_kcat_mw.reset_index(drop=True, inplace=True)

    print('reaction_kcat_mw generated')
    return reaction_kcat_mw

def vitro_to_vivo(reaction_kcat_MW_file, reaction_kcat_mw_modify_file,
                          super_threshold=1e5, non_rate_threshold=1e3,
                          super_coff=3.0, non_rate_coff=2.0, rate_coff=1.0):
    df = pd.read_csv(reaction_kcat_MW_file)
    data = df['catalytic_efficiency']
    for index, row in df.iterrows():
        if row['catalytic_efficiency'] > super_threshold:  
            df.at[index, 'kcat'] = row['kcat'] * super_coff
            df.at[index, 'kcat_MW'] = row['kcat_MW'] * super_coff
        elif non_rate_threshold < row['catalytic_efficiency'] <= super_threshold:  
            df.at[index, 'kcat'] = row['kcat'] * non_rate_coff
            df.at[index, 'kcat_MW'] = row['kcat_MW'] * non_rate_coff
        else: 
            df.at[index, 'kcat'] = row['kcat'] * rate_coff
            df.at[index, 'kcat_MW'] = row['kcat_MW'] * rate_coff

    df.to_csv(reaction_kcat_mw_modify_file, index=False)

def main(args):
    kcat_folder = f"{args.path}/analysis/get_kcat_mw_by_KinLLM/"
    dataset_path = f"{args.path}/dataset/"
    create_file(kcat_folder)
    create_file(dataset_path)
    sbml_path = f"{args.path}/network.xml"
    comdf_file = f'{dataset_path}comdf.csv'
    ouputdf_file = f'{kcat_folder}saved_kcat_km.csv'
    reaction_kcat_mw_file = f'{kcat_folder}reaction_kcat_MW.csv'
    reaction_kcat_mw_modify_file = f'{kcat_folder}reaction_kcat_MW_modify.csv'
    # Step 0: read GEM
    if re.search('\.xml', sbml_path):
        model = cobra.io.read_sbml_model(sbml_path)
    elif re.search('\.json', sbml_path):
        model = cobra.io.json.load_json_model(sbml_path)

    starttime = datetime.datetime.now()

    # Step 1: use Kcat method to calculate kcat
    path_text = args.path
    path_text = path_text.replace(" ", "\\ ")

    # ###################################################################################
    # print("Starting to Use Kcat method calculate kcat...")
    # env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # print("------------1_get_ESM_tensor------------")
    # cmd_str = f"python ./dataset/_1_get_ESM_tensor.py --input_dir {path_text}"
    # subprocess.run(cmd_str, env=env, shell=True)
    # print("------------2_get_pdb------------")
    # cmd_str = f"python ./dataset/_2_get_pdb.py --input_dir {path_text}" # 40g~80g cuda, chai
    # subprocess.run(cmd_str, env=env, shell=True)
    # print("------------3_get_dssp------------")
    # cmd_str = f"python ./dataset/_3_get_dssp.py --input_dir {path_text} --sh_path ./dataset"
    # subprocess.run(cmd_str, env=env, shell=True)
    # print("------------4_get_uni_mol_tensor------------")
    # cmd_str = f"python ./dataset/_4_get_uni_mol_tensor.py --input_dir {path_text}"
    # subprocess.run(cmd_str, env=env, shell=True)
    # print("------------5_get_llm_tensor------------") 
    # #### --resume: Determine whether to continue from the last interruption
    # env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    # cmd_str = f"python ./dataset/_5_get_llm_tensor.py --input_dir {path_text} --json_output . --resume" # 20g cuda
    # subprocess.run(cmd_str, env=env, shell=True)
    # print("------------predict_kcat_km------------")
    # env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # cmd_str = f"python ./predict_kcat_km.py --path {path_text} --batch_size {args.batch_size}"
    # subprocess.run(cmd_str, env=env, shell=True, check=True)  # 32g cuda
    # print("Kcat method done!")
    # print()
    # endtime = datetime.datetime.now()
    # print(endtime - starttime)
    ###################################################################################


    starttime = datetime.datetime.now()
    # Step 2: get the kcat_mw file
    print("Starting to get reaction kcat_mw for model......")
    ouputdf = pd.read_csv(ouputdf_file, sep=',')
    comdf = pd.read_csv(comdf_file)
    reaction_kcat_mw = kcat_km_mw_calculation(ouputdf, comdf)
    reaction_kcat_mw = reaction_kcat_mw.dropna(subset=['kcat', 'km', 'MW', 'kcat_MW'])
    reaction_kcat_mw.to_csv(reaction_kcat_mw_file, index=False)
    print("Reaction kcat_mw done!")
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 3: get ecGEM
    print("Starting to get reaction kcat_mw for model......")
    method = "KinLLM"  # DLkcat AutoPACMEN KinLLM
    ecGEM_path = f"{args.path}/analysis/get_kcat_mw_by_{method}/ecGEM"
    create_file(ecGEM_path)

    reaction_kcat_MW_file = f"{args.path}/analysis/get_kcat_mw_by_{method}/reaction_kcat_MW.csv"
    # paxdb丰度数据
    gene_abundance_colname = 'abundance'
    gene_abundance_file = './predict/data_file/gene_abundance.csv'  # downolad from https://pax-db.org/download
    ecModel_output_file = f"{ecGEM_path}/ecGEM_irr_enz_constraint.json"

    normal_model = cobra.io.read_sbml_model(sbml_path)
    # if args.is_auto:
    #     obj = 'BIOMASS_Ec_SynAuto_1'  # BIOMASS_Ec_SynAuto_1 BIOMASS_Ec_SynMixo_1  BIOMASS_Ec_SynHetero_1 obj是反应，一般指向带有BIOMASS的
    #     model.objective = obj
    #     # model.reactions.get_by_id("EX_ac_e").bounds = (-10, 10)
    #     normal_model = set_auto_model(normal_model)
    # else:
    #     model.reactions.get_by_id("EX_ac_e").bounds = (-10, 10)
    # normal_model.objective = obj
    f = 0.4  # The enzyme mass fraction 
    ptot = 0.56  # The total protein fraction in cell. 
    sigma = 0.5  # The approximated saturation of enzyme.e.g.,0.5/1. 
    lowerbound = 0  # Lowerbound  of enzyme concentration constraint.
    if "iECDH1ME8569_1439" in args.path:
        upperbound = 0.35 
    else:
        upperbound = 0.5  # round(ptot * f * sigma, 3)  # total enzyme
    # if 'Synechocystis' in args.path:
    #     vitro_to_vivo(reaction_kcat_MW_file, reaction_kcat_mw_modify_file,
    #                           super_threshold=1e5, non_rate_threshold=1e3,
    #                           super_coff=3.0, non_rate_coff=2.0, rate_coff=1.0)
    # else:
    #     vitro_to_vivo(reaction_kcat_MW_file, reaction_kcat_mw_modify_file,
    #                           super_threshold=1e5, non_rate_threshold=1e3,
    #                           super_coff=1.1, non_rate_coff=1.0, rate_coff=0.5)
    
    df_file = reaction_kcat_MW_file
    trans_model2enz_json_model_split_isoenzyme(ecGEM_path, sbml_path, df_file, f, ptot, sigma, lowerbound,
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
    # predict/ENGRO2
    # predict/Bacillus subtilis
    # predict/iM1515
    # predict/yeast9
    parser.add_argument("--path", type=str,
                        default="predict/Synechocystis sp")
    parser.add_argument("--is_auto", type=bool, default=True)
    # multi GPU command： CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_test_model.py --multi_gpu True
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--multi_gpu", type=bool, default=False,
                        help="use multi gpus. If you want to use multi gpu for training, "
                             "you should use this command:"
                             "CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_test_model.py --multi_gpu True")
    parser.add_argument('--gpu_id', type=str, default='0')  #
    parser.add_argument("--batch_size", type=str, default="256")
    parser.add_argument("--model_name", type=str, default='fusion') 
    parser.add_argument("--result_dir", type=str, default='best')
    # parser.add_argument("--vitro_to_vivo", type=bool, default=False)
    args = parser.parse_args()
    main(args=args)

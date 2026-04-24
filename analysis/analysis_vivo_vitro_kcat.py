import cobra
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(r'./')
sys.path.append(r'../')
sys.path.append(r'./script/')
sys.path.append(r'../script/')
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from script.AutoPACMEN_function import *
from script.ECMpy_function import *
from ast import literal_eval


def extract_gene_annotations(model):
    """提取基因的refseq_name到UniProt的映射"""
    gene_mapping = defaultdict(list)
    reaction_mapping = defaultdict(list)
    kcat_mapping = defaultdict(list)
    kcat_MW_mapping = defaultdict(list)
    for gene in model.genes:
        # 获取refseq_name（可能存在于注释或属性中）
        if gene.annotation.__contains__("refseq_name"):
            refseq_name = gene.annotation["refseq_name"]
            uniprot = gene.annotation['uniprot']
            gene_mapping[refseq_name] = uniprot
            reaction_ids = [rxn.id for rxn in gene.reactions]
            kcat_mapping[refseq_name] = [rxn.kcat for rxn in gene.reactions]
            kcat_MW_mapping[refseq_name] = [rxn.kcat_MW for rxn in gene.reactions]
            reaction_mapping[refseq_name] = reaction_ids

            # print(reaction_ids)
    return gene_mapping, reaction_mapping, kcat_mapping, kcat_MW_mapping

def get_enzyme_usage(model, sol_result, gene_uniprot_map, gene_reaction_map):
    # 创建一个集合存储所有的酶促反应
    enzyme_reactions = set()

    # 遍历基因-反应映射
    for gene, reactions in gene_reaction_map.items():
        if gene in gene_uniprot_map:  # 确保该基因有对应的 UniProt ID
            # 为每个反应创建对应的酶使用反应ID
            for reaction in reactions:
                enzyme_reactions.add(reaction)

    # 找到这些酶促反应在模型中的索引
    usages = [rxn.id for i, rxn in enumerate(model.reactions)
              if rxn.id in enzyme_reactions]

    # 获取这些反应的通量值（绝对值）
    usage_flux = np.abs(sol_result[usages])
    usage_kcat = {}
    usage_kcat_MW = {}
    usage_method = {}
    for i, reaction in enumerate(model.reactions):
        if reaction.id in usages:
            usage_kcat[reaction.id] = reaction.kcat
            usage_kcat_MW[reaction.id] = reaction.kcat_MW

    return usages, usage_flux, usage_kcat, usage_kcat_MW, usage_method

def enzymatic_reactions(model):
    '''
        Returns a list of cobra Reaction objects catalyzed by enzymes.
    '''
    # 筛选出至少有一个基因的反应
    enzymatic_reactions = {r: list(r.genes) for r in model.reactions if len(r.genes) >= 1}
    # reactions = filter(lambda r: len(model.genes) >= 1, model.reactions)
    # genes = [list(model.genes) for model in reactions]
    return enzymatic_reactions

def reactions_by_homomeric_enzymes(model):
    '''
        Returns a list of reactions (as cobra REACTION objects)
        in the model catalyzed by unique enzymes which are composed
        of a single polypeptide chain, i.e., unique homomeric enzymes.
    '''
    # 获取所有由酶催化的反应
    enzymatic_reactions = {r: list(r.genes) for r in model.reactions if len(r.genes) >= 1}
    # 筛选出由单一基因催化的反应
    homomers = {r: genes[0] for r, genes in enzymatic_reactions.items() if len(genes) == 1}
    return homomers


def convert_copies_fL_to_mmol_gCDW(expression_data):
    '''
        Convertes the units of proteomics data (usually reported in
        copies per fL of cytoplasm) to units of mmol per gCDW.
        This unit conversion is performed to match flux units from
        metabolic models (usually given in mmol/gCDW/h)
    '''
    rho = 1100  # average cell density gr/liter
    DW_fraction = 0.3  # fraction of DW of cells
    Avogadro = 6.02214129  # Avogadro's number "exponent-less"
    unnamed_col = expression_data['Unnamed: 0'].copy()
    expression_data = expression_data.drop('Unnamed: 0', axis=1)
    expression_data[expression_data < 10] = np.nan
    expression_data /= (Avogadro * 1e5)
    expression_data /= (rho * DW_fraction)
    expression_data['Unnamed: 0'] = unnamed_col
    return expression_data

def map_expression_by_reaction(model, proteins_mmol_gCDW):

    # gc = v.fluxes.keys() & proteins_mmol_gCDW.columns
    tmp = {k.id: v.id for k, v in reactions_by_homomeric_enzymes(model).items()}
    mapC = {}
    for key, key2 in tmp.items():
        # 在 mapB 中查找 key2 对应的 value
        if key2 in our_id_refseq_name_map:
            mapC[key] = our_id_refseq_name_map[key2]
    tmp2 = {}
    for key, key2 in mapC.items():
        # 在 mapB 中查找 key2 对应的 value
        if key2 in name_id_map:
            tmp2[key] = name_id_map[key2]
    # 创建 df1
    df1 = pd.DataFrame(list(tmp2.items()), columns=['reaction', 'gene'])
    # 创建 df2
    df2 = pd.DataFrame(list(mapC.items()), columns=['reaction', 'name'])
    # 创建 df3
    df3 = pd.DataFrame(list(tmp.items()), columns=['reaction', 'our_name'])
    df_merged = pd.merge(df1, df2, on='reaction', how='outer')
    df_merged = pd.merge(df_merged, df3, on='reaction', how='outer')
    df_merged = df_merged.dropna()
    proteins_mmol_gCDW = proteins_mmol_gCDW.rename(columns={'Unnamed: 0': 'gene'})
    columns_to_add = [col for col in proteins_mmol_gCDW.columns if col != 'gene']
    # for col in columns_to_add:
    #     df_merged[col] = None
    for index, row in df_merged.iterrows():
        gene = row['gene']  # 假设 df_merged 中基因列名为 'gene'，根据实际情况调整
        match = proteins_mmol_gCDW[proteins_mmol_gCDW['gene'] == gene]
        if not match.empty:
            for col in columns_to_add:
                df_merged.at[index, col] = match[col].values[0]
    # df_merged = pd.DataFrame(index=tmp.keys(), columns=proteins_mmol_gCDW.columns)
    # for i in E.index:
    #     if tmp[i] in proteins_mmol_gCDW.index:
    #         E.loc[i] = proteins_mmol_gCDW.loc[tmp[i]]
    # E.dropna(how='all', inplace=True)
    return df_merged

# 定义一个函数，尝试将值转换为浮点数，如果失败则返回 0
def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return 0.0


if __name__ == '__main__':
    json_file = f"../predict/iECDH1ME8569_1439/iJO1366.json"  # iJO1366.json的gene名是bigg标准命名
    ref_model = cobra.io.json.load_json_model(json_file)
    name_id_map = {}
    id_name_map = {}
    for gene in ref_model.genes:
        name_id_map[gene.name] = gene.id
        id_name_map[gene.id] = gene.name
    methods = [
               "FluxGen",
               "KinLLM",
               "Catpred", "UniKP", "TurNup", "DLKcat",
               "AutoPACMEN"
               ]
    # obj = 'BIOMASS_Ec_iJO1366_core_53p95M'
    for method in methods:
        # input_csv = "../predict/iECDH1ME8569_1439/proteomics.csv"
        if method == "KinLLM":
            sbml_file = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_KinLLM/ecGEM/ecGEM_irr_enz_constraint.json"
            output_csv = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_KinLLM/vivo_vitro_with_uniprot_normal.csv"
            model = get_enzyme_constraint_model(sbml_file)
        elif method == "FluxGen":
            # sbml_file = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_KinLLM/ecGEM/Bayesian/best_ecGEM.json"
            sbml_file = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_KinLLM/ecGEM/Bayesian/best_ecGEM_vivo.json"
            output_csv = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_KinLLM/vivo_vitro_with_uniprot_FluxGen.csv"
            model = get_enzyme_constraint_model(sbml_file)
        else:
            sbml_file = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_{method}/ecGEM/ecGEM_irr_enz_constraint.json"
            output_csv = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_{method}/vivo_vitro_with_uniprot.csv"
            model = get_enzyme_constraint_model(sbml_file)
        # 1. 加载代谢模型
        our_id_refseq_name_map = {}
        for gene in model.genes:
            our_id_refseq_name_map[gene.id] = gene.name
        proteins_copies_fL = pd.read_csv('../predict/iECDH1ME8569_1439/meta_abundance.copies_fL.csv', sep=",")
        proteins_mmol_gCDW = convert_copies_fL_to_mmol_gCDW(proteins_copies_fL)
        model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (0, 10)  # glucose uptake
        model.reactions.get_by_id("EX_ac_e").bounds = (0, 10)  # glucose uptake
        model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)  # glucose uptake
        model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
        model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
        E = map_expression_by_reaction(model, proteins_mmol_gCDW)
        proteins = 'GLC_BATCH_mu=0.6_H'
        E_df = E[['reaction', 'gene', 'name', 'our_name', f'{proteins}']]
        E_df = E_df.dropna()
        if method == "AutoPACMEN":
            sol_result = model.optimize()
        else:
            sol_result = cobra.flux_analysis.pfba(model)
        gene_uniprot_map, gene_reaction_map, kcat_map, kcat_MW_map = extract_gene_annotations(model)
        usages, usage_flux, usage_kcat, usage_kcat_MW, usage_method = get_enzyme_usage(model, sol_result, gene_uniprot_map,
                                                                                       gene_reaction_map)
        kins = []
        kouts = []
        indexes = []
        for index, raw in E_df.iterrows():
            reaction = raw['reaction']
            gene = raw['gene']
            E1 = raw[f'{proteins}']
            if usage_flux.__contains__(reaction):
                v = usage_flux[reaction]
                v /= 3600
                kin = v/E1
                kout = usage_kcat[reaction]
                kins.append(kin)
                kouts.append(kout)
                indexes.append(True)
            else:
                kins.append(0)
                kouts.append(0)
                indexes.append(False)

        indexes = np.array(indexes)
        convert_func = np.vectorize(convert_to_float)
        kouts = convert_func(kouts)
        vivo_kcat = np.array(kins) + 1
        vitro_kcat = np.array(kouts) + 1
        E_df['kcat_vivo'] = vivo_kcat
        E_df['kcat_vitro'] = vitro_kcat
        vivo_kcat = vivo_kcat[indexes]
        vitro_kcat = vitro_kcat[indexes]
        E_df = E_df[indexes]
        indexes = vivo_kcat != 1
        vitro_kcat = vitro_kcat[indexes]
        vivo_kcat = vivo_kcat[indexes]
        E_df = E_df[E_df['kcat_vivo'] != 1]
        E_df['kcat_vitro_log10'] = np.log10(vitro_kcat)
        E_df['kcat_vivo_log10'] = np.log10(vivo_kcat)
        E_df = E_df.dropna()
        # 统计 kcat 列中每个值的出现次数
        kcat_counts = E_df['kcat_vitro'].value_counts()
        # 找出出现次数最多的值
        most_common_kcat = kcat_counts.idxmax()
        # 去掉这些值对应的行
        E_df = E_df[E_df['kcat_vitro'] != most_common_kcat]
        x = E_df['kcat_vitro_log10']
        y = E_df['kcat_vivo_log10']
        pearson_corr, p_value = pearsonr(x, y)
        print("Pearson Correlation Coefficient:", pearson_corr)
        print("P-value:", p_value)

        # spearman_corr, p_value = spearmanr(x, y)
        # print("Spearman Rank Correlation Coefficient:", spearman_corr)
        # print("P-value:", p_value)

        # kendall_corr, p_value = kendalltau(x, y)
        # print("Kendall Tau Correlation Coefficient:", kendall_corr)
        # print("P-value:", p_value)
        # 计算密度
        xy = np.vstack([x, y])
        density = gaussian_kde(xy)(xy)
        # 将密度值添加到数据表
        E_df['kcat_Density'] = density
        E_df['absolute_error'] = (E_df['kcat_vitro_log10'] - E_df['kcat_vivo_log10']).abs()
        E_df['square_error'] = (E_df['kcat_vitro_log10'] - E_df['kcat_vivo_log10']) ** 2
        E_df['error'] = (E_df['kcat_vitro_log10'] - E_df['kcat_vivo_log10'])
        print(f"{method}  MAE: {E_df['absolute_error'].sum() / E_df.shape[0]}")
        print(f"{method}  r2: {r2_score(x, y)}")
        print(f"{method} RMSE: {math.sqrt(mean_squared_error(x, y))}")
        print(f"{method} MSE: {mean_squared_error(x, y)}")
        # 保存为新的 TSV 文件
        E_df.to_csv(output_csv, index=False)
        print("save density!")

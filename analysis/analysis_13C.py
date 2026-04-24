import math
import re

import cobra
import numpy as np
import pandas as pd
from collections import defaultdict
import sys

import sklearn.metrics
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

sys.path.append(r'./')
sys.path.append(r'../')
sys.path.append(r'./script/')
sys.path.append(r'../script/')
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from script.AutoPACMEN_function import *
from script.ECMpy_function import *
from sklearn.metrics import r2_score, mean_squared_error
from ast import literal_eval

def convert_values_to_positive(input_dict):
    return {key: abs(value) for key, value in input_dict.items()}

def filter_reactions_by_avg_flux_threshold(C13_data: pd.DataFrame, flux_threshold: float) -> List[str]:
    """
    Filter reactions based on the average flux across all samples (wild type + mutants).

    Args:
        C13_data: 13C flux DataFrame that contains an 'Id' column and reaction columns.
        flux_threshold: Flux threshold.

    Returns:
        filtered_reactions: Reactions whose average flux exceeds the threshold.
    """
    # Extract all reaction columns except the Id column
    reaction_cols = [col for col in C13_data.columns if col != 'Id']

    # Compute the average flux across all samples (wild type + mutants)
    all_samples_flux = C13_data[reaction_cols].abs()  # Use absolute flux values
    avg_flux_per_reaction = all_samples_flux.mean(axis=0)  # Average flux per reaction

    # Keep reactions whose average flux is above the threshold
    active_reactions = avg_flux_per_reaction[avg_flux_per_reaction > flux_threshold].index.tolist()

    return active_reactions


def calculate_net_reversible_fluxes(flux_dict):
    """
    Compute the net flux for reversible reaction pairs.

    For reversible pairs (e.g., FBA and FBA_reverse) compute the net flux:
    - If forward - reverse > 0, set forward = forward - reverse and reverse = 0.
    - If forward - reverse < 0, set forward = 0 and reverse = |forward - reverse|.
    - If forward - reverse = 0, set both fluxes to 0.

    Args:
        flux_dict: Flux dictionary {reaction_id: flux_value}.

    Returns:
        net_flux_dict: Dictionary containing the net flux values.
    """
    result = flux_dict.copy()

    # Identify base reaction names (without the _reverse suffix)
    forward_reactions = set()
    reverse_reactions = set()

    for rxn_id in flux_dict.keys():
        if rxn_id.endswith('_reverse'):
            reverse_reactions.add(rxn_id)
        else:
            forward_reactions.add(rxn_id)

    # Find reversible pairs
    processed = set()

    for forward_rxn in forward_reactions:
        reverse_rxn = forward_rxn + '_reverse'

        if reverse_rxn in reverse_reactions and forward_rxn not in processed:
            # Process one reversible pair
            forward_flux = flux_dict.get(forward_rxn, 0)
            reverse_flux = flux_dict.get(reverse_rxn, 0)

            net_flux = forward_flux - reverse_flux

            if net_flux > 0:
                # Forward direction dominates
                result[forward_rxn] = net_flux
                result[reverse_rxn] = 0
            elif net_flux < 0:
                # Reverse direction dominates
                result[forward_rxn] = 0
                result[reverse_rxn] = abs(net_flux)
            else:
                # Net flux equals zero
                result[forward_rxn] = 0
                result[reverse_rxn] = 0

            processed.add(forward_rxn)
            processed.add(reverse_rxn)

    return result

def aggregate_ground_truth_flux(flux_data: pd.DataFrame, target_reactions: List[str], sample_id: str = None) -> Dict[str, float]:
    """
    Aggregate isozyme fluxes in the ground-truth data.

    Uses the same aggregation logic as project_flux_to_bigg_gems to merge entries such as:
    - MDH, MDH2, MDH3 -> MDH
    - ACACT1r, ACACT2r, ... -> ACACTr
    - ECOAH1, ECOAH2, ... -> ECOAH
    - HACD1, HACD2, ... -> HACD

    Args:
        flux_data: DataFrame containing 13C flux data with reaction IDs as column names.
        target_reactions: Target reaction list that uses the unified names.
        sample_id: Optional sample identifier when multiple rows exist.

    Returns:
        A dictionary of aggregated fluxes {reaction_id: flux_value}.
    """
    # Gather available reaction columns
    available_cols = [col for col in flux_data.columns if col != 'Id']

    # If sample_id is provided, select the corresponding row
    if sample_id is not None:
        row_data = flux_data[flux_data['Id'] == sample_id]
        if row_data.empty:
            return {rxn: 0.0 for rxn in target_reactions}
        flux_series = row_data.iloc[0]
    else:
        # Assume a single row of data
        flux_series = flux_data.iloc[0]

    result = {}

    for rxn_id in target_reactions:
        # Strategy 1: direct match
        direct_flux = abs(float(flux_series.get(rxn_id, 0.0))) if rxn_id in available_cols else 0.0
        
        # Strategy 2: look for isozyme variants
        isozyme_fluxes = []

        # Skip reverse-specific handling for now (left as reference below)
        # is_reverse = rxn_id.endswith('_reverse')
        # if is_reverse:
        #     base_rxn = rxn_id[:-8]  # Remove the '_reverse' suffix
        # else:
        #     base_rxn = rxn_id
        
        isozyme_fluxes = list(flux_series[flux_series.index.str.startswith(rxn_id + '_num', na=False)])
        if rxn_id == "MDH":
            # Match MDH2, MDH3
            pattern = r'^MDH\d$'
            isozyme_fluxes = list(flux_series[flux_series.index.str.match(pattern, na=False)])
        # elif rxn_id == "MDH_reverse":
        #     # Match MDH2_reverse, MDH3_reverse
        #     pattern = r'^MDH\d+_reverse$'
        #     isozyme_fluxes = list(flux_series[flux_series.index.str.match(pattern, na=False)])
            # for col in available_cols:
                # Pattern 1: match rxn_id_num1, rxn_id_num2, rxn_id_num3, etc.
                # if col.startswith(rxn_id + '_num'):
                #     isozyme_fluxes.append(abs(float(flux_series[col])))
                # Pattern 2: match other isozyme suffixes (e.g., _copy1, _copy2)
                # elif col.startswith(rxn_id + '_') and not col.endswith('_reverse'):
                #     suffix = col[len(rxn_id)+1:]
                #     if suffix.startswith('num') or suffix.startswith('copy'):
                #         isozyme_fluxes.append(abs(float(flux_series[col])))

                # Pattern 3: match numeric suffixes (e.g., MDH2, MDH3 -> MDH)
                # # Or ACACT1r, ACACT2r -> ACACTr; ECOAH1, ECOAH2 -> ECOAH
                # if is_reverse:
                #     # For reverse reactions, match base_rxn + number + _reverse
                #     pattern1 = f'^{re.escape(base_rxn)}\\d+_reverse$'
                #     if re.match(pattern1, col):
                #         isozyme_fluxes.append(abs(float(flux_series[col])))
                #     # Match base_rxn without the final letter + number + trailing letter + _reverse
                #     if base_rxn and base_rxn[-1].isalpha():
                #         base_without_suffix = base_rxn[:-1]
                #         suffix_char = base_rxn[-1]
                #         pattern2 = f'^{re.escape(base_without_suffix)}\\d+{re.escape(suffix_char)}_reverse$'
                #         if re.match(pattern2, col):
                #             isozyme_fluxes.append(abs(float(flux_series[col])))
                # else:
                #     # For forward reactions, match base_rxn + number
                #     pattern1 = f'^{re.escape(base_rxn)}\\d+$'
                #     if re.match(pattern1, col):
                #         isozyme_fluxes.append(abs(float(flux_series[col])))
                #     # Match base_rxn without the final letter + number + trailing letter
                #     if base_rxn and base_rxn[-1].isalpha():
                #         base_without_suffix = base_rxn[:-1]
                #         suffix_char = base_rxn[-1]
                #         pattern2 = f'^{re.escape(base_without_suffix)}\\d+{re.escape(suffix_char)}$'
                #         if re.match(pattern2, col) and col != rxn_id:
                #             isozyme_fluxes.append(abs(float(flux_series[col])))

        # Aggregation strategy: direct flux + sum of all isozyme fluxes
        if isozyme_fluxes:
            total_isozyme_flux = sum(abs(f) for f in isozyme_fluxes)
            result[rxn_id] = direct_flux + total_isozyme_flux
        else:
            result[rxn_id] = direct_flux
    result = calculate_net_reversible_fluxes(result)
    return result


# Extended list of central metabolism reactions, including forward and reverse directions.
# Covers glycolysis, pentose phosphate pathway, Entner-Doudoroff, TCA, anaplerosis, glyoxylate shunt, and acetate metabolism.
# Adds key amino-acid precursors, one-carbon metabolism, and selected nucleotide precursor reactions.
# Keep _reverse reactions to detect metabolic overflow and related physiological states.
CENTRAL_METABOLISM_60 = [
    # ===== EMP =====
    "PGI", # Phosphoglucose isomerase (pentose-6-phosphate isomerase)
    "PFK_num1", "PFK_num2", # Phosphofructokinase
    "FBP_num1", "FBP_num2", "FBP_num3", # Fructose-1,6-bisphosphatase
    "FBA_num1", "FBA_num2", "FBA_num3", # Fructose-bisphosphate aldolase
    "TPI", # Triose phosphate isomerase
    "GAPD", # Glyceraldehyde-3-phosphate dehydrogenase
    "PGK",
    "PGM_num1", "PGM_num2", "PGM_num3", # Phosphoglycerate mutase
    "ENO", 
    "GLCptspp",  "PYK", "TPI",
    "FBA_reverse_num1", "FBA_reverse_num2", "FBA_reverse_num3",  # FBA reverse flux
    "PGK_reverse",  # PGK reverse (substrate-level phosphorylation)
    "ENO_reverse",  # ENO reverse flux
    "GAPD_reverse", # GAPD reverse flux
    "TPI_reverse",  # TPI reverse flux (DHAP/GAP interconversion)
    "PGI_reverse",  # PGI reverse flux (glucose-6-phosphate/fructose-6-phosphate)
    "PGM_reverse_num1", "PGM_reverse_num2", "PGM_reverse_num3",  # PGM reverse flux

    # ===== Pentose Phosphate Pathway =====
    "G6PDH2r", "PGL", "GND", "RPE_num1", "RPE_num2", "RPI_num1", "RPI_num2", "TKT1", "TKT2", "TALA",
    # Reversible PPP reactions
    "G6PDH2r_reverse",  # Glucose-6-phosphate dehydrogenase reverse flux
    "RPE_reverse_num1", "RPE_reverse_num2",   # Ribulose-5-phosphate-3-epimerase reverse flux
    "RPI_reverse_num1", "RPI_reverse_num2",   # Ribose-5-phosphate isomerase reverse flux
    "TKT1_reverse",  # Transketolase reverse flux
    "TKT2_reverse",  # Transketolase reverse flux
    "TALA_reverse",  # Transaldolase reverse flux

    # ===== Entner-Doudoroff Pathway =====
    "EDA", # 2-dehydro-3-deoxy-phosphogluconate aldolase
    "EDD", # 6-phosphogluconate dehydratase

    # ===== TCA Cycle =====
    "PDH", # Pyruvate dehydrogenase
    "CS", # Citrate synthase
    "ACONTa_num1", "ACONTa_num2", # Aconitase a
    "ACONTb_num1", "ACONTb_num2",  # Aconitase b
    "ICDHyr", # Isocitrate dehydrogenase
    "AKGDH", # Alpha-ketoglutarate dehydrogenase
    "SUCOAS", # Succinyl-CoA synthetase
    "SUCDi", # Succinate dehydrogenase
    "FUM_num1", "FUM_num2", # Fumarase
    "MDH", # Malate dehydrogenase (multiple isoforms)
    # Reversible TCA reactions
    "ACONTa_reverse_num1", "ACONTa_reverse_num2", # Aconitase a reverse flux
    "ACONTb_reverse_num1", "ACONTb_reverse_num2", # Aconitase b reverse flux
    "ICDHyr_reverse", # Isocitrate dehydrogenase reverse flux
    "SUCOAS_reverse", # Succinyl-CoA synthetase reverse flux
    "FUM_reverse_num1", "FUM_reverse_num2",   # Fumarase reverse flux
    "MDH_reverse",   # Malate dehydrogenase reverse flux
    
    # ===== Anaplerotic Reactions =====
    "ME1", "ME2", "PPC", "PPCK",

    # ===== Glyoxylate Shunt =====
    "ICL", # Isocitrate lyase
    "MALS_num1", "MALS_num2", # Malate synthase

    # ===== Acetate Metabolism =====
    "ACKr_num1", "ACKr_num2", "ACKr_num3", # Acetate kinase
    "PTAr_num1", "PTAr_num2", # Phosphotransacetylase
    "ACKr_reverse_num1", "ACKr_reverse_num2", "ACKr_reverse_num3", # Acetate kinase reverse flux
    "PTAr_reverse_num1", "PTAr_reverse_num2",  # Phosphotransacetylase reverse flux
    # ===== Key Amino Acid Precursor Biosynthesis =====
    # Aromatic amino acids
    "CHORS", # chorismate synthase
    "DHQS", # 3-dehydroquinate synthase
    "DHQTi", # 3-dehydroquinate dehydratase
    # Branched-chain amino acids
    "DHAD1", # 2,3-dihydroxy-3-methylbutanoate dehydratase
    "IPMD", # isopropylmalate dehydrogenase
    "IPPS", # isopropylmalate synthase
    # Serine/glycine branch
    "PSERT", # phosphoserine transaminase
    "PSP_L", # phosphoserine phosphatase
    "SERAT", # serine acetyltransferase
    "SERAT_reverse", # Serine acetyltransferase reverse flux
    # Lysine pathway
    "DAPDC", # diaminopimelate decarboxylase
    "DAPE", # diaminopimelate epimerase
    "DHDPS", # dihydrodipicolinate synthase
    "DHDPRy", # dihydrodipicolinate reductase
    "DAPE_reverse",
    # ===== One-Carbon Metabolism =====
    "MTHFC", # methenyltetrahydrofolate cyclohydrolase
    "MTHFD", # methylenetetrahydrofolate dehydrogenase
    "MTHFR2", # methylenetetrahydrofolate reductase
    "MTHFC_reverse",
    "MTHFD_reverse",
    # ===== Selected Nucleotide Precursors =====
    "GLUPRT", # glutamine phosphoribosylpyrophosphate amidotransferase
    "PRAIS", # phosphoribosylamine--glycine ligase
    "PPS" # Phosphoenolpyruvate synthase (gluconeogenesis)
]

# Extended reaction list (based on pathway_annotations in 13C_log2fc_all_reaction_cluster2.py).
# Covers a broader set spanning glycolysis, PPP, TCA, amino acids, fatty acids, nucleotides, etc.

EXTEND_REACTIONS = [
    # ===== EMP =====
    "PGI", # 磷酸葡萄糖异构酶 pehtose-6-phosphate isomerase
    "PFK", # 磷酸果糖激酶 phosphofructokinase
    "FBA", # 果糖二磷酸醛缩酶 fructose-bisphosphate aldolase
    "TPI", # 磷酸丙糖异构酶 triose phosphate isomerase
    "GAPD", # 甘油醛-3-磷酸脱氢酶 glyceraldehyde-3-phosphate dehydrogenase
    "PGK",
    "PGM", # 磷酸甘油酸变位酶 phosphoglycerate mutase
    "ENO", 
    "GLCptspp",  "PYK",
    "FBA_reverse",  # FBA 逆流
    "PGK_reverse",  # PGK 逆流：磷酸化底物合成
    "ENO_reverse",  # ENO 逆流
    "GAPD_reverse", # GAPD 逆流
    "TPI_reverse",  # TPI 逆流（磷酸二羟丙酮/甘油醛-3-磷酸互变）
    "PGI_reverse",  # PGI 逆流（葡萄糖-6-磷酸/果糖-6-磷酸互变）
    "PGM_reverse",  # PGM 逆流（磷酸甘油酸变位酶）

    # ===== PP (Pentose Phosphate Pathway) =====
    "G6PDH2r", "PGL", "GND", "RPE", "RPI", "TKT1", "TKT2", "TALA",
    "G6PDH2r_reverse",  # 葡萄糖-6-磷酸脱氢酶逆流
    "RPE_reverse",   # 核酮糖-5-磷酸-3-差向异构酶
    "RPI_reverse",   # 核糖-5-磷酸异构酶
    "TKT1_reverse",  # 可逆转酮醇酶反应
    "TKT2_reverse",  # 可逆转酮醇酶反应
    "TALA_reverse",  # 可逆转醛醇酶反应
    # ===== ED途径 (Entner-Doudoroff) =====
    "EDA", # 2-脱氢-3-脱氧-磷酸葡萄糖酸醛缩酶
    "EDD", # 6-磷酸葡萄糖酸脱水酶

    # ===== TCA循环 (TCA Cycle) =====
    "PDH", # 丙酮酸脱氢酶 pyruvate dehydrogenase
    "CS", # 柠檬酸合成酶 citrate synthase
    "ACONTa", # 顺乌头酸酶a aconitase a
    "ACONTb",  # 顺乌头酸酶b aconitase b
    "ICDHyr", # 异柠檬酸脱氢酶 isocitrate dehydrogenase
    "AKGDH", # α-酮戊二酸脱氢酶 alpha-ketoglutarate dehydrogenase
    "SUCOAS", # 琥珀酰辅酶A合成酶 succinyl-CoA synthetase
    "SUCDi", # 琥珀酸脱氢酶 succinate dehydrogenase
    "FUM", # 延胡索酸酶 fumarase
    "MDH", # 苹果酸脱氢酶同工酶
    # TCA 循环的可逆反应
    "ACONTa_reverse", # 顺乌头酸酶a逆流
    "ACONTb_reverse", # 顺乌头酸酶b逆流
    "ICDHyr_reverse", # 异柠檬酸脱氢酶逆流
    "SUCOAS_reverse", # 琥珀酰辅酶A合成酶逆流
    "FUM_reverse",   # 延胡索酸酶逆流
    "MDH_reverse",   # 苹果酸脱氢酶逆流
    
    # ===== 回补反应 (Anaplerotic Reactions) =====
    "FBP", # 果糖二磷酸酶 fructose-1,6-bisphosphatase
    "ME1", "ME2", "PPC", 

    # ===== 脂肪酸 (Fatty Acid) =====
    # 脂肪酸同工酶（不合并，展开为具体的同工酶）
    'ACCOAC',
    'ACACT1r', 'ACACT2r', 'ACACT3r', 'ACACT4r', 'ACACT5r', 'ACACT6r', 'ACACT7r', 'ACACT8r',
    'ACACT1r_reverse', 'ACACT2r_reverse', 'ACACT3r_reverse', 'ACACT4r_reverse',
    'ACACT5r_reverse', 'ACACT6r_reverse', 'ACACT7r_reverse', 'ACACT8r_reverse',
    'ECOAH1', 'ECOAH2', 'ECOAH3', 'ECOAH4', 'ECOAH5', 'ECOAH6', 'ECOAH7', 'ECOAH8',
    'ECOAH1_reverse', 'ECOAH2_reverse', 'ECOAH3_reverse', 'ECOAH4_reverse',
    'ECOAH5_reverse', 'ECOAH6_reverse', 'ECOAH7_reverse', 'ECOAH8_reverse',
    'HACD1', 'HACD2', 'HACD3', 'HACD4', 'HACD5', 'HACD6', 'HACD7', 'HACD8',
    'HACD1_reverse', 'HACD2_reverse', 'HACD3_reverse', 'HACD4_reverse',
    'HACD5_reverse', 'HACD6_reverse', 'HACD7_reverse', 'HACD8_reverse',
    
    # ===== 糖异生 (Gluconeogenesis) =====
    "PPCK", # 磷酸烯醇丙酮酸羧激酶 phosphoenolpyruvate carboxykinase
    "PPS",  # 磷酸烯醇丙酮酸合成酶 phosphoenolpyruvate synthase
    # 注意：FBP已在糖酵解部分列出

    # ===== 氨基酸代谢 (Amino Acid Metabolism) =====
    "ASPTA", "ALATA_L", "GLNS", "THRS", "METS",
    # 氨基酸转氨酶可逆反应
    "ASPTA_reverse",    # 天冬氨酸转氨酶逆流
    "ALATA_L_reverse",  # 丙氨酸转氨酶逆流

    # ===== 乙醛酸循环 (Glyoxylate Shunt) =====
    "ICL", # 异柠檬酸裂解酶 isocitrate lyase
    "MALS", # 苹果酸合成酶 malate synthase

    # ===== 乙酸代谢 (Acetate Metabolism) =====
    "ACKr", "PTAr",
    # 乙酸代谢的逆向（乙酸摄取 vs 分泌）
    "ACKr_reverse",  # 乙酸激酶逆流
    "PTAr_reverse",  # 磷酸转乙酰酶逆流
    # ===== 关键氨基酸前体合成 (Key Amino Acid Precursor Biosynthesis) =====
    # 芳香族氨基酸途径
    "CHORS", # chorismate synthase
    "DHQS", # 3-dehydroquinate synthase
    "DHQTi", # 3-dehydroquinate dehydratase
    # 支链氨基酸途径
    "DHAD1", # 2,3-dihydroxy-3-methylbutanoate dehydratase
    "IPMD", # isopropylmalate dehydrogenase
    "IPPS", # isopropylmalate synthase
    # 丝氨酸/甘氨酸途径
    "PSERT", # phosphoserine transaminase
    "PSP_L", # phosphoserine phosphatase
    "SERAT", # serine acetyltransferase
    "SERAT_reverse", # serine acetyltransferase
    # 赖氨酸途径
    "DAPDC", # diaminopimelate decarboxylase
    "DAPE", # diaminopimelate epimerase
    "DHDPS", # dihydrodipicolinate synthase
    "DHDPRy", # dihydrodipicolinate reductase
    # Cystathionine途径
    "CYSTL", # cystathionine lyase
    # Anthranilate synthase
    "ANS",
    # ===== 一碳代谢 (One-Carbon Metabolism) =====
    "GHMT2r", # glycine hydroxymethyltransferase
    "MTHFC", # methenyltetrahydrofolate cyclohydrolase
    "MTHFD", # methylenetetrahydrofolate dehydrogenase
    "MTHFR2", # methylenetetrahydrofolate reductase
    "GHMT2r_reverse",
    "MTHFC_reverse",
    "MTHFD_reverse",
    # ===== 部分核苷酸前体 (Nucleotide Precursors) =====
    "GLUPRT", # glutamine phosphoribosylpyrophosphate amidotransferase
    "PRAIS", # phosphoribosylamine--glycine ligase
]

# EXTEND_REACTIONS = [
#     # ===== EMP =====
#     "PGI", # 磷酸葡萄糖异构酶 pehtose-6-phosphate isomerase
#     "PFK_num1", "PFK_num2", # 磷酸果糖激酶 phosphofructokinase
#     "FBP_num1", "FBP_num2", "FBP_num3", # 果糖二磷酸酶 fructose-1,6-bisphosphatase
#     "FBA_num1", "FBA_num2", "FBA_num3", # 果糖二磷酸醛缩酶 fructose-bisphosphate aldolase
#     "TPI", # 磷酸丙糖异构酶 triose phosphate isomerase
#     "GAPD", # 甘油醛-3-磷酸脱氢酶 glyceraldehyde-3-phosphate dehydrogenase
#     "PGK",
#     "PGM_num1", "PGM_num2", "PGM_num3", # 磷酸甘油酸变位酶 phosphoglycerate mutase
#     "ENO", 
#     "GLCptspp",  "PYK", "TPI",
#     "FBA_reverse_num1", "FBA_reverse_num2", "FBA_reverse_num3",  # FBA 逆流
#     "PGK_reverse",  # PGK 逆流：磷酸化底物合成
#     "ENO_reverse",  # ENO 逆流
#     "GAPD_reverse", # GAPD 逆流
#     "TPI_reverse",  # TPI 逆流（磷酸二羟丙酮/甘油醛-3-磷酸互变）
#     "PGI_reverse",  # PGI 逆流（葡萄糖-6-磷酸/果糖-6-磷酸互变）
#     "PGM_reverse_num1", "PGM_reverse_num2", "PGM_reverse_num3",  # PGM 逆流（磷酸甘油酸变位酶）

#     # ===== PP (Pentose Phosphate Pathway) =====
#     "G6PDH2r", "PGL", "GND", "RPE_num1", "RPE_num2", "RPI_num1", "RPI_num2", "TKT1", "TKT2", "TALA",

#     # ===== ED途径 (Entner-Doudoroff) =====
#     "EDA", # 2-脱氢-3-脱氧-磷酸葡萄糖酸醛缩酶
#     "EDD", # 6-磷酸葡萄糖酸脱水酶

#     # ===== TCA循环 (TCA Cycle) =====
#     "PDH", # 丙酮酸脱氢酶 pyruvate dehydrogenase
#     "CS", # 柠檬酸合成酶 citrate synthase
#     "ACONTa_num1", "ACONTa_num2", # 顺乌头酸酶a aconitase a
#     "ACONTb_num1", "ACONTb_num2",  # 顺乌头酸酶b aconitase b
#     "ICDHyr", # 异柠檬酸脱氢酶 isocitrate dehydrogenase
#     "AKGDH", # α-酮戊二酸脱氢酶 alpha-ketoglutarate dehydrogenase
#     "SUCOAS", # 琥珀酰辅酶A合成酶 succinyl-CoA synthetase
#     "SUCDi", # 琥珀酸脱氢酶 succinate dehydrogenase
#     "FUM_num1", "FUM_num2", # 延胡索酸酶 fumarase
#     "MDH", # 苹果酸脱氢酶 malate dehydrogenase
#     # TCA 循环的可逆反应
#     "ACONTa_reverse_num1", "ACONTa_reverse_num2", # 顺乌头酸酶a逆流
#     "ACONTb_reverse_num1", "ACONTb_reverse_num2", # 顺乌头酸酶b逆流
#     "ICDHyr_reverse", # 异柠檬酸脱氢酶逆流
#     "SUCOAS_reverse", # 琥珀酰辅酶A合成酶逆流
#     "FUM_reverse_num1", "FUM_reverse_num2",   # 延胡索酸酶逆流
#     "MDH_reverse",   # 苹果酸脱氢酶逆流
    
#     # ===== 回补反应 (Anaplerotic Reactions) =====
#     "ME1", "ME2", "PPC", "PPCK",

#     # ===== 脂肪酸 (Fatty Acid) =====
#     'ACCOAC', 
#     'ACACT1r', 'ACACT2r', 'ACACT3r', 'ACACT4r', 'ACACT5r', 'ACACT6r', 'ACACT7r', 'ACACT8r',
#     'ACACT1r_reverse', 'ACACT2r_reverse', 'ACACT3r_reverse', 'ACACT4r_reverse', 'ACACT5r_reverse', 'ACACT6r_reverse', 'ACACT7r_reverse', 'ACACT8r_reverse',
#     'ECOAH1', 'ECOAH2', 'ECOAH3', 'ECOAH4', 'ECOAH5', 'ECOAH6', 'ECOAH7', 'ECOAH8',
#     'ECOAH1_reverse', 'ECOAH2_reverse', 'ECOAH3_reverse', 'ECOAH4_reverse', 'ECOAH5_reverse', 'ECOAH6_reverse', 'ECOAH7_reverse', 'ECOAH8_reverse',
#     'HACD1', 'HACD2', 'HACD3', 'HACD4', 'HACD5', 'HACD6', 'HACD7', 'HACD8',
#     'HACD1_reverse', 'HACD2_reverse', 'HACD3_reverse', 'HACD4_reverse', 'HACD5_reverse', 'HACD6_reverse', 'HACD7_reverse', 'HACD8_reverse',
    
#     # ===== 糖异生 (Gluconeogenesis) =====
#     "PPCK", # 磷酸烯醇丙酮酸羧激酶 phosphoenolpyruvate carboxykinase
#     "PPS",  # 磷酸烯醇丙酮酸合成酶 phosphoenolpyruvate synthase
#     # 注意：FBP已在糖酵解部分列出

#     # ===== 氨基酸代谢 (Amino Acid Metabolism) =====
#     "ASPTA", "ALATA_L", "GLNS", "GHMT2r", "THRS", "SERAT", "METS",
#     # 氨基酸转氨酶可逆反应
#     "ASPTA_reverse",    # 天冬氨酸转氨酶逆流
#     "ALATA_L_reverse",  # 丙氨酸转氨酶逆流
#     "SERAT_reverse",    # 丝氨酸转乙酰酶逆流
#     "GHMT2r_reverse",   # 甘氨酸羟甲基转移酶逆流

#     # 氨基酸（可能是代谢物或合成反应，根据实际情况保留）
#     # "GLU", "ASP", "SER", "GLY", "ALA", "VAL", "LEU", "ILE", "LYS", "ARG",
#     # "HIS", "PRO", "TRP", "PHE", "TYR", "CYS", "MET", "THR",

#     # ===== 脂肪酸代谢 (Fatty Acid Metabolism) =====
#     "ACCOAC",

#     # ===== 回补反应 (Anaplerotic Reactions) =====
#     "PPC", "ME1", "ME2", "HCO3E",
#     # 注意：ME1、ME2、PPC 通常是单向反应（释放 CO2，热力学不可逆）

#     # ===== 乙醛酸循环 (Glyoxylate Shunt) =====
#     "ICL", # 异柠檬酸裂解酶 isocitrate lyase
#     "MALS_num1", "MALS_num2", # 苹果酸合成酶 malate synthase
#     # 乙醛酸循环可逆反应
#     # "MALS_reverse",  # 苹果酸合成酶逆流

#     # ===== 乙酸代谢 (Acetate Metabolism) =====
#     "ACKr", "PTAr",
#     # 乙酸代谢的逆向（乙酸摄取 vs 分泌）
#     "ACKr_reverse",  # 乙酸激酶逆流
#     "PTAr_reverse",  # 磷酸转乙酰酶逆流
#     # ===== 关键氨基酸前体合成 (Key Amino Acid Precursor Biosynthesis) =====
#     # 芳香族氨基酸途径
#     "CHORS", # chorismate synthase
#     "DHQS", # 3-dehydroquinate synthase
#     "DHQTi", # 3-dehydroquinate dehydratase
#     # 支链氨基酸途径
#     "DHAD1", # 2,3-dihydroxy-3-methylbutanoate dehydratase
#     "IPMD", # isopropylmalate dehydrogenase
#     "IPPS", # isopropylmalate synthase
#     # 丝氨酸/甘氨酸途径
#     "PSERT", # phosphoserine transaminase
#     "PSP_L", # phosphoserine phosphatase
#     "SERAT", # serine acetyltransferase
#     "SERAT_reverse", # serine acetyltransferase
#     # 赖氨酸途径
#     "DAPDC", # diaminopimelate decarboxylase
#     "DAPE", # diaminopimelate epimerase
#     "DHDPS", # dihydrodipicolinate synthase
#     "DHDPRy", # dihydrodipicolinate reductase
#     # Cystathionine途径
#     "CYSTL_num1", "CYSTL_num2", # cystathionine lyase
#     # Anthranilate synthase
#     "ANS",
#     # ===== 一碳代谢 (One-Carbon Metabolism) =====
#     "GHMT2r", # glycine hydroxymethyltransferase
#     "MTHFC", # methenyltetrahydrofolate cyclohydrolase
#     "MTHFD", # methylenetetrahydrofolate dehydrogenase
#     "MTHFR2", # methylenetetrahydrofolate reductase
#     "GHMT2r_reverse",
#     "MTHFC_reverse",
#     "MTHFD_reverse",
#     # ===== 部分核苷酸前体 (Nucleotide Precursors) =====
#     "GLUPRT", # glutamine phosphoribosylpyrophosphate amidotransferase
#     "PRAIS", # phosphoribosylamine--glycine ligase
#     "PPS" # 糖异生磷酸烯醇丙酮酸合成酶 gluconeogenesis phosphoenolpyruvate synthase
# ]

def project_flux_to_bigg_gems(pred_flux: Dict[str, float], ground_truth_reactions: List[str],
                               flux_threshold: float = 1e-6) -> Dict[str, float]:
    """
    直接映射 GEMs 分辨率的预测流量到 ground truth 反应
    处理同工酶反应（如 GLCptspp_num1, GLCptspp_num2 等），将它们聚合到基础反应名称

    重要：保留 _reverse 后缀，不将正向和反向反应合并
          例如：PGK 和 PGK_reverse 是两个独立的反应
          这对于识别代谢溢出（overflow）等生理状态至关重要

    同工酶合并规则：
    1. _num1, _num2, _num3 模式: FBA_num1, FBA_num2 -> FBA
    2. 数字后缀模式: MDH, MDH2, MDH3 -> MDH; ACACT1r, ACACT2r -> ACACTr
    3. _reverse 单独处理，不与正向反应合并

    参数:
    pred_flux: 预测的流量字典 {reaction_id: flux_value}
    ground_truth_reactions: ground truth 中的反应列表
    flux_threshold: 流量阈值，低于此值的反应会被过滤（默认 1e-6）

    返回:
    映射后的流量字典
    """
    result = {}

    for rxn_id in ground_truth_reactions:
        # 方法1: 直接匹配（精确名称）
        direct_flux = abs(pred_flux.get(rxn_id, 0.0))

        # 方法2: 查找同工酶变体
        isozyme_fluxes = []

        # 检查是否是 _reverse 反应
        is_reverse = rxn_id.endswith('_reverse')
        if is_reverse:
            base_rxn = rxn_id[:-8]  # 去掉 '_reverse' 后缀
        else:
            base_rxn = rxn_id

        for key in pred_flux.keys():
            # 模式1: 匹配 rxn_id_num1, rxn_id_num2, rxn_id_num3 等
            if key.startswith(rxn_id + '_num'):
                isozyme_fluxes.append(abs(pred_flux[key]))
            # 模式2: 匹配其他同工酶后缀（_copy1, _copy2 等）
            elif key.startswith(rxn_id + '_') and not key.endswith('_reverse'):
                suffix = key[len(rxn_id)+1:]
                if suffix.startswith('num') or suffix.startswith('copy'):
                    isozyme_fluxes.append(abs(pred_flux[key]))

            # 模式3: 匹配以数字结尾的同工酶（如 MDH, MDH2, MDH3 -> MDH）
            if rxn_id == "MDH":
                pattern = r'^MDH\d+$'
                if re.match(pattern, key):
                    isozyme_fluxes.append(abs(pred_flux[key]))
            # 或者 ACACT1r, ACACT2r -> ACACTr; ECOAH1, ECOAH2 -> ECOAH
            # if is_reverse:
            #     # 对于 _reverse 反应，匹配 base_rxn + 数字 + _reverse
            #     # 例如 MDH_reverse 匹配 MDH_reverse, MDH2_reverse, MDH3_reverse
            #     # 例如 ACACTr_reverse 匹配 ACACT1r_reverse, ACACT2r_reverse
            #     # 匹配 base_rxn + 数字 + _reverse（如 MDH2_reverse, MDH3_reverse）
            #     pattern1 = f'^{re.escape(base_rxn)}\\d+_reverse$'
            #     if re.match(pattern1, key):
            #         isozyme_fluxes.append(abs(pred_flux[key]))
            #     # 匹配 base_rxn去掉末尾字母 + 数字 + 末尾字母 + _reverse
            #     # 例如 ACACTr_reverse 匹配 ACACT1r_reverse, ACACT2r_reverse
            #     if base_rxn and base_rxn[-1].isalpha():
            #         base_without_suffix = base_rxn[:-1]  # 如 ACACT
            #         suffix_char = base_rxn[-1]  # 如 r
            #         pattern2 = f'^{re.escape(base_without_suffix)}\\d+{re.escape(suffix_char)}_reverse$'
            #         if re.match(pattern2, key):
            #             isozyme_fluxes.append(abs(pred_flux[key]))
            # else:
            #     # 对于正向反应，匹配 base_rxn + 数字
            #     # 例如 MDH 匹配 MDH, MDH2, MDH3
            #     # 例如 ACACTr 匹配 ACACT1r, ACACT2r
            #     # 匹配 base_rxn + 数字（如 MDH2, MDH3）
            #     pattern1 = f'^{re.escape(base_rxn)}\\d+$'
            #     if re.match(pattern1, key):
            #         isozyme_fluxes.append(abs(pred_flux[key]))
            #     # 匹配 base_rxn去掉末尾字母 + 数字 + 末尾字母
            #     # 例如 ACACTr 匹配 ACACT1r, ACACT2r
            #     # 例如 ECOAH 匹配 ECOAH1, ECOAH2
            #     # 例如 HACD 匹配 HACD1, HACD2
            #     if base_rxn and base_rxn[-1].isalpha():
            #         base_without_suffix = base_rxn[:-1]  # 如 ACACT, ECOA, HAC
            #         suffix_char = base_rxn[-1]  # 如 r, H, D
            #         pattern2 = f'^{re.escape(base_without_suffix)}\\d+{re.escape(suffix_char)}$'
            #         if re.match(pattern2, key) and key != rxn_id:
            #             isozyme_fluxes.append(abs(pred_flux[key]))

        # 聚合策略：取所有同工酶的总和（因为它们代表同一个反应的不同酶）
        if isozyme_fluxes:
            total_isozyme_flux = sum(isozyme_fluxes)
            # 同时考虑基础反应的流量，使用总和而不是max
            result[rxn_id] = direct_flux + total_isozyme_flux
        else:
            result[rxn_id] = direct_flux
    return result

def project_flux_to_bigg(pred_flux: Dict[str, float], organisms) -> Dict[str, float]:
    r = {}
    # pred_flux = convert_values_to_positive(pred_flux)
    if organisms == "Synechocystis sp":
        # —— CBB ——（与 get_result 完全一致的聚合/重命名逻辑）
        r["RBPC_1"] = pred_flux.get("RBPC_1", 0.0)
        r["PGK"] = pred_flux.get("PGK", 0.0)
        r["GAPDi_nadp"] = pred_flux.get("GAPDi_nadp", 0.0)
        r["TPI"] = max(pred_flux.get("TPI_reverse", 0.0), abs(pred_flux.get("TPI", 0.0)))
        r["FBA"] = max(pred_flux.get("FBA_reverse_num1", 0.0), pred_flux.get("FBA_reverse_num2", 0.0), 
                       abs(pred_flux.get("FBA", 0.0)))
        r["FBP"] = max(pred_flux.get("FBP_num1", 0.0), pred_flux.get("FBP_num2", 0.0), 
                       abs(pred_flux.get("FBP", 0.0)))
        r["TKT2\t"] = max(pred_flux.get("TKT2_reverse", 0.0), 
                          abs(pred_flux.get("TKT2", 0.0)))
        r["FBA3"] = max(pred_flux.get("FBA3_reverse_num1", 0.0), pred_flux.get("FBA3_reverse_num2", 0.0), 
                        abs(pred_flux.get("FBA3", 0.0)))
        r["SBP"] = pred_flux.get("SBP", 0.0)
        r["TKT1"] = max(pred_flux.get("TKT1_reverse", 0.0), abs(pred_flux.get("TKT1", 0.0)))
        r["RPE"] = max(pred_flux.get("RPE_reverse", 0.0), abs(pred_flux.get("RPE", 0.0)))
        r["RPI"] = max(pred_flux.get("RPI_num1", 0.0), pred_flux.get("RPI_num2", 0.0), pred_flux.get("RPI", 0.0))
        r["PRUK"] = pred_flux.get("PRUK", 0.0)

        # —— TCA —— 
        r["PDH"] = max(pred_flux.get("PDH_num1", 0.0), pred_flux.get("PDH_num2", 0.0), 
                       pred_flux.get("PDH_num3", 0.0), pred_flux.get("PDH", 0.0))
        # r["PGI"] = max(pred_flux.get("PGI_reverse", 0.0), pred_flux.get("PGI", 0.0))
        r["PGI"] = pred_flux.get("PGI", 0.0)
        r["PGM"] = max(pred_flux.get("PGM_reverse_num1", 0.0), pred_flux.get("PGM_reverse_num2", 0.0),
                    pred_flux.get("PGM_reverse_num3", 0.0), pred_flux.get("PGM_reverse_num4", 0.0), 
                    abs(pred_flux.get("PGM", 0.0)))
        r["ENO"] = pred_flux.get("ENO", 0.0)
        r["PYK"] = max(pred_flux.get("PYK2_num1", 0.0), pred_flux.get("PYK2_num2", 0.0),
                    pred_flux.get("PYK_num1", 0.0), pred_flux.get("PYK_num2", 0.0), pred_flux.get("PYK", 0.0))
        r["CS"] = pred_flux.get("CS", 0.0)
        r["ACONTa_1"] = pred_flux.get("ACONTa_1", 0.0)
        r["ACONTb_1"] = pred_flux.get("ACONTb_1", 0.0)
        r["ICDHyr"] = pred_flux.get("ICDHyr", 0.0)
        r["SUCDu_syn"] = pred_flux.get("SUCDu_syn", 0.0)
        r["FUM"] = pred_flux.get("FUM", 0.0)
        r["MDH"] = pred_flux.get("MDH", 0.0)
        r["PEPC"] = pred_flux.get("PEPC", 0.0)
        r["ME2"] = pred_flux.get("ME2", 0.0)

        # —— 光呼吸 —— 
        r["RBCh_2"] = pred_flux.get("RBCh_2", 0.0)
        r["PGLYCP"] = max(pred_flux.get("PGLYCP_num1", 0.0), pred_flux.get("PGLYCP_num2", 0.0),
                        pred_flux.get("PGLYCP_num3", 0.0), pred_flux.get("PGLYCP", 0.0))
        r["GLYCLTDx"] = max(pred_flux.get("GLYCLTDx_reverse_num1", 0.0),
                            pred_flux.get("GLYCLTDx_reverse_num2", 0.0), abs(pred_flux.get("GLYCLTDx", 0.0)))
        r["GLXCL"] = pred_flux.get("GLXCL", 0.0)
        # print("____________________EX_pyr_e",pred_flux.get("EX_pyr_e", 0.0))
        # print("____________________EX_akg_e",pred_flux.get("EX_akg_e", 0.0))
    elif organisms == "iECDH1ME8569_1439":
        r["pts"] = max(pred_flux.get("GLCptspp_num1", 0.0), pred_flux.get("GLCptspp_num2", 0.0), 
                       pred_flux.get("GLCptspp_num3", 0.0), pred_flux.get("GLCptspp", 0.0))  
        r["zwf"] = pred_flux.get("G6PDH2r", 0.0)
        r["gnd"] = pred_flux.get("GND", 0.0)
        r["pgi"] = pred_flux.get("PGI", 0.0)
        r["edd"] = pred_flux.get("EDD", 0.0)
        r["eda"] = pred_flux.get("EDA", 0.0)
        r["pfk"] = max(pred_flux.get("PFK_num1", 0.0), pred_flux.get("PFK_num2", 0.0), pred_flux.get("PFK", 0.0))
        r["fba"] = max(pred_flux.get("FBA_num1", 0.0), pred_flux.get("FBA_num2", 0.0), 
                       pred_flux.get("FBA_num3", 0.0), pred_flux.get("FBA", 0.0))
        r['tktA'] = pred_flux.get("TKT1", 0.0)
        r['tktB'] = pred_flux.get("TKT2", 0.0)
        r['tal'] = max(pred_flux.get("TALA_num1", 0.0), pred_flux.get("TALA_num2", 0.0), 
                       pred_flux.get("TALA", 0.0))
        r['gap'] = pred_flux.get("GAPD", 0.0)
        r['pgk'] = max(pred_flux.get("PGK_reverse", 0.0), pred_flux.get("PGK", 0.0))
        r['eno'] = pred_flux.get("ENO", 0.0)
        r['pyk'] = max(pred_flux.get("PYK_num1", 0.0), pred_flux.get("PYK_num2", 0.0), pred_flux.get("PYK", 0.0))
        r['pdh'] = pred_flux.get("PDH", 0.0)
        r['glta'] = pred_flux.get("CS", 0.0) 
        r['can'] = max(pred_flux.get("HCO3E_num1", 0.0), pred_flux.get("HCO3E_num2", 0.0), 
                       pred_flux.get("HCO3E", 0.0))
        r['icd'] = pred_flux.get("ICDHyr", 0.0)
        r['suc'] = pred_flux.get("AKGDH", 0.0)
        r['sdh'] = pred_flux.get("SUCDi", 0.0)
        r['fum'] = max(pred_flux.get("FUM_num1", 0.0), pred_flux.get("FUM_num2", 0.0), pred_flux.get("FUM", 0.0))
        r['mdh'] = max(pred_flux.get("MDH", 0.0), pred_flux.get("MDH2", 0.0), pred_flux.get("MDH3", 0.0))
        r['mae'] = max(pred_flux.get("ME1", 0.0), pred_flux.get("ME2", 0.0))
        r['pck'] = pred_flux.get("PPCK", 0.0)
        r['pcc'] = pred_flux.get("PPC", 0.0)
        r['pta'] = max(pred_flux.get("PTAr_num1", 0.0), pred_flux.get("PTAr_num2", 0.0), pred_flux.get("PTAr", 0.0))
        r['acka'] = max(pred_flux.get("ACKr_reverse_num1", 0.0), pred_flux.get("ACKr_reverse_num2", 0.0), 
                        pred_flux.get("ACKr_reverse_num3", 0.0), pred_flux.get("ACKr", 0.0))
        r['acea'] = pred_flux.get("ICL", 0.0)
        r['aceb'] = max(pred_flux.get("MALS_num1", 0.0), pred_flux.get("MALS_num2", 0.0), pred_flux.get("MALS", 0.0))
        # r['glc_uptake'] = pred_flux.get("EX_glc__D_e_reverse", 0)
        # r['acetate secretion'] = pred_flux.get("EX_ac_e", 0)
        # r['Biomass yield'] = pred_flux.get("BIOMASS_Ec_iJO1366_core_53p95M", 0)
        # print("____________________",pred_flux.get("EX_ac_e", 0.0))

    elif organisms == "Bacillus subtilis":
        r["glc_g6p"] = pred_flux.get("GLCpts", 0.0)
        r["g6p_f6p_rev"] = pred_flux.get("PGI", 0.0)
        r["f6p_fbp_rev"] = pred_flux.get("PFK", 0.0)
        r["fbp_gap_dhap_rev"] = max(pred_flux.get("FBA_num1", 0.0), pred_flux.get("FBA_num2", 0.0), 
                                    pred_flux.get("FBA", 0.0))
        r["dhap_gap_rev"] = pred_flux.get("TPI", 0.0)
        r["gap_bpg_rev"] = pred_flux.get("GAPD", 0.0)
        r["bpg_pga_rev"] = max(pred_flux.get("PGK", 0.0), pred_flux.get("PGK_reverse", 0.0))
        r["pga_pep_rev"] = max(pred_flux.get("PGM_reverse", 0.0), pred_flux.get("PGM", 0.0))
        r["pep_pyr_rev"] = pred_flux.get("ENO", 0.0)
        r["g6p_p6g_rev"] = pred_flux.get("G6PDH2r", 0.0)  
        r["p6g_p5p_co2"] = pred_flux.get("GND", 0.0) ## GND
        r["p5p_e4p_gap_f6p_rev"] = max(pred_flux.get("TKT2_reverse", 0.0), pred_flux.get("TKT2", 0.0))
        r["p5p_p5p_s7p_gap_rev"] = max(pred_flux.get("TKT1_reverse", 0.0), pred_flux.get("TKT1", 0.0))
        r["gap_s7p_e4p_f6p_rev"] = max(pred_flux.get("TALA_reverse", 0.0), pred_flux.get("TALA", 0.0))
        r["pyr_accoa_co2"] = pred_flux.get("PDH", 0.0)
        r["oaa_accoa_citicit"] = max(pred_flux.get("CS_num1", 0.0), pred_flux.get("CS_num2", 0.0), 
                                     pred_flux.get("CS", 0.0))
        r["citicit_oga_co2"] = pred_flux.get("ACONT", 0.0)
        r["oga_suc_co2"] = pred_flux.get("AKGDH", 0.0)
        r["suc_mal_rev"] = pred_flux.get("FUM", 0.0)
        r["suc_mal_rev2"] = max(pred_flux.get("SUCD1_reverse", 0.0), pred_flux.get("SUCD1", 0.0))
        r["mal_oaa_rev"] = pred_flux.get("MDH", 0.0)
        r["pyr_co2_oaa"] = pred_flux.get("PC", 0.0)
        r["oaa_pep_co2_rev"] = pred_flux.get("PPCK", 0.0)
        r["mal_pyr_co2_rev"] = pred_flux.get("ME2", 0.0)
        r["2pyr_acetolactate_co2"] = max(pred_flux.get("ACLS_num1", 0.0), pred_flux.get("ACLS_num2", 0.0), 
                                         pred_flux.get("ACLS", 0.0))
        r["acetolactate_acetoin_co2"] = pred_flux.get("ACLDC", 0.0)
        r["acetoin_acetoin_out"] = max(pred_flux.get("ACTNabc", 0.0), pred_flux.get("ACTNabc1", 0.0))
        r["citrate_citrate_out"] = pred_flux.get("EX_cit_e", 0.0)
        r["oga_oga_out"] = pred_flux.get("EX_akg_e", 0.0)
        r["suc_suc_out"] = pred_flux.get("EX_succ_e", 0.0)
        
    return r

if __name__ == '__main__':

    # organisms = "Synechocystis sp"
    organisms = "iECDH1ME8569_1439"
    # organisms = "Bacillus subtilis"
    methods = [
        "FluxGen",
        "KinLLM",
        # "AutoPACMEN",
        "FBA",
        # "Catpred",
        # "UniKP",
        # "DLKcat",
        # "TurNup"
    ]
    print(organisms)

    # ========== 评估样本选择（超参数）==========
    # 可选值:
    #   "wildtype"   - 只评估wildtype（原始行为）
    #   "train"      - 只评估训练集样本（需要读取data_split.csv）- 与贝叶斯优化一致
    # EVAL_SAMPLES = "train"  # 默认使用训练集（与贝叶斯优化一致）
    EVAL_SAMPLES = "wildtype"  # 只评估wildtype

    # ========== 定义不同物种的野生型Id ==========
    WILD_TYPE_IDS = {
        "Synechocystis sp": "WILD TYPE",
        "iECDH1ME8569_1439": "WILD TYPE",
        "Bacillus subtilis": "Wild-type"
    }
    wild_type_id = WILD_TYPE_IDS.get(organisms, "WILD TYPE")  # 默认值为 "WILD TYPE"
    print(f"野生型 Id: {wild_type_id}")

    # ================================================================================
    # 评价方法选择（超参数）
    # ================================================================================
    # 可选值:
    #   "central_60"       - 使用固定的 60 个中心碳代谢反应（推荐，生物学意义强）
    #   "core_30"          - 使用最核心的 30 个反应（基于project_flux_to_bigg映射）
    #   "extend_reactions" - 使用扩展的反应集（包含pathway_annotations中的所有反应+逆反应）
    #   "threshold"        - 使用阈值过滤的反应（真实值 > FLUX_THRESHOLD）
    #   "all_reactions"    - 使用所有反应（不推荐，包含很多非活跃反应）
    # EVALUATION_MODE = "threshold"  # 使用阈值过滤
    # EVALUATION_MODE = "central_60"  # 使用中心碳代谢反应
    # EVALUATION_MODE = "core_30"  # 使用最核心的30个反应
    EVALUATION_MODE = "extend_reactions"  # 使用扩展反应集

    # 设置流量阈值（低于此阈值的反应会被过滤）
    # 仅在 EVALUATION_MODE = "central_60" 时使用
    # 可以根据数据分布调整，建议设置为最大流量的 0.1% - 1%
    # 0.01 :0.340978,1.068042,1.703491  450+反应
    # 0.1  :0.607039,1.838905,3.057813  252反应
    # 0.5  :1.482860,3.669417,6.267239  101反应
    FLUX_THRESHOLD = 0.5  # 0.1 mmol/g/h

    # 读取 13C ground truth 数据
    C13_data_file = f"../predict/{organisms}/13C_data.csv"  # 使用新生成的 GEMs 分辨率数据

    # 检查文件是否存在，如果不存在则使用旧的 train_13C.csv
    import os
    if os.path.exists(C13_data_file):
        print(f"使用 GEMs 分辨率的 13C 数据: {C13_data_file}")
        C13_data = pd.read_csv(C13_data_file)
        use_gems_resolution = True
    # else:
    #     print(f"使用传统的 13C 数据: ../predict/{organisms}/train_13C.csv")
    #     C13_data = pd.read_csv(f"../predict/{organisms}/train_13C.csv", sep=",")
    #     use_gems_resolution = False

    vitro_exp = pd.read_csv(f"../predict/{organisms}/vitro_exp_data.csv", sep=",")
    all_df = None
    skip_c = ["GLYCK2", "TRSARr", "GLXO3r", "OXADC", "GLYCTO1"]

    # ========== 根据EVAL_SAMPLES选择要评估的样本 ==========
    print(f"\n{'='*80}")
    print(f"评估样本选择")
    print(f"{'='*80}")

    if EVAL_SAMPLES == "wildtype":
        # 模式1：只评估wildtype（原始行为）
        eval_sample_ids = [wild_type_id]
        print(f"模式: 仅评估wildtype")
        print(f"样本: {wild_type_id}")
    elif EVAL_SAMPLES == "train":
        # 模式2：只评估训练集样本（需要读取data_split.csv）
        data_split_file = f"../predict/{organisms}/analysis/get_kcat_mw_by_KinLLM/ecGEM/Bayesian/data_split.csv"
        if os.path.exists(data_split_file):
            data_split = pd.read_csv(data_split_file)
            eval_sample_ids = data_split[data_split['Split'] == 'train']['Sample_ID'].tolist()
            print(f"模式: 仅评估训练集样本（与贝叶斯优化一致）")
            print(f"样本数: {len(eval_sample_ids)}")
            print(f"样本: {eval_sample_ids[:5]}{'...' if len(eval_sample_ids) > 5 else ''}")
        else:
            print(f"WARNING: 未找到 {data_split_file}，回退到wildtype模式")
            eval_sample_ids = [wild_type_id]
    else:
        raise ValueError(f"不支持的EVAL_SAMPLES值: {EVAL_SAMPLES}")

    print(f"{'='*80}\n")

    # 根据评价模式选择要评估的反应集合
    if use_gems_resolution:
        print("\n" + "="*80)

        if EVALUATION_MODE == "central_60":
            # 模式1：使用固定的 60 个中心碳代谢反应
            # 先验证哪些反应在数据中存在
            available_reactions = [col for col in C13_data.columns if col != 'Id']
            unified_reactions = [rxn for rxn in CENTRAL_METABOLISM_60 if rxn in available_reactions]

            # 报告缺失的反应
            missing_reactions = [rxn for rxn in CENTRAL_METABOLISM_60 if rxn not in available_reactions]

            print(f"评价模式: 中心碳代谢反应 (central_60)")
            print(f"预期反应数: {len(CENTRAL_METABOLISM_60)} 个")
            print(f"数据中存在: {len(unified_reactions)} 个")

            if missing_reactions:
                print(f"\n警告: 以下 {len(missing_reactions)} 个反应在 13C 数据中缺失:")
                for rxn in missing_reactions:
                    print(f"  - {rxn}")
                print(f"\n建议: 请重新运行 _3_make_13C_data.py 生成包含 _reverse 反应的数据")
                print(f"      当前将仅评估数据中存在的 {len(unified_reactions)} 个反应\n")

        # elif EVALUATION_MODE == "extend_reactions":
            # 模式3：使用扩展的反应集（基于pathway_annotations + 逆反应）
            # 先验证哪些反应在数据中存在
            available_reactions = [col for col in C13_data.columns if col != 'Id']
            unified_reactions = [rxn for rxn in EXTEND_REACTIONS if rxn in available_reactions]

            # 报告缺失的反应
            missing_reactions = [rxn for rxn in EXTEND_REACTIONS if rxn not in available_reactions]

            print(f"评价模式: 扩展反应集 (extend_reactions)")
            print(f"预期反应数: {len(EXTEND_REACTIONS)} 个")
            print(f"数据中存在: {len(unified_reactions)} 个")

            if missing_reactions:
                print(f"\n警告: 以下 {len(missing_reactions)} 个反应在 13C 数据中缺失:")
                # 只显示前20个缺失反应，避免输出过长
                for rxn in missing_reactions[:20]:
                    print(f"  - {rxn}")
                if len(missing_reactions) > 20:
                    print(f"  ... 以及其他 {len(missing_reactions) - 20} 个反应")
                print(f"\n建议: 请重新运行 _3_make_13C_data.py 生成包含扩展反应的数据")
                print(f"      当前将仅评估数据中存在的 {len(unified_reactions)} 个反应\n")

        elif EVALUATION_MODE == "threshold":
            # 模式4：使用阈值过滤的反应
            # 计算所有野生型+突变型的反应流量平均值
            reaction_cols = [col for col in C13_data.columns if col != 'Id']

            # 计算所有样本（野生型+突变型）的流量平均值
            all_samples_flux = C13_data[reaction_cols].abs()  # 取绝对值
            avg_flux_per_reaction = all_samples_flux.mean(axis=0)  # 计算每个反应的平均流量

            # 筛选平均流量 > 阈值的反应
            active_reactions = avg_flux_per_reaction[avg_flux_per_reaction > FLUX_THRESHOLD].index.tolist()
            unified_reactions = active_reactions

            print(f"评价模式: 阈值过滤反应 (threshold)")
            print(f"流量阈值: {FLUX_THRESHOLD} mmol/g/h")
            print(f"总样本数（野生型+突变型）: {len(C13_data)} 个")
            print(f"活跃反应数量: {len(unified_reactions)} 个")
            print(f"过滤掉的反应数: {len(reaction_cols) - len(unified_reactions)} 个")

        # elif EVALUATION_MODE == "all_reactions":
        #     # 模式5：使用所有反应
        #     reaction_cols = [col for col in C13_data.columns if col != 'Id']
        #     unified_reactions = reaction_cols
        #     print(f"评价模式: 全部反应 (all_reactions)")
        #     print(f"反应数量: {len(unified_reactions)} 个")
        #     print(f"警告: 包含大量非活跃反应，可能导致评价指标偏高")

        # else:
        #     raise ValueError(f"不支持的评价模式: {EVALUATION_MODE}. 请选择 'central_60', 'core_30', 'extend_reactions', 'threshold', 或 'all_reactions'")

        # print("="*80)
    else:
        unified_reactions = None

    for m in methods:
        print(f"\n{'='*80}")
        print(f"评估方法: {m}")
        print(f"{'='*80}")

        # 确定模型文件路径
        if m == "FBA":
            model_file = f"../predict/{organisms}/analysis/get_kcat_mw_by_KinLLM/ecGEM/ecGEM_irr_enz_constraint.json"
            use_enzyme_model = False
        elif m == "KinLLM":
            model_file = f"../predict/{organisms}/analysis/get_kcat_mw_by_KinLLM/ecGEM/ecGEM_irr_enz_constraint.json"
            use_enzyme_model = True
        elif m == "FluxGen":
            model_file = f"../predict/{organisms}/analysis/get_kcat_mw_by_KinLLM/ecGEM/Bayesian/best_ecGEM.json"
            use_enzyme_model = True
        else:
            model_file = f"../predict/{organisms}/analysis/get_kcat_mw_by_{m}/ecGEM/ecGEM_irr_enz_constraint.json"
            use_enzyme_model = True

        # 对每个样本进行评估
        sample_losses = []
        all_pred_fluxes = []  # 存储所有样本的预测流量
        all_true_fluxes = []  # 存储所有样本的真实流量

        for sample_id in eval_sample_ids:
            # 每次循环都重新加载模型，确保状态干净
            if use_enzyme_model:
                model = get_enzyme_constraint_model(model_file)
            else:
                model = cobra.io.json.load_json_model(model_file)

            # 获取该样本的体外实验数据
            sample_vitro_data = vitro_exp[vitro_exp["Id"] == sample_id]
            if sample_vitro_data.empty:
                print(f"  警告: 样本 {sample_id} 没有体外实验数据，跳过")
                continue

            glucose_uptake = float(sample_vitro_data["Specific glucose uptake rate(mmol/g/h)"].iloc[0])
            acetate_secretion = float(sample_vitro_data["Specific acetate secretion rate(mmol/g/h)"].iloc[0])
            biomass = float(sample_vitro_data["Specific growth rate(h-1)"].iloc[0])

            # 根据物种设置约束
            if organisms == "Synechocystis sp":
                co2_uptake = 6
                photon_uptake = 100
                model.reactions.get_by_id("EX_co2_e_reverse").bounds = (0, co2_uptake)
                model.reactions.get_by_id("EX_photon_e_reverse").bounds = (0, photon_uptake)
                model.reactions.get_by_id("EX_ac_e").bounds = (0, 0)
                obj = "BIOMASS_Ec_SynAuto_1"
                model.objective = obj
            elif organisms == "iECDH1ME8569_1439":
                model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
                model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
                model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)
                model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
                # model.reactions.get_by_id("XYLI2").bounds = (0, 0)
                # model.reactions.get_by_id("XYLI2_reverse").bounds = (0, 0)
                # model.reactions.get_by_id("HEX1").bounds = (0, 0)
                model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
                model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (biomass, biomass)
            elif organisms == "Bacillus subtilis":
                model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)
                model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
                model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
                model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
                model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
                model.reactions.get_by_id("BIOMASS_BS_10").bounds = (biomass, biomass)

            # 运行FBA
            try:
                solution = model.optimize()
                if solution.status != "optimal":
                    print(f"  警告: 样本 {sample_id} FBA求解失败 (status: {solution.status})，跳过")
                    continue
            except Exception as e:
                print(f"  警告: 样本 {sample_id} FBA求解异常: {str(e)}，跳过")
                continue

            # 获取该样本的真实流量
            sample_flux_data = C13_data[C13_data['Id'] == sample_id]
            if sample_flux_data.empty:
                print(f"  警告: 样本 {sample_id} 没有13C数据，跳过")
                continue

            # 映射预测流量
            if use_gems_resolution:
                if EVALUATION_MODE == "extend_reactions":
                    active_reactions = EXTEND_REACTIONS
                result = project_flux_to_bigg_gems(
                    solution.fluxes.to_dict(),
                    active_reactions,
                    flux_threshold=FLUX_THRESHOLD
                )

                # 获取该样本的预测和真实流量
                pred_fluxes = np.array([result[rxn] for rxn in active_reactions])
                # 使用 aggregate_ground_truth_flux 来聚合同工酶反应
                agg_ground_truth = aggregate_ground_truth_flux(sample_flux_data, active_reactions)
                true_fluxes = np.array([agg_ground_truth[rxn] for rxn in active_reactions])

                # 计算该样本的MSE
                sample_mse = mean_squared_error(true_fluxes, pred_fluxes)
                sample_losses.append(sample_mse)

                # 存储用于最终计算
                all_pred_fluxes.append(pred_fluxes)
                all_true_fluxes.append(true_fluxes)

        # 计算所有样本的平均损失
        if len(sample_losses) == 0:
            print(f"  错误: 没有成功评估的样本！")
            continue

        avg_loss = np.mean(sample_losses)

        # 计算平均的预测和真实流量（用于计算整体指标）
        avg_pred_fluxes = np.mean(all_pred_fluxes, axis=0)
        avg_true_fluxes = np.mean(all_true_fluxes, axis=0)

        print(f"  {m} 评估样本数: {len(sample_losses)}")
        print(f"  {m} 使用统一评估集: {len(active_reactions)} 个反应")
        print(f"{m} 平均 MSE: {avg_loss:.6f}")
        print(f"{m} 平均 RMSE: {math.sqrt(avg_loss):.6f}")
        print(f"{m} r2 (基于平均流量): {r2_score(avg_true_fluxes, avg_pred_fluxes):.6f}")
        print(f"{m} PCC (基于平均流量): {pearsonr(avg_true_fluxes, avg_pred_fluxes)[0]:.4f}")
        print("")

        # 为了生成CSV文件，使用第一个样本（通常是wildtype）的数据
        if use_gems_resolution:
            tmp = pd.DataFrame({
                'Reaction': active_reactions,
                f'{m}': all_pred_fluxes[0]  # 使用第一个样本的预测
            })

            # 构建 ground truth DataFrame
            first_sample_data = C13_data[C13_data['Id'] == eval_sample_ids[0]]
            # 使用 aggregate_ground_truth_flux 来聚合同工酶反应
            agg_ground_truth = aggregate_ground_truth_flux(first_sample_data, active_reactions)

            gt_df = pd.DataFrame({
                'Reaction': active_reactions,
                'Value': [agg_ground_truth[rxn] for rxn in active_reactions]
            })

            if all_df is None:
                all_df = pd.merge(gt_df, tmp, on="Reaction", how="outer")
            else:
                all_df = pd.merge(all_df, tmp, on="Reaction", how="outer")

        else:
            print(f"  警告: 不支持非GEMs分辨率的多样本评估")
            continue

    # 根据评价模式保存结果到不同的文件
    output_suffix = ""
    if use_gems_resolution:
        if EVALUATION_MODE == "central_60":
            output_suffix = ""  # 默认模式，不加后缀
        elif EVALUATION_MODE == "core_30":
            output_suffix = "_core_30"  # 核心30个反应
        elif EVALUATION_MODE == "extend_reactions":
            output_suffix = "_extend_reactions"  # 扩展反应集
        elif EVALUATION_MODE == "threshold":
            output_suffix = f"_threshold_{FLUX_THRESHOLD}"
        elif EVALUATION_MODE == "all_reactions":
            output_suffix = "_all_reactions"

    all_df.to_csv(f"../predict/{organisms}/analysis/13C_analysis_data{output_suffix}.csv", index=False)

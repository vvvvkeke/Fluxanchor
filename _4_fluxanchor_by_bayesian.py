import json
import sys
import signal
import os
import re

from analysis.analysis_13C import project_flux_to_bigg, project_flux_to_bigg_gems, aggregate_ground_truth_flux, filter_reactions_by_avg_flux_threshold
from analysis.analysis_proteomics import convert_copies_fL_to_mmol_gCDW, convert_to_float, extract_gene_annotations, get_enzyme_usage, map_expression_by_reaction
sys.path.append(r'./script/')
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pyprobar import bar, probar
import numpy as np
import pandas as pd
import cobra
import logging
logging.getLogger('cobra').setLevel(logging.ERROR)
from cobra import Model
from cobra.util.solver import linear_reaction_coefficients
from script.ECMpy_function import *
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import dump, load
BAYESIAN_OPT_AVAILABLE = True

# Import fitness landscape visualizer
import matplotlib.pyplot as plt

import signal

from analysis.analysis_13C import CENTRAL_METABOLISM_60
from analysis.analysis_13C import EXTEND_REACTIONS

# =========================
# Data structures and utilities
# =========================

import math
import cobra
import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, kendalltau, spearmanr

# =========================
# Data structures and utilities
# =========================

@dataclass
class EnzymeThresholds:
    t1: float  # Upper bound threshold for rate-limiting enzymes (below t1 -> limiting)
    t2: float  # Boundary between non-limiting and ultra-high enzymes (above t2 -> ultra)
    
@dataclass
class ScaleCoeffs:
    limiting: float      # Scaling for rate-limiting enzymes
    normal: float        # Scaling for non-limiting enzymes
    ultra: float         # Scaling for ultra-fast enzymes

class TimeoutException(Exception):
    """Custom exception: FBA optimization timed out"""
    pass

def handler(signum, frame):
    raise TimeoutException("FBA optimization exceeded 10 seconds and was aborted.")

def get_result(solution, model):

    json_file = f"./predict/iECDH1ME8569_1439/iJO1366.json"  # gene names in iJO1366.json follow BiGG IDs
    ref_model = cobra.io.json.load_json_model(json_file)
    name_id_map = {}
    id_name_map = {}
    for gene in ref_model.genes:
        name_id_map[gene.name] = gene.id
        id_name_map[gene.id] = gene.name

    our_id_refseq_name_map = {}
    for gene in model.genes:
        our_id_refseq_name_map[gene.id] = gene.name
    proteins_copies_fL = pd.read_csv('./predict/iECDH1ME8569_1439/meta_abundance.copies_fL.csv', sep=",")
    proteins_mmol_gCDW = convert_copies_fL_to_mmol_gCDW(proteins_copies_fL)
    E = map_expression_by_reaction(model, proteins_mmol_gCDW, our_id_refseq_name_map, name_id_map)
    proteins = 'GLC_BATCH_mu=0.6_H' 
    E_df = E[['reaction', 'gene', 'name', 'our_name', f'{proteins}']]
    E_df = E_df.dropna()
    gene_uniprot_map, gene_reaction_map, kcat_map, kcat_MW_map = extract_gene_annotations(model)
    usages, usage_flux, usage_kcat, usage_kcat_MW, usage_method = get_enzyme_usage(model, solution, gene_uniprot_map,
                                                                                    gene_reaction_map)
    predict_proteomics = []
    real_proteomics = []
    indexes = []
    for index, raw in E_df.iterrows():
        reaction = raw['reaction']
        gene = raw['gene']
        E1 = raw[f'{proteins}']
        if usage_flux.__contains__(reaction) and usage_kcat[reaction] != "":
            v = usage_flux[reaction]
            v /= 3600
            pred = v/usage_kcat[reaction]  # v/kcat = E
            real = E1
            predict_proteomics.append(pred)
            real_proteomics.append(real)
            indexes.append(True)
        else:
            predict_proteomics.append(0)
            real_proteomics.append(E1)
            indexes.append(False)

    indexes = np.array(indexes)
    convert_func = np.vectorize(convert_to_float)
    real_proteomics = convert_func(real_proteomics)
    pred_prot = np.array(predict_proteomics)
    real_prot = np.array(real_proteomics)
    E_df['real_prot'] = real_prot * 1e6
    E_df['pred_prot'] = pred_prot * 1e6
    real_prot = real_prot[indexes]
    pred_prot = pred_prot[indexes]
    E_df = E_df[indexes]
    indexes = pred_prot != 1
    real_prot = real_prot[indexes]
    pred_prot = pred_prot[indexes]
    E_df = E_df[E_df['real_prot'] != 1]
    E_df = E_df.dropna()

    # # Count occurrences of each value in kcat column
    kcat_counts = E_df['pred_prot'].value_counts()
    # Find the most common value
    most_common_kcat = kcat_counts.idxmax()
    # Remove rows containing the most common value
    E_df = E_df[E_df['pred_prot'] != most_common_kcat]

    x = E_df['pred_prot']
    y = E_df['real_prot']
    return x, y

def calculate_proteomics_mse(solution, model) -> float:
    """Compute proteomics MSE using the same logic as analysis_proteomics.py."""
    try:
        x, y = get_result(solution, model)
    except Exception:
        return 1e9

    if len(x) == 0 or len(y) == 0:
        return 1e9

    try:
        return mean_squared_error(x, y)
    except Exception:
        return 1e9


def calculate_vivo_to_vitro_mse(
    model: cobra.Model,
    organisms: str,
    proteins_column: str = "GLC_BATCH_mu=0.6_H"
) -> float:
    """Compute the log10 MSE between vitro and vivo following analysis_vivo_vitro_kcat.py."""
    try:
        sol_result = cobra.flux_analysis.pfba(model)
    except Exception:
        return 1e9

    if getattr(sol_result, "status", None) != "optimal":
        return 1e9

    json_file = f"./predict/{organisms}/iJO1366.json"
    proteins_file = f"./predict/{organisms}/meta_abundance.copies_fL.csv"
    if not os.path.exists(json_file) or not os.path.exists(proteins_file):
        return 1e9

    ref_model = cobra.io.json.load_json_model(json_file)
    name_id_map = {gene.name: gene.id for gene in ref_model.genes}
    our_id_refseq_name_map = {gene.id: gene.name for gene in model.genes}

    proteins_copies_fL = pd.read_csv(proteins_file, sep=",")
    proteins_mmol_gCDW = convert_copies_fL_to_mmol_gCDW(proteins_copies_fL)
    E = map_expression_by_reaction(model, proteins_mmol_gCDW, our_id_refseq_name_map, name_id_map)

    required_cols = ['reaction', 'gene', 'name', 'our_name', proteins_column]
    if any(col not in E.columns for col in required_cols):
        return 1e9

    E_df = E[required_cols].dropna().copy()
    if E_df.empty:
        return 1e9

    gene_uniprot_map, gene_reaction_map, _, _ = extract_gene_annotations(model)
    usages, usage_flux, usage_kcat, _, _ = get_enzyme_usage(
        model, sol_result, gene_uniprot_map, gene_reaction_map
    )

    if len(usages) == 0:
        return 1e9

    kins = []
    kouts = []
    indexes = []

    for _, raw in E_df.iterrows():
        reaction = raw['reaction']
        enzyme_amount = raw[proteins_column]
        if usage_flux.__contains__(reaction) and usage_kcat.get(reaction, "") != "":
            v = usage_flux[reaction] / 3600  # convert to per second
            if enzyme_amount == 0:
                kins.append(0)
                kouts.append(0)
                indexes.append(False)
                continue
            kin = v / enzyme_amount
            kins.append(kin)
            kouts.append(usage_kcat[reaction])
            indexes.append(True)
        else:
            kins.append(0)
            kouts.append(0)
            indexes.append(False)

    indexes = np.array(indexes)
    if not indexes.any():
        return 1e9

    convert_func = np.vectorize(convert_to_float)
    kouts = convert_func(kouts)
    vivo_kcat = np.array(kins) + 1
    vitro_kcat = np.array(kouts) + 1

    E_df = E_df.assign(kcat_vivo=vivo_kcat, kcat_vitro=vitro_kcat)
    E_df = E_df[indexes]

    if E_df.empty:
        return 1e9

    valid_mask = E_df['kcat_vivo'] != 1
    E_df = E_df[valid_mask]

    if E_df.empty:
        return 1e9

    E_df = E_df.copy()
    E_df['kcat_vitro_log10'] = np.log10(E_df['kcat_vitro'])
    E_df['kcat_vivo_log10'] = np.log10(E_df['kcat_vivo'])
    E_df = E_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['kcat_vitro_log10', 'kcat_vivo_log10']
    )

    if E_df.empty or 'kcat_vitro' not in E_df.columns:
        return 1e9

    kcat_counts = E_df['kcat_vitro'].value_counts()
    if not kcat_counts.empty:
        most_common_kcat = kcat_counts.idxmax()
        E_df = E_df[E_df['kcat_vitro'] != most_common_kcat]

    if E_df.empty:
        return 1e9

    x = E_df['kcat_vitro_log10']
    y = E_df['kcat_vivo_log10']
    return mean_squared_error(x, y)

def piecewise_scale(enzyme_type: str, coeffs: ScaleCoeffs) -> float:
    return {
        "limiting": coeffs.limiting,
        "normal": coeffs.normal,
        "ultra": coeffs.ultra,
    }[enzyme_type]

def set_auto_model(model):
    model.reactions.EX_glc__D_e.bounds = (0, 0)  # forbid glucose uptake
    # model.reactions.EX_fru_e.bounds = (0, 0)  # forbid fructose uptake
    # model.reactions.EX_glcglyc_e.bounds = (0, 0)  # forbid glycerol uptake
    # model.reactions.EX_sucr_e.bounds = (0, 0)  # forbid sucrose uptake
    # Use nitrate as the sole nitrogen source [-0.008,-0.05]
    # model.reactions.get_by_id('EX_no3_e').lower_bound = -8000
    # Configure light conditions
    model.reactions.get_by_id('EX_photon_e').lower_bound = -10000
    model.reactions.get_by_id('EX_photon_e').upper_bound = 10000
    # Use CO2 as the only carbon source
    model.reactions.get_by_id('EX_co2_e').lower_bound = -100  # specify CO2 uptake [-0.001, -0.25]
    return model

def trans_ecGEM_model(model_path, model, reaction_kcat_mw: pd.DataFrame, pop_index,
                      f=0.4, ptot=0.56 , sigma=0.5,
                      lowerbound=0, upperbound=0.35,   # 0.35?
                      synauto=True, save_best_model=False, count=1):
    json_path = f"{model_path}/irreversible.json"
    if count == 1:    ### first call uses this branch to keep flux settings consistent
        # print(f"---------------------------{count} first run")
        # if synauto:
        #     model = set_auto_model(model)
        # else:
        #     model.reactions.get_by_id("EX_ac_e").bounds = (-10, 100)

        convert_to_irreversible(model)
        model = isoenzyme_split(model)
        # model_name = model_file.split('/')[-1].split('.')[0]
        cobra.io.save_json_model(model, json_path)

    dictionary_model = json_load(json_path)
    dictionary_model['enzyme_constraint'] = {'enzyme_mass_fraction': f, 'total_protein_fraction': ptot,
                                             'average_saturation': sigma, 'lowerbound': lowerbound,
                                             'upperbound': upperbound}

    matched_count = 0
    for eachreaction in range(len(dictionary_model['reactions'])):
        reaction_id = dictionary_model['reactions'][eachreaction]['id']
        # if re.search('_num',reaction_id):
        #    reaction_id=reaction_id.split('_num')[0]
        if reaction_id in reaction_kcat_mw.index:
            dictionary_model['reactions'][eachreaction]['kcat'] = reaction_kcat_mw.loc[reaction_id, 'kcat']
            dictionary_model['reactions'][eachreaction]['kcat_MW'] = reaction_kcat_mw.loc[reaction_id, 'kcat_MW']
            dictionary_model['reactions'][eachreaction]['data_type'] = reaction_kcat_mw.loc[reaction_id, 'data_type']
            dictionary_model['reactions'][eachreaction]['km'] = reaction_kcat_mw.loc[reaction_id, 'km']
            matched_count += 1
        else:
            dictionary_model['reactions'][eachreaction]['kcat'] = ''
            dictionary_model['reactions'][eachreaction]['kcat_MW'] = ''
            dictionary_model['reactions'][eachreaction]['km'] = ''

    if save_best_model:
        json_write(f"{model_path}/best_ecGEM.json", dictionary_model)
        enz_model = get_enzyme_constraint_model(f"{model_path}/best_ecGEM.json")
    else:
        json_write(f"{model_path}/{pop_index}_temp_ecGEM.json", dictionary_model)
        enz_model = get_enzyme_constraint_model(f"{model_path}/{pop_index}_temp_ecGEM.json")
    return enz_model


def vitro_to_vivo(
    base_model: cobra.Model,
    enzyme_df: pd.DataFrame,
    thresholds: EnzymeThresholds,
    coeffs: ScaleCoeffs,
    pop_index,
    model_path = None,
    save = False,
    synauto = False,
    count = 1
):
    temp_df = enzyme_df.copy()
    non_rate_threshold = thresholds.t1
    super_threshold = thresholds.t2
    rate_coff = coeffs.limiting
    non_rate_coff = coeffs.normal
    super_coff = coeffs.ultra
    # rate_coff = 1
    # non_rate_coff = 1
    # super_coff = 1
    # data = enzyme_df['catalytic_efficiency']
    # Iterate with iterrows() and adjust each row
    for index, row in temp_df.iterrows():
        if row['catalytic_efficiency'] > super_threshold:  # ultra-fast enzyme
            temp_df.at[index, 'kcat'] = row['kcat'] * super_coff
            temp_df.at[index, 'kcat_MW'] = row['kcat_MW'] * super_coff
        elif non_rate_threshold < row['catalytic_efficiency'] <= super_threshold:  # non-limiting enzyme
            temp_df.at[index, 'kcat'] = row['kcat'] * non_rate_coff
            temp_df.at[index, 'kcat_MW'] = row['kcat_MW'] * non_rate_coff
        else:  # rate-limiting enzyme
            temp_df.at[index, 'kcat'] = row['kcat'] * rate_coff
            temp_df.at[index, 'kcat_MW'] = row['kcat_MW'] * rate_coff

    # mask_s = temp_df['catalytic_efficiency'] > super_threshold
    # mask_n = (temp_df['catalytic_efficiency'] > non_rate_threshold) & ~mask_s

    # temp_df.loc[mask_s, ['kcat', 'kcat_MW']] *= super_coff
    # temp_df.loc[mask_n, ['kcat', 'kcat_MW']] *= non_rate_coff
    # temp_df.loc[~(mask_s | mask_n), ['kcat', 'kcat_MW']] *= rate_coff

    # Save best result
    if save:
        temp_df = temp_df.reset_index().rename(columns={'index': 'reactions'})
        cols = ['reactions'] + [c for c in temp_df.columns if c != 'reactions']
        temp_df = temp_df[cols]
        temp_df.to_csv(f"{model_path}/best_enzyme_df.csv", index=False)
        temp_df = pd.read_csv(f"{model_path}/best_enzyme_df.csv", index_col=0)

    model = trans_ecGEM_model(model_path, base_model,
                              temp_df, pop_index, save_best_model=save, synauto=synauto, count=count)

    return model

def fba_predict(model: cobra.Model) -> Dict[str, float]:
    def handler(signum, frame):
        raise TimeoutError("pFBA timed out")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(15)  # 15-second timeout
    try:
        sol = model.optimize()
        signal.alarm(0)
        if sol.status != "optimal":
            return 1e9
        return sol.fluxes.to_dict()
    except TimeoutError:
        return 1e9
    except:
        return 1e9

def flux_loss_13C(
    pred_flux: Dict[str, float],
    flux_df: pd.DataFrame,
    organisms
) -> float:
    """
    Compute the flux loss for a single sample.
    pred_flux: dictionary of predicted fluxes
    flux_df: DataFrame with the ground-truth flux (single row)
    """
    # Extract ground-truth reaction list from flux_df
    ground_truth_reactions = flux_df['BIGG id'].tolist()

    agg_pred = project_flux_to_bigg_gems(pred_flux, ground_truth_reactions)
    tmp = pd.DataFrame({f'BIGG id': agg_pred.keys(), f'pred': agg_pred.values()})
    all_df = pd.merge(flux_df, tmp, on="BIGG id", how="outer")

    # Create a sort key based on the order of BIGG IDs in tmp (matches evaluation)
    sort_order = {bigg_id: i for i, bigg_id in enumerate(tmp["BIGG id"])}
    all_df["sort_key"] = all_df["BIGG id"].map(sort_order)
    all_df = all_df.sort_values("sort_key").drop("sort_key", axis=1)

    # Process data: absolute value and fill NaNs with zero
    all_df['Value'] = all_df['Value'].astype(float).abs()
    all_df['pred'] = all_df['pred'].fillna(0.0).astype(float).abs()

    true_data = all_df['Value'].astype(float).abs()
    pred_data = all_df['pred'].astype(float).abs()

    return mean_squared_error(true_data, pred_data)


def calculate_training_loss(
    model_path: str,
    train_samples: List[str],
    flux_data_all: pd.DataFrame,
    active_reactions: List[str],
    vitro_exp_df: pd.DataFrame,
    wild_type_id: str,
    organisms: str,
    gene_mapping: Dict[str, str] = None,
    train_obj: str = "default"
) -> float:
    """
    Calculate the average loss for the training set.

    Args:
    model_path: path to model file (used for reloading)
    train_samples: list of sample IDs in the training set
    flux_data_all: DataFrame containing 13C fluxes for all samples
    active_reactions: list of reactions to evaluate
    vitro_exp_df: in vitro experimental data
    wild_type_id: identifier for the wild-type sample
    organisms: organism name
    gene_mapping: gene mapping dictionary (for knockouts)
    train_obj: training objective hyperparameter, options include:
        - "default": legacy loss (13C flux MSE only)
        - "overflow": adds overflow-related penalty terms
          sample_loss = sample_loss + (EX_ac_e - ac_e_true) + (BIOMASS - biomass_true)
        - "vivo_to_vitro": use the kcat comparison MSE from analysis_vivo_vitro_kcat.py
        - "proteomics": use the proteomics MSE from analysis_proteomics.py

    Returns:
    Mean MSE loss for the training samples.
    """
    from analysis.analysis_13C import project_flux_to_bigg_gems

    if gene_mapping is None:
        gene_mapping = {}

    sample_losses = []

    # Optimization: load the model only once if evaluating wild type alone (no knockouts needed)
    only_wildtype = (len(train_samples) == 1 and train_samples[0] == wild_type_id)

    if only_wildtype:
        # Single shared model instance for all wild-type evaluations
        shared_model = get_enzyme_constraint_model(model_path)

    for sample_id in train_samples:
        try:
            # Reload model only when gene knockouts are required
            if only_wildtype:
                # Reuse shared model for wild type
                sample_model = shared_model
            else:
                # Reload per mutant to avoid lingering knockouts
                sample_model = get_enzyme_constraint_model(model_path)

            # Retrieve sample-specific constraints from vitro_exp_df
            sample_vitro_data = vitro_exp_df[vitro_exp_df["Id"] == sample_id]

            if sample_vitro_data.empty:
                # Skip if no in vitro data available
                continue

            # Apply sample-specific bounds
            biomass = float(sample_vitro_data["Specific growth rate(h-1)"].iloc[0])
            glucose_uptake = float(sample_vitro_data["Specific glucose uptake rate(mmol/g/h)"].iloc[0])
            acetate_secretion = float(sample_vitro_data["Specific acetate secretion rate(mmol/g/h)"].iloc[0])

            # Set organism-specific constraints (align with evaluation)
            if organisms == "Bacillus subtilis":
                sample_model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)
                sample_model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
                sample_model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
                sample_model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
                if train_obj != "overflow":
                    sample_model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
                    sample_model.reactions.get_by_id("BIOMASS_BS_10").bounds = (biomass, biomass)
                #
                # sample_model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
                # sample_model.reactions.get_by_id("BIOMASS_BS_10").bounds = (biomass, biomass)
            elif organisms == "iECDH1ME8569_1439":
                sample_model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
                sample_model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
                sample_model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)
                sample_model.reactions.get_by_id("EX_glc__D_e").bounds = (0, 0)  # forbid glucose export
                sample_model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
                # overflow training does not consider acetate secretion and biomass constraints
                if train_obj != "overflow":
                    sample_model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
                    sample_model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (biomass, biomass)
                # sample_model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
                # sample_model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (biomass, biomass)

            # Perform gene knockouts for mutants
            if sample_id != wild_type_id:
                # Determine which genes to knock out
                genes_to_knockout = find_genes_to_knockout(sample_model, sample_id, gene_mapping, organisms)

                if not genes_to_knockout:
                    # Skip if no matching gene found
                    continue

                # Execute gene knockouts
                for gene in genes_to_knockout:
                    gene.knock_out()

            # Solve FBA
            solution = sample_model.optimize()

            if solution.status != "optimal":
                # Solver failed; penalize heavily
                return 1e9

            if train_obj == "proteomics":
                sample_loss = calculate_proteomics_mse(solution, sample_model)
                sample_losses.append(sample_loss)
                continue

            if train_obj == "vivo_to_vitro":
                sample_loss = calculate_vivo_to_vitro_mse(sample_model, organisms)
                sample_losses.append(sample_loss)
                continue

            # print(f"-----{solution['EX_ac_e']}")
            # print(f"-----{solution['BIOMASS_Ec_iJO1366_core_53p95M']}")
            # Retrieve ground-truth flux for this sample
            sample_flux_data = flux_data_all[flux_data_all['Id'] == sample_id]

            if sample_flux_data.empty:
                continue

            agg_ground_truth = aggregate_ground_truth_flux(sample_flux_data, active_reactions)
            flux_df = pd.DataFrame({
                'BIGG id': list(agg_ground_truth.keys()),
                'Value': list(agg_ground_truth.values())
            })
            flux_df = flux_df.dropna()

            # Compute loss for this sample
            pred_flux_dict = solution.fluxes.to_dict()
            sample_loss = flux_loss_13C(pred_flux_dict, flux_df, organisms)

            # Add overflow penalty if requested
            if train_obj == "overflow":
                # Real acetate secretion and biomass
                ac_e_true = acetate_secretion
                biomass_true = biomass
                # Predicted values
                ac_e_pred = solution['EX_ac_e']
                biomass_pred = solution['BIOMASS_Ec_iJO1366_core_53p95M']
                # Add overflow loss term
                overflow_loss = abs(ac_e_pred - ac_e_true) + abs(biomass_pred - biomass_true)
                sample_loss = sample_loss + overflow_loss * 2

            sample_losses.append(sample_loss)

        except Exception as e:
            # Failure on one sample should not halt others
            print(f"  Warning: sample {sample_id} failed: {str(e)}")
            continue

    # If every sample failed, return a large loss
    if len(sample_losses) == 0:
        return 1e9

    # Return the mean loss across successful samples
    return np.mean(sample_losses)


def find_genes_to_knockout(model: cobra.Model, gene_name: str,
                           gene_mapping: Dict[str, str], organism: str) -> List:
    """Find genes to knock out (logic adapted from test utilities)."""
    genes_to_knockout = []

    # Handle multi-gene knockouts (e.g., ytsJ_gapB)
    if '_' in gene_name and organism == "Bacillus subtilis":
        gene_names = gene_name.split('_')
    else:
        gene_names = [gene_name]

    for single_gene in gene_names:
        # Find mapped gene ID
        target_gene_id = None
        gene_name_lower = single_gene.lower()

        if gene_name_lower in gene_mapping:
            target_gene_id = gene_mapping[gene_name_lower]
        else:
            target_gene_id = single_gene

        # Search for the gene in the model
        for gene in model.genes:
            if (gene.id == target_gene_id or
                gene.name == target_gene_id or
                (gene.annotation.get("refseq_synonym") and target_gene_id in gene.annotation["refseq_synonym"]) or
                (gene.annotation.get("refseq_name") and gene.annotation["refseq_name"] == target_gene_id) or
                gene.id == single_gene or
                gene.name == single_gene or
                (gene.annotation.get("refseq_synonym") and single_gene in gene.annotation["refseq_synonym"]) or
                (gene.annotation.get("refseq_name") and gene.annotation["refseq_name"] == single_gene)):
                genes_to_knockout.append(gene)
                break

    return genes_to_knockout

def flux_loss(model):
    try:
        sol = cobra.flux_analysis.pfba(model)
        if sol.status != "optimal":
            return 1e9
        x, y = get_result(sol, model)
        return mean_squared_error(x, y)
    except:
        return 1e9
    

@dataclass
class BayesianOptConfig:
    n_calls: int = 50  # total evaluations for Bayesian optimization
    n_initial_points: int = 10  # number of random initial points
    acq_func: str = 'EI'  # acquisition function: 'EI', 'LCB', 'PI'
    xi: float = 0.01  # exploration-exploitation trade-off
    kappa: float = 1.96  # LCB parameter
    random_state: int = 42
    verbose: bool = True

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def encode_params(th: EnzymeThresholds, sc: ScaleCoeffs) -> np.ndarray:
    return np.array([th.t1, th.t2, sc.limiting, sc.normal, sc.ultra], dtype=float)

def bayesian_optimize(
    base_model: Model,
    enzyme_df: pd.DataFrame,
    flux_df: pd.DataFrame,
    bounds: List[Tuple[float, float]],
    model_path,
    cfg: BayesianOptConfig = BayesianOptConfig(),
    organisms = "",
    synauto = False,
    model_suffix = "",
    train_samples: List[str] = None,
    flux_data_all: pd.DataFrame = None,
    active_reactions: List[str] = None,
    vitro_exp_df: pd.DataFrame = None,
    wild_type_id: str = None,
    gene_mapping: Dict[str, str] = None,
    train_obj: str = "default"
):
    """
    Optimize enzyme parameters via Bayesian optimization.
    Tune t1, t2, limiting_scale, normal_scale, and ultra_scale to minimize loss.
    """

    def decode_params(vec: np.ndarray) -> Tuple[EnzymeThresholds, ScaleCoeffs]:
        t1, t2, a, b, c = vec
        return EnzymeThresholds(t1=t1, t2=t2), ScaleCoeffs(limiting=a, normal=b, ultra=c)

    # Define search space
    dimensions = [
        Real(bounds[0][0], bounds[0][1], name='t1'),
        Real(bounds[1][0], bounds[1][1], name='t2'),
        Real(bounds[2][0], bounds[2][1], name='limiting'),
        Real(bounds[3][0], bounds[3][1], name='normal'),
        Real(bounds[4][0], bounds[4][1], name='ultra')
    ]

    # Objective function
    @use_named_args(dimensions)
    def objective(**params):
        if not hasattr(objective, 'call_count'):
            objective.call_count = 0
            objective.first_call_params = None
            objective.best_loss = float('inf')
            objective.best_model_path = None

        objective.call_count += 1
        # Convert parameters to vector
        vec = np.array([params['t1'], params['t2'], params['limiting'],
                       params['normal'], params['ultra']])

        try:
            th, sc = decode_params(vec)

            if cfg.verbose:
                print(f"Evaluating: t1={th.t1:.3f}, t2={th.t2:.3f}, "
                      f"limiting={sc.limiting:.3f}, normal={sc.normal:.3f}, ultra={sc.ultra:.3f}")

            # Build model and compute loss
            pop_model = vitro_to_vivo(base_model, enzyme_df, th, sc,
                                    pop_index=0, model_path=model_path, synauto=synauto, count = objective.call_count)

            # Use the training-loss helper that iterates across samples,
            # applies knockouts/FBA, and averages the results
            if train_samples is not None and flux_data_all is not None:
                # Save current model to a temporary file
                temp_model_file = f"{model_path}/0_temp_ecGEM.json"
                loss = calculate_training_loss(
                    model_path=temp_model_file,
                    train_samples=train_samples,
                    flux_data_all=flux_data_all,
                    active_reactions=active_reactions,
                    vitro_exp_df=vitro_exp_df,
                    wild_type_id=wild_type_id,
                    organisms=organisms,
                    gene_mapping=gene_mapping,
                    train_obj=train_obj
                )
            else:
                # Backward-compatible branch (legacy path)
                pred = fba_predict(pop_model)
                
                if pred == 1e9:
                    loss = 1e9
                else:
                    loss = flux_loss_13C(pred, flux_df, organisms)

            # Save model if a new best loss is achieved
            if loss < objective.best_loss and loss != 1e9:
                objective.best_loss = loss
                import shutil
                shutil.copy(f"{model_path}/0_temp_ecGEM.json", f"{model_path}/best_during_opt_ecGEM.json")
                if cfg.verbose:
                    print(f"  New best loss {loss:.6f}, saved model")

            # loss = flux_loss(pop_model)
            # if cfg.verbose:
            #     print(f"Loss: {loss:.6f}")

            # # Penalty terms
            viol_0 = max(0, th.t1 - th.t2 + 1e-6)  # enforce th.t1 < th.t2
            viol_1 = max(0, sc.limiting - sc.normal + 1e-6)  # enforce sc.limiting < sc.normal
            viol_2 = max(0, sc.normal - sc.ultra + 1e-6)     # enforce sc.normal < sc.ultra
            penalty = 1e9 * (viol_0 + viol_1 + viol_2)  # sufficiently large linear penalty
            return loss + penalty
            # return loss

        except Exception as e:
            if cfg.verbose:
                print(f"Error during evaluation: {e}")
            return 1e9  # return large penalty if evaluation fails

    # Run Bayesian optimization
    if cfg.verbose:
        print("Starting Bayesian optimization...")
        print(f"Parameter space: t1 in {bounds[0]}, t2 in {bounds[1]}, limiting in {bounds[2]}, "
              f"normal in {bounds[3]}, ultra in {bounds[4]}")
        print(f"Total evaluations: {cfg.n_calls}, initial random points: {cfg.n_initial_points}")

    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=cfg.n_calls,
        n_initial_points=cfg.n_initial_points,
        acq_func=cfg.acq_func,
        xi=cfg.xi,
        kappa=cfg.kappa,
        random_state=cfg.random_state,
        verbose=cfg.verbose
    )

    # Retrieve best parameters
    best_params = np.array(result.x)
    best_loss = result.fun

    if cfg.verbose:
        print(f"\n=== Bayesian optimization complete ===")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Best params: {best_params}")
        print(f"Total function evaluations: {len(result.func_vals)}")

    # Build and save final model using best parameters
    th, sc = decode_params(best_params)

    # Copy the best model saved during optimization instead of rebuilding it
    # This guarantees identical model state to the one achieving best loss
    print("\n=== Saving best model ===")
    import shutil
    import os
    if os.path.exists(f"{model_path}/best_during_opt_ecGEM.json"):
        shutil.copy(f"{model_path}/best_during_opt_ecGEM.json", f"{model_path}/best_ecGEM{model_suffix}.json")
        print(f"Copied best model from optimization to best_ecGEM{model_suffix}.json")

        # Save the corresponding enzyme_df as well
        temp_df = enzyme_df.copy()
        for index, row in temp_df.iterrows():
            if row['catalytic_efficiency'] > th.t2:
                temp_df.at[index, 'kcat'] = row['kcat'] * sc.ultra
                temp_df.at[index, 'kcat_MW'] = row['kcat_MW'] * sc.ultra
            elif th.t1 < row['catalytic_efficiency'] <= th.t2:
                temp_df.at[index, 'kcat'] = row['kcat'] * sc.normal
                temp_df.at[index, 'kcat_MW'] = row['kcat_MW'] * sc.normal
            else:
                temp_df.at[index, 'kcat'] = row['kcat'] * sc.limiting
                temp_df.at[index, 'kcat_MW'] = row['kcat_MW'] * sc.limiting

        temp_df = temp_df.reset_index().rename(columns={'index': 'reactions'})
        cols = ['reactions'] + [c for c in temp_df.columns if c != 'reactions']
        temp_df = temp_df[cols]
        temp_df.to_csv(f"{model_path}/best_enzyme_df{model_suffix}.csv", index=False)
        print(f"Saved best_enzyme_df{model_suffix}.csv")
    else:
        print("WARNING: best_during_opt_ecGEM.json not found, reconstructing best model...")
        _ = vitro_to_vivo(base_model, enzyme_df, th, sc,
                         model_path=model_path, save=True, pop_index=None, synauto=synauto, count=999)

    # Validate the saved best model
    print("\n=== Validating best model ===")
    best_model_file = f"{model_path}/best_ecGEM{model_suffix}.json"

    # Reuse the exact loss calculation from the optimization phase
    if train_samples is not None and flux_data_all is not None:
        # Multi-sample training mode: call calculate_training_loss
        test_loss = calculate_training_loss(
            model_path=best_model_file,
            train_samples=train_samples,
            flux_data_all=flux_data_all,
            active_reactions=active_reactions,
            vitro_exp_df=vitro_exp_df,
            wild_type_id=wild_type_id,
            organisms=organisms,
            gene_mapping=gene_mapping,
            train_obj=train_obj
        )
    else:
        # Single-sample (wild type) mode: simple FBA verification
        test_model = get_enzyme_constraint_model(best_model_file)
        test_pred = fba_predict(test_model)
        if test_pred != 1e9:
            test_loss = flux_loss_13C(test_pred, flux_df, organisms)
        else:
            test_loss = 1e9

    print(f"Best model loss: {test_loss:.6f}")
    print(f"Optimization best loss: {best_loss:.6f}")

    if test_loss != 1e9:
        if abs(test_loss - best_loss) < 0.01:
            print("SUCCESS: model validation passed!")
        else:
            print("WARNING: loss mismatch; potential issue detected")
            print(f"  Difference: {abs(test_loss - best_loss):.6f}")
    else:
        print("WARNING: model solve failed!")

    # Do not call fba_predict here because constraints may be incomplete
    # best_loss already contains the optimal loss from optimization

    # Persist optimization summary
    optimization_result = {
        'x': result.x,
        'fun': result.fun,
        'func_vals': result.func_vals,
        'x_iters': result.x_iters
    }

    # with open(f"{model_path}/bayesian_opt_result.pkl", "wb") as f:
    #     dump(result, f)

    return {
        "best_loss": best_loss,
        "best_params_vec": best_params.tolist(),
        "best_thresholds": th,
        "best_scales": sc,
        "optimization_result": optimization_result
    }


# =========================
# Main workflow
# =========================

def main():

    TRAIN_OBJ = "default"
    # TRAIN_OBJ = "overflow"  # focus on overflow objective
    # TRAIN_OBJ = "vivo_to_vitro"  # use vivo_to_vitro objective
    # TRAIN_OBJ = "proteomics"  # use proteomics objective
    # organisms = "Synechocystis sp"
    organisms = "iECDH1ME8569_1439"
    # organisms = "Bacillus subtilis"
    print(organisms)
    m = "KinLLM_Bayesian"  # rename method to highlight Bayesian optimization

    # ================================================================================
    # Training data selection hyperparameters
    # ================================================================================
    # Options:
    #   "wildtype"       - train only on the wild type (default behavior)
    #   "customize"      - use a custom list of samples (see CUSTOMIZE_SAMPLES)
    #   numeric (0.0-1.0) - fraction reserved for the test set (e.g., 0.3 -> 30% test)
    TRAIN_DATA_RATIO = "wildtype"  # default to wild type only
    # TRAIN_DATA_RATIO = "customize"
    # TRAIN_DATA_RATIO = 0.8  # e.g., 80% train / 20% test

    # Custom sample list (used only when TRAIN_DATA_RATIO == "customize")
    # Sample names must match the Id column in 13C_data.csv
    CUSTOMIZE_SAMPLES = ["WILD TYPE", "Pgi", "Zwf"]  # example: wild type + two mutants

    # Random seed to keep train/test split reproducible
    RANDOM_SEED = 42

    # ================================================================================
    # Training objective hyperparameter
    # ================================================================================
    # Options:
    #   "default"      - original loss logic (13C flux MSE only)
    #   "overflow"     - adds overflow penalty:
    #                    sample_loss += (EX_ac_e - ac_e_true) + (BIOMASS - biomass_true)
    #   "vivo_to_vitro" - use MSE from analysis_vivo_vitro_kcat.py
    #   "proteomics"   - use proteomics MSE from analysis_proteomics.py

    # Determine model filename suffix based on TRAIN_DATA_RATIO
    if TRAIN_DATA_RATIO == "wildtype":
        MODEL_SUFFIX = ""  # default name for wild-type mode
    elif TRAIN_DATA_RATIO == "customize":
        MODEL_SUFFIX = "_customize"
    else:
        MODEL_SUFFIX = f"_{TRAIN_DATA_RATIO}"

    # Append suffix for special training objectives
    if TRAIN_OBJ == "overflow":
        MODEL_SUFFIX = MODEL_SUFFIX + "_overflow"
    elif TRAIN_OBJ == "vivo_to_vitro":
        MODEL_SUFFIX = MODEL_SUFFIX + "_vivo"
    elif TRAIN_OBJ == "proteomics":
        MODEL_SUFFIX = MODEL_SUFFIX + "_prote"

    # ================================================================================
    # Evaluation mode (consistent with analysis_13C.py)
    # ================================================================================
    # Options:
    #   "central_60"       - fixed set of 60 central carbon reactions
    #   "core_30"          - core 30 reactions defined in project_flux_to_bigg
    #   "extend_reactions" - extended set (pathway annotations + reverse reactions)
    #   "threshold"        - reactions filtered by average flux > FLUX_THRESHOLD
    # EVALUATION_MODE = "threshold"
    # EVALUATION_MODE = "central_60"
    # EVALUATION_MODE = "core_30"
    EVALUATION_MODE = "extend_reactions"
    # EVALUATION_MODE = "threshold" 

    # Flux threshold (only used when EVALUATION_MODE == "threshold")
    FLUX_THRESHOLD = 0.01  # mmol/g/h

    file_path = f"./predict/{organisms}/analysis/get_kcat_mw_by_KinLLM"
    model_path = f"{file_path}/ecGEM/ecGEM_irr_enz_constraint.json"          # or .json
    enzyme_csv = f"{file_path}/reaction_kcat_MW.csv" # columns: reaction_id,kcat_over_km(,enzyme_id)
    # Load flux data from the WILD TYPE row in 13C_data.csv
    flux_data_csv = f"./predict/{organisms}/13C_data.csv"
    best_model_path = f"{file_path}/ecGEM/Bayesian"  # renamed output folder
    if not os.path.exists(best_model_path):
        os.mkdir(best_model_path)
    vitro_exp_df = pd.read_csv(f"./predict/{organisms}/vitro_exp_data.csv", sep=",")
    # Load the raw model with cobra instead of get_enzyme_constraint_model
    # because the raw JSON lacks kcat_MW constraints
    base_model = cobra.io.json.load_json_model(model_path)
    enzyme_df = pd.read_csv(enzyme_csv, index_col=0)

    # Read 13C_data.csv and filter according to EVALUATION_MODE (same as analysis_13C.py)
    flux_data_all = pd.read_csv(flux_data_csv)

    # Define wild-type IDs for each organism
    WILD_TYPE_IDS = {
        "Synechocystis sp": "WILD TYPE",
        "iECDH1ME8569_1439": "WILD TYPE",
        "Bacillus subtilis": "Wild-type"
    }
    wild_type_id = WILD_TYPE_IDS.get(organisms, "WILD TYPE")

    # ================================================================================
    # Data splitting logic
    # ================================================================================
    print(f"\n{'='*80}")
    print("Training data selection strategy")
    print(f"{'='*80}")

    if TRAIN_DATA_RATIO == "wildtype":
        # Mode 1: wild type only
        print("Mode: wild type only")
        print(f"Wild type ID: {wild_type_id}")

        flux_data_wide = flux_data_all[flux_data_all['Id'] == wild_type_id]
        if flux_data_wide.empty:
            raise ValueError(f"{wild_type_id} not found in 13C_data.csv")

        train_samples = [wild_type_id]
        test_samples = []

        print("Training sample count: 1 (wild type only)")

    elif TRAIN_DATA_RATIO == "customize":
        # Mode 2: custom list
        print("Mode: custom sample list")
        print(f"Custom samples: {CUSTOMIZE_SAMPLES}")

        # Validate custom samples
        all_sample_ids = flux_data_all['Id'].tolist()
        missing_samples = [s for s in CUSTOMIZE_SAMPLES if s not in all_sample_ids]
        if missing_samples:
            raise ValueError(f"Custom samples missing in 13C_data.csv: {missing_samples}\n"
                           f"Available samples: {all_sample_ids}")

        train_samples = CUSTOMIZE_SAMPLES
        test_samples = []

        print(f"Training sample count: {len(train_samples)}")
        print(f"Training samples: {train_samples}")

        # Use the mean flux across training samples as optimization target
        flux_data_train = flux_data_all[flux_data_all['Id'].isin(train_samples)]

        # Average flux across training samples
        reaction_cols = [col for col in flux_data_train.columns if col != 'Id']
        flux_data_wide = flux_data_train[reaction_cols].mean().to_frame().T
        flux_data_wide['Id'] = 'TRAIN_AVG'

    else:
        # Mode 3: random split (wild type always in training)
        test_ratio = float(TRAIN_DATA_RATIO)
        if not 0.0 < test_ratio < 1.0:
            raise ValueError(f"TRAIN_DATA_RATIO must be between 0 and 1 (got {test_ratio})")

        print("Mode: random split")
        print(f"Test ratio: {test_ratio*100:.1f}%")
        print(f"Training ratio: {(1-test_ratio)*100:.1f}%")
        print(f"Random seed: {RANDOM_SEED}")
        print(f"Note: wild type ({wild_type_id}) always in training")

        # Gather IDs and split wild type vs mutants
        all_sample_ids = flux_data_all['Id'].tolist()

        if wild_type_id not in all_sample_ids:
            raise ValueError(f"{wild_type_id} not found in 13C_data.csv")

        mutant_ids = [sid for sid in all_sample_ids if sid != wild_type_id]

        n_mutants = len(mutant_ids)
        n_test = int(n_mutants * test_ratio)
        n_train_mutants = n_mutants - n_test

        # Reproducible shuffle
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        shuffled_indices = np.random.permutation(n_mutants)

        test_indices = shuffled_indices[:n_test]
        train_mutant_indices = shuffled_indices[n_test:]

        test_samples = [mutant_ids[i] for i in test_indices]
        train_mutant_samples = [mutant_ids[i] for i in train_mutant_indices]

        # Training set = wild type + selected mutants
        train_samples = [wild_type_id] + train_mutant_samples

        print(f"Total samples: {len(all_sample_ids)} (1 wild type + {n_mutants} mutants)")
        print(f"Training samples: {len(train_samples)} (1 wild type + {n_train_mutants} mutants)")
        print(f"Test samples: {len(test_samples)} (mutants only)")
        print(f"Training list: {train_samples[:5]}{'...' if len(train_samples) > 5 else ''}")
        print(f"Test sample list: {test_samples[:5]}{'...' if len(test_samples) > 5 else ''}")

        # Extract training data and compute mean flux for optimization
        flux_data_train = flux_data_all[flux_data_all['Id'].isin(train_samples)]

        # Average flux for all training samples
        reaction_cols = [col for col in flux_data_train.columns if col != 'Id']
        flux_data_wide = flux_data_train[reaction_cols].mean().to_frame().T
        flux_data_wide['Id'] = 'TRAIN_AVG'

    print(f"{'='*80}\n")

    # Extract reaction columns (excluding Id)
    reaction_cols = [col for col in flux_data_wide.columns if col != 'Id']

    print(f"\n{'='*80}")
    print("Evaluation mode selection")
    print(f"{'='*80}")

    # Select reactions to evaluate based on the chosen mode
    if EVALUATION_MODE == "central_60":
        active_reactions = CENTRAL_METABOLISM_60
        print("Mode: central carbon (central_60)")
        print(f"Reactions: {len(active_reactions)}")
        print("Includes: core carbon metabolism plus key amino acid/nucleotide precursors")

    elif EVALUATION_MODE == "extend_reactions":
        active_reactions = EXTEND_REACTIONS
        print("Mode: extended reaction set (extend_reactions)")
        print(f"Reactions: {len(active_reactions)}")
        print("Includes: glycolysis, PPP, TCA, amino acids, fatty acids, nucleotides, etc.")

    elif EVALUATION_MODE == "threshold":
        active_reactions = filter_reactions_by_avg_flux_threshold(flux_data_all, FLUX_THRESHOLD)
        print("Mode: threshold-based reactions")
        print(f"Flux threshold: {FLUX_THRESHOLD} mmol/g/h")
        print(f"Total samples (wild type + mutants): {len(flux_data_all)}")
        print(f"Active reactions: {len(active_reactions)}")
        print(f"Filtered out reactions: {len(reaction_cols) - len(active_reactions)}")

    else:
        raise ValueError(f"Unsupported evaluation mode: {EVALUATION_MODE}. "
                         "Choose from 'central_60', 'core_30', 'extend_reactions', or 'threshold'")

    print(f"{'='*80}")

    print(f"\n=== 13C Ground Truth Statistics ===")
    print(f"Total reactions (ground truth): {len(reaction_cols)}")
    print(f"Reactions used: {len(active_reactions)}")

    agg_ground_truth = aggregate_ground_truth_flux(flux_data_wide, active_reactions)
    flux_df = pd.DataFrame({
        'BIGG id': list(agg_ground_truth.keys()),
        'Value': list(agg_ground_truth.values())
    })
    flux_df = flux_df.dropna()

    print(f"Note: optimization and evaluation will use these {len(flux_df)} reactions\n")
    if organisms == "Synechocystis sp":
        co2_uptake = 6
        photon_uptake = 100
        base_model.reactions.get_by_id("EX_co2_e_reverse").bounds = (0, co2_uptake)
        base_model.reactions.get_by_id("EX_photon_e_reverse").bounds = (0, photon_uptake)
        base_model.reactions.get_by_id("EX_ac_e").bounds = (0, 0)
        # base_model.reactions.get_by_id("EX_succ_e").bounds = (0, 0)
        base_model.reactions.get_by_id("TALA").bounds = (0, 100)
        # base_model.reactions.get_by_id("GLGC").bounds = (0, 0)
        obj = "BIOMASS_Ec_SynAuto_1"
        base_model.objective = obj
        synauto = True
        # param_bounds = [
        #     (1e4, 1e5),    # t1: rate-limiting threshold (small)
        #     (1e5, 1e8),    # t2: ultra threshold (>> t1)
        #     (1e-2, 1e-1),  # limiting scale: small
        #     (1e-1, 5),     # normal scale: near 1
        #     (1, 5),        # ultra scale: large
        # ]
        param_bounds = [  
            (1e4, 1e5),    # t1: rate-limiting threshold
            (1e5, 1e6),    # t2: ultra threshold (> t1)
            (1e-3, 1),     # limiting scale
            (1e-1, 5),     # normal scale
            (3, 10),       # ultra scale
        ]
        # Bayesian optimization configuration (increase n_calls for better results at higher cost)
        cfg = BayesianOptConfig(
            n_calls=150,  # total evaluations
            n_initial_points=10,  # random initial points
            acq_func='EI',  # Expected Improvement
            xi=0.01,  # exploration parameter
            random_state=42,
            verbose=True
        )
    elif organisms == "iECDH1ME8569_1439":  # 6.9697332676007635
        synauto = False
        # biomass_yield = float(flux_df[flux_df["BIGG id"] == "Biomass yield"]["Value"])
        # acetate_secretion = float(flux_df[flux_df["BIGG id"] == "acetate secretion"]["Value"])
        glucose_uptake = float(vitro_exp_df[vitro_exp_df["Id"] == "WILD TYPE"]["Specific glucose uptake rate(mmol/g/h)"])
        acetate_secretion = float(vitro_exp_df[vitro_exp_df["Id"] == "WILD TYPE"]["Specific acetate secretion rate(mmol/g/h)"])
        biomass = float(vitro_exp_df[vitro_exp_df["Id"] == "WILD TYPE"]["Specific growth rate(h-1)"])
        base_model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
        base_model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
        # base_model.reactions.get_by_id("PYK_num1").bounds = (9.98, 10)
        # base_model.reactions.get_by_id("GLCptspp_num1").bounds = (8.12, 10) 
        # base_model.reactions.get_by_id("PGI").bounds = (5.71, 5.71)
        # base_model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (0.41, 0.41)
        
        ############################## Constraints per training objective ###################################################
        if TRAIN_OBJ == "default":
            base_model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)  # glucose uptake
            base_model.reactions.get_by_id("EX_glc__D_e").bounds = (0, 0)  # forbid glucose export
            base_model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
            base_model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
            base_model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (biomass, biomass)
            
            # param_bounds = [
            #     (1e4, 1e5),    # t1: limiting enzyme threshold
            #     (1e6, 1e7),    # t2: ultra enzyme threshold
            #     (1e-2, 1),     # limiting scale
            #     (1e-2, 5),     # normal scale
            #     (1, 8),        # ultra scale
            # ]

            param_bounds = [   
                (1e3, 1e4),    # t1: limiting enzyme threshold  
                (1e5, 4e6),     # t2: ultra enzyme threshold
                (1e-2, 1),      # limiting scale
                (5e-1, 1),      # normal scale
                (8e-1, 2),      # ultra scale
            ]
            cfg = BayesianOptConfig(
                n_calls=150,
                n_initial_points=10,
                acq_func='EI',
                xi=0.01,
                random_state=42,
                verbose=True
            )
        elif TRAIN_OBJ == "vivo_to_vitro":
            base_model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (0, glucose_uptake)
            base_model.reactions.get_by_id("EX_glc__D_e").bounds = (0, 0)
            base_model.reactions.get_by_id("EX_ac_e").bounds = (0, 100)
            base_model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (0, 100)
            param_bounds = [
                (1e2, 1e3),
                (1e3, 1e6),
                (1e-2, 1e-1),
                (0.1, 1),
                (1, 7),
            ]
            cfg = BayesianOptConfig(
                n_calls=20,
                n_initial_points=10,
                acq_func='EI',
                xi=0.01,
                random_state=42,
                verbose=True
            )
        elif TRAIN_OBJ == "proteomics":
            base_model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (0, glucose_uptake)
            base_model.reactions.get_by_id("EX_glc__D_e").bounds = (0, 0)
            base_model.reactions.get_by_id("EX_ac_e").bounds = (0, 100)
            base_model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (0, 100)
            param_bounds = [
                (1e2, 1e3),
                (1e3, 1e6),
                (1e-2, 1e-1),
                (0.1, 1),
                (1, 7),
            ]
            cfg = BayesianOptConfig(
                n_calls=20,
                n_initial_points=10,
                acq_func='EI',
                xi=0.01,
                random_state=42,
                verbose=True
            )
        elif TRAIN_OBJ == "overflow":
        ############################## Overflow scenario without limiting acetate #####################################################
            base_model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (0, glucose_uptake)
            base_model.reactions.get_by_id("EX_glc__D_e").bounds = (0, 0)  # forbid glucose output
            base_model.reactions.get_by_id("EX_ac_e").bounds = (0, 100)
            base_model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (0, 100)

            param_bounds = [  # proteomics     
                (1e2, 1e3),    # t1: limiting enzyme threshold  
                (1e3, 1e6),     # t2: ultra enzyme threshold
                (1e-2, 1e-1),      # limiting scale
                (0.1, 1),      # normal scale
                (1, 7),      # ultra scale
            ]
            # # # param_bounds = [  # proteomics     3.1653
            # # #     (1e2, 1e3),    # t1: limiting enzyme threshold  
            # # #     (1e3, 1e6),     # t2: ultra enzyme threshold
            # # #     (1e-2, 1e-1),      # limiting scale
            # # #     (0.1, 1),      # normal scale
            # # #     (1, 2),      # ultra scale
            # # # ]
            cfg = BayesianOptConfig( 
                n_calls=20,  # total evaluations (adjust as needed)
                n_initial_points=10,  # random initial points
                acq_func='EI',  # acquisition function: Expected Improvement
                xi=0.01,  # exploration parameter
                random_state=42,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported TRAIN_OBJ: {TRAIN_OBJ}")
    elif organisms == "Bacillus subtilis":
        # Read wild-type experimental data from vitro_exp
        glucose_uptake = float(vitro_exp_df[vitro_exp_df["Id"] == wild_type_id]["Specific glucose uptake rate(mmol/g/h)"])
        acetate_secretion = float(vitro_exp_df[vitro_exp_df["Id"] == wild_type_id]["Specific acetate secretion rate(mmol/g/h)"])
        biomass = float(vitro_exp_df[vitro_exp_df["Id"] == wild_type_id]["Specific growth rate(h-1)"])

        base_model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)  # glucose uptake
        base_model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
        base_model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
        base_model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
        base_model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
        base_model.reactions.get_by_id("BIOMASS_BS_10").bounds = (biomass, biomass)
        
        param_bounds = [  #  26.955980781080697
            (100, 1e4),    # t1: limiting enzyme threshold  
            (1e5, 1e6),     # t2: ultra enzyme threshold
            (0.005, 1e-1),      # limiting scale
            (1e-1, 5),      # normal scale
            (1, 10),      # ultra scale
        ]

        # param_bounds = [  # 26.992324
        #     (1e3, 1e4),    # t1: limiting enzyme threshold  
        #     (1e5, 1e6),     # t2: ultra enzyme threshold
        #     (1e-2, 1),      # limiting scale
        #     (1e-1, 5),      # normal scale
        #     (3, 10),      # ultra scale
        # ]

        synauto = False
        # Bayesian optimization config (increase n_calls for better accuracy but more time)
        cfg = BayesianOptConfig(
            n_calls=20,  # total evaluations
            n_initial_points=10,  # random initial points
            acq_func='EI',  # Expected Improvement
            xi=0.01,  # exploration parameter
            random_state=42,
            verbose=True
        )

    

    # if use_flux_ratio and organisms == "iECDH1ME8569_1439":
    #     print("\n=== Add 13C Flux Ratio constraints ===")
    #     flux_ratio_file = "./predict/iECDH1ME8569_1439/flux_ratio_constrain.csv"
    #     base_model, constraints_added = add_flux_ratio_constraints(base_model, flux_ratio_file)
    #     print(f"Added {constraints_added} flux ratio constraints\n")

    # Generate irreversible.json once before optimization to ensure a consistent model structure
    print("\n=== Preprocessing: build irreversible model ===")
    dummy_th = EnzymeThresholds(t1=1000, t2=10000)
    dummy_sc = ScaleCoeffs(limiting=1.0, normal=1.0, ultra=1.0)
    _ = vitro_to_vivo(base_model, enzyme_df, dummy_th, dummy_sc,
                     model_path=best_model_path, save=False, pop_index=-1, synauto=synauto, count=1)
    print("irreversible.json created, starting optimization...\n")

    # Load gene mapping file (for knockouts)
    print("\n=== Loading gene mapping file ===")
    gene_mapping = {}
    if organisms == "iECDH1ME8569_1439":
        gene_mapping_file = "./predict/iECDH1ME8569_1439/goal_blast.blast"
    elif organisms == "Bacillus subtilis":
        gene_mapping_file = None  # no mapping file for Bacillus subtilis
    else:
        gene_mapping_file = None

    if gene_mapping_file and os.path.exists(gene_mapping_file):
        try:
            with open(gene_mapping_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            test_gene = parts[0].strip()
                            target_gene_info = parts[1].strip()
                            # Extract gene ID
                            if '|' in target_gene_info:
                                target_gene_id = target_gene_info.split('|')[0]
                            else:
                                target_gene_id = target_gene_info
                            gene_mapping[test_gene.lower()] = target_gene_id
            print(f"Loaded {len(gene_mapping)} gene mappings")
        except Exception as e:
            print(f"WARNING: unable to read gene mapping file {gene_mapping_file}: {str(e)}")
    else:
            print("No gene mapping file found; using original gene names")

    result = bayesian_optimize(
        base_model=base_model,
        enzyme_df=enzyme_df,
        flux_df=flux_df,
        bounds=param_bounds,
        model_path=best_model_path,
        cfg=cfg,
        organisms=organisms,
        synauto=synauto,
        model_suffix=MODEL_SUFFIX,
        train_samples=train_samples,
        flux_data_all=flux_data_all,
        active_reactions=list(flux_df['BIGG id']),
        vitro_exp_df=vitro_exp_df,
        wild_type_id=wild_type_id,
        gene_mapping=gene_mapping,
        train_obj=TRAIN_OBJ
    )

    print("\n=== Bayesian Optimization Results ===")
    print("Best loss:", result["best_loss"])
    print("Best thresholds:", result["best_thresholds"])
    print("Best scales:", result["best_scales"])

    # # Save parameters and data split info
    save = {
        "best_loss": result["best_loss"],
        "best_params_vec": result["best_params_vec"],
        "best_thresholds": result["best_thresholds"].__dict__,
        "best_scales": result["best_scales"].__dict__,
        "train_data_ratio": TRAIN_DATA_RATIO,
        "random_seed": RANDOM_SEED,
        "train_samples": train_samples,
        "test_samples": test_samples
    }
    with open(f"{best_model_path}/best_params.json", "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)

    # Save train/test sample IDs for reference
    split_info = pd.DataFrame({
        'Sample_ID': train_samples + test_samples,
        'Split': ['train'] * len(train_samples) + ['test'] * len(test_samples)
    })
    split_info.to_csv(f"{best_model_path}/data_split.csv", index=False)
    print(f"\nData split saved to: {best_model_path}/data_split.csv")

    # Report optimization diagnostics
    if "optimization_result" in result:
        opt_result = result["optimization_result"]
        print(f"Function evaluations: {len(opt_result['func_vals'])}")
        print(f"Best loss history (first 10): {opt_result['func_vals'][:10]}...")

    # Load best model
    best_model_file = f"{best_model_path}/best_ecGEM{MODEL_SUFFIX}.json"

    print(f"\n=== Final evaluation ===")
    print("Using the same evaluation procedure as optimization")
    print(f"Evaluation samples: {train_samples}")

    # Apply the identical loss calculation
    final_loss = calculate_training_loss(
        model_path=best_model_file,
        train_samples=train_samples,
        flux_data_all=flux_data_all,
        active_reactions=list(flux_df['BIGG id']),
        vitro_exp_df=vitro_exp_df,
        wild_type_id=wild_type_id,
        organisms=organisms,
        gene_mapping=gene_mapping,
        train_obj=TRAIN_OBJ
    )

    print(f"\nFinal evaluation loss (MSE): {final_loss:.6f}")
    print(f"Best loss during optimization: {result['best_loss']:.6f}")

    if abs(final_loss - result['best_loss']) < 0.01:
        print("Final evaluation matches optimization result!")
    else:
        diff = abs(final_loss - result['best_loss'])
        print(f"WARNING: final evaluation differs from optimization (delta={diff:.6f})")

    # Run a dedicated wild-type prediction for detailed reporting
    print(f"\nGenerating detailed predictions (based on {wild_type_id})...")

    # Reload model to ensure a clean state
    eval_model = get_enzyme_constraint_model(f"{best_model_path}/best_ecGEM{MODEL_SUFFIX}.json")

    # Extract wild-type experimental data
    wt_vitro_data = vitro_exp_df[vitro_exp_df["Id"] == wild_type_id].iloc[0]
    glucose_uptake = float(wt_vitro_data["Specific glucose uptake rate(mmol/g/h)"])
    acetate_secretion = float(wt_vitro_data["Specific acetate secretion rate(mmol/g/h)"])
    biomass = float(wt_vitro_data["Specific growth rate(h-1)"])

    # Apply constraints (same as calculate_training_loss)
    if organisms == "Bacillus subtilis":
        eval_model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)
        eval_model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
        eval_model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
        eval_model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
        eval_model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
        eval_model.reactions.get_by_id("BIOMASS_BS_10").bounds = (biomass, biomass)
    elif organisms == "iECDH1ME8569_1439":
        eval_model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
        eval_model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
        eval_model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)
        eval_model.reactions.get_by_id("EX_glc__D_e").bounds = (0, 0)  # forbid glucose export
        eval_model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
        eval_model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
        eval_model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (biomass, biomass)
    elif organisms == "Synechocystis sp":
        eval_model.reactions.get_by_id("EX_co2_e_reverse").bounds = (0, co2_uptake)
        eval_model.reactions.get_by_id("EX_photon_e_reverse").bounds = (0, photon_uptake)
        eval_model.reactions.get_by_id("EX_ac_e").bounds = (0, 0)
        eval_model.reactions.get_by_id("EX_succ_e").bounds = (0, 0)
        eval_model.objective = "BIOMASS_Ec_SynAuto_1"

    solution1 = eval_model.optimize()

    # Extract ground-truth reaction list from flux_df
    ground_truth_reactions = flux_df['BIGG id'].tolist()

    print(f"Number of ground-truth reactions used: {len(flux_df)}")

    res = project_flux_to_bigg_gems(solution1.fluxes.to_dict(), ground_truth_reactions)
    tmp = pd.DataFrame({f'BIGG id': res.keys(), f'{m}': res.values()})
    all_df = pd.merge(flux_df, tmp, on="BIGG id", how="outer")

    # Create a sort key based on BIGG ID order in tmp
    sort_order = {bigg_id: i for i, bigg_id in enumerate(tmp["BIGG id"])}

    # Sort all_df according to that key
    all_df["sort_key"] = all_df["BIGG id"].map(sort_order)
    all_df = all_df.sort_values("sort_key").drop("sort_key", axis=1)

    # Take absolute values and fill NaNs with zero (align with loss calculation)
    all_df['Value'] = all_df['Value'].astype(float).abs()
    all_df[f'{m}'] = all_df[f'{m}'].fillna(0.0).astype(float).abs()

    true_data = all_df['Value'].astype(float).abs()
    pred_data = all_df[f'{m}'].astype(float).abs()

    print(f"\nDetailed metrics for {wild_type_id}:")
    print(f"{m} RMSE: {math.sqrt(mean_squared_error(true_data, pred_data))}")
    print(f"{m} MSE: {mean_squared_error(true_data, pred_data)}")
    print(f"{m} r2: {r2_score(true_data, pred_data)}")
    print(f"{m} PCC: {pearsonr(true_data, pred_data)[0]:.4f}")
    print("")

    all_df.to_csv(f"./predict/{organisms}/analysis/13C_analysis_bayesian.csv", index=False)

    # Save additional files depending on evaluation mode
    output_suffix = ""
    if EVALUATION_MODE == "central_60":
        output_suffix = ""
    elif EVALUATION_MODE == "core_30":
        output_suffix = "_core_30"
    elif EVALUATION_MODE == "extend_reactions":
        output_suffix = "_extend_reactions"
    elif EVALUATION_MODE == "threshold":
        output_suffix = f"_threshold_{FLUX_THRESHOLD}"

    # Save a copy with the evaluation-mode suffix if available
    if output_suffix:
        all_df.to_csv(f"./predict/{organisms}/analysis/13C_analysis_bayesian{output_suffix}.csv", index=False)
        print("\nResults saved:")
        print(f"  - Default file: ./predict/{organisms}/analysis/13C_analysis_bayesian.csv")
        print(f"  - Suffixed file: ./predict/{organisms}/analysis/13C_analysis_bayesian{output_suffix}.csv")
        print(f"  - Evaluation mode: {EVALUATION_MODE}")
        if EVALUATION_MODE == "threshold":
            print(f"  - Flux threshold: {FLUX_THRESHOLD} mmol/g/h")
        print(f"  - Reactions: {len(active_reactions)}")
    else:
        print("\nResults saved:")
        print(f"  - File: ./predict/{organisms}/analysis/13C_analysis_bayesian.csv")
        print(f"  - Evaluation mode: {EVALUATION_MODE}")
        print(f"  - Reactions: {len(active_reactions)}")
if __name__ == "__main__":
    main()

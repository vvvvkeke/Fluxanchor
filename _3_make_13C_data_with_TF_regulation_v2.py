#!/usr/bin/env python
"""
Generate ecGEM-based 13C flux data (supports transcription factor regulation) - V2

Supports multiple organisms:
- E. coli (Escherichia coli)
- B. subtilis (Bacillus subtilis)

What's improved:
1. TF knockouts are simulated by constraining reaction bounds (upper/lower) instead of editing kcat_MW.
2. This is biologically consistent: regulatory genes change enzyme expression and thus the maximum reaction rate.
3. kcat_MW stays unchanged because it is an intrinsic kinetic parameter of the enzyme.

Regulation implementation:
- activation / repression keep their standard biological meanings.
- effect_strength denotes the remaining target activity/expression fraction after TF knockout.
- Activation: TF knockout -> loss of positive regulation -> enzyme expression decreases -> lower the reaction upper bound.
- Repression: TF knockout -> repression is released (de-repression) -> current implementation leaves bounds unchanged.

Usage:
python _3_make_13C_data_with_TF_regulation_v2.py
"""

import sys
sys.path.append(r'./')
sys.path.append(r'./script/')

import pandas as pd
import numpy as np
import cobra
import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Callable
import os

# Import regulatory rule module - E. coli
from tf_regulatory_rules_v2 import (
    get_regulatory_effects as get_regulatory_effects_ecoli,
    get_enzyme_knockout_reactions as get_enzyme_knockout_reactions_ecoli,
    is_transcription_factor as is_transcription_factor_ecoli,
    is_metabolic_enzyme as is_metabolic_enzyme_ecoli,
    REGULATORY_RULES_V2 as REGULATORY_RULES_ECOLI,
    ENZYME_KNOCKOUTS as ENZYME_KNOCKOUTS_ECOLI,
)

# Import regulatory rule module - B. subtilis
# from tf_regulatory_rules_bsubtilis import (
#     get_regulatory_effects_bsubtilis,
#     get_enzyme_knockout_reactions_bsubtilis,
#     is_transcription_factor_bsubtilis,
#     is_metabolic_enzyme_bsubtilis,
#     REGULATORY_RULES_BSUBTILIS,
#     ENZYME_KNOCKOUTS_BSUBTILIS,
# )

# Import ECMpy helpers
from script.ECMpy_function import json_load, json_write


# =============================================================================
# Multi-organism configuration
# =============================================================================

ORGANISM_CONFIGS = {
    "E. coli": {
        "name": "E. coli (Escherichia coli)",
        "wild_type_id": "WILD TYPE",
        "biomass_reaction": "BIOMASS_Ec_iJO1366_core_53p95M",
        "vitro_exp_file": "./predict/iECDH1ME8569_1439/vitro_exp_data.csv",
        "flux_data_file": "./predict/iECDH1ME8569_1439/flux_data.csv",
        "flux_ratio_constraint_file": "./predict/iECDH1ME8569_1439/flux_ratio_constrain.csv",
        "model_file": "./predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian/best_ecGEM.json",
        "output_file": "./predict/iECDH1ME8569_1439/13C_data_TF_regulated_v2.csv",
        "get_regulatory_effects": get_regulatory_effects_ecoli,
        "get_enzyme_knockout_reactions": get_enzyme_knockout_reactions_ecoli,
        "is_transcription_factor": is_transcription_factor_ecoli,
        "is_metabolic_enzyme": is_metabolic_enzyme_ecoli,
        "regulatory_rules": REGULATORY_RULES_ECOLI,
        "enzyme_knockouts": ENZYME_KNOCKOUTS_ECOLI,
        "flux_data_sep": "\t",
    },
    # "B. subtilis": {
    #     "name": "B. subtilis (Bacillus subtilis)",
    #     "wild_type_id": "Wild-type",
    #     "biomass_reaction": "BIOMASS_BS_10",
    #     "vitro_exp_file": "./predict/Bacillus subtilis/vitro_exp_data.csv",
    #     "flux_data_file": "./predict/Bacillus subtilis/flux_data.csv",
    #     "flux_ratio_constraint_file": "./predict/Bacillus subtilis/flux_ratio_constrain.csv",
    #     "model_file": "./predict/Bacillus subtilis/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian/best_ecGEM.json",
    #     "output_file": "./predict/Bacillus subtilis/13C_data_TF_regulated_v2.csv",
    #     "get_regulatory_effects": get_regulatory_effects_bsubtilis,
    #     "get_enzyme_knockout_reactions": get_enzyme_knockout_reactions_bsubtilis,
    #     "is_transcription_factor": is_transcription_factor_bsubtilis,
    #     "is_metabolic_enzyme": is_metabolic_enzyme_bsubtilis,
    #     "regulatory_rules": REGULATORY_RULES_BSUBTILIS,
    #     "enzyme_knockouts": ENZYME_KNOCKOUTS_BSUBTILIS,
    #     "flux_data_sep": "\t",
    # },
}


def find_reactions_producing_metabolite(model, metabolite_id: str) -> List[Tuple[str, float]]:
    """
    Find all reactions in the metabolic model that produce the given metabolite.
    """
    producing_reactions = []

    try:
        metabolite = model.metabolites.get_by_id(metabolite_id)
    except KeyError:
        print(f"    Warning: Metabolite {metabolite_id} not found")
        return producing_reactions

    for reaction in metabolite.reactions:
        stoich_coeff = reaction.metabolites[metabolite]
        if stoich_coeff > 0:
            producing_reactions.append((reaction.id, stoich_coeff))

    return producing_reactions


def identify_product_metabolite(model, numerator_reaction_ids: List[str], pathway_name: str = None) -> str:
    """
    Identify the key metabolite produced by the numerator reactions.
    """
    if pathway_name:
        pathway_lower = str(pathway_name).lower()
        if "through glycolysis" in pathway_lower or "through pp pathway" in pathway_lower:
            if 'g3p_c' in model.metabolites:
                print(f"    Identified key metabolite: g3p_c (from pathway name: {pathway_name})")
                return 'g3p_c'

    reaction_patterns = {
        'PYK': 'pyr_c',
        'PPCK': 'pep_c',
        'PEPCK': 'oaa_c',
        'PC': 'oaa_c',
        'GHMT2r': 'ser__L_c',
        'SHMT': 'ser__L_c',
        'GLYCL': 'ser__L_c',
        'PGCD': 'ser__L_c',
        'PSERT': 'ser__L_c',
        'PSP_L': 'ser__L_c',
    }

    for rxn_id in numerator_reaction_ids:
        base_rxn_id = rxn_id.replace('_reverse', '')
        for pattern, metabolite_id in reaction_patterns.items():
            if pattern in base_rxn_id.upper():
                if metabolite_id in model.metabolites:
                    print(f"    Identified key metabolite: {metabolite_id} (based on reaction {rxn_id})")
                    return metabolite_id

    for rxn_id in numerator_reaction_ids:
        try:
            if rxn_id in model.reactions:
                reaction = model.reactions.get_by_id(rxn_id)
            elif rxn_id.replace('_reverse', '') in model.reactions:
                reaction = model.reactions.get_by_id(rxn_id.replace('_reverse', ''))
            else:
                continue

            for metabolite, coeff in reaction.metabolites.items():
                if coeff > 0 and metabolite.id.endswith('_c'):
                    met_id = metabolite.id
                    if any(key in met_id for key in ['pyr', 'pep', 'oaa', 'ser']):
                        print(f"    Identified key metabolite from reaction products: {met_id} (reaction {rxn_id})")
                        return met_id
        except Exception as e:
            continue

    print(f"    Warning: Unable to identify key metabolite; numerator reactions: {numerator_reaction_ids}")
    return None


def add_flux_ratio_constraints(model, flux_ratio_file):
    """
    Add the 13C flux ratio constraints to the model.
    """
    df = pd.read_csv(flux_ratio_file, header=None)

    pathway_names = df.iloc[0].values
    flux_ratios = df.iloc[1].values
    numerators = df.iloc[2].values
    target_metabolites = df.iloc[3].values if len(df) > 3 else [None] * len(pathway_names)

    print("Adding flux ratio constraints to model...")
    print("-" * 80)

    constraints_added = 0

    for i in range(len(pathway_names)):
        pathway = pathway_names[i]
        ratio = flux_ratios[i]
        num_str = numerators[i]
        target_met = target_metabolites[i] if i < len(target_metabolites) else None

        if pd.isna(ratio) or pd.isna(num_str):
            continue

        try:
            ratio = float(ratio)
        except:
            continue

        pathway_lower = str(pathway).lower()
        if "upper bound" in pathway_lower:
            constraint_type = "upper"
        elif "lower bound" in pathway_lower:
            constraint_type = "lower"
        else:
            constraint_type = "equality"

        num_reactions = [r.strip() for r in str(num_str).split(',') if r.strip()]

        def find_reaction_variants(rxn_id, model):
            variants = []
            if rxn_id in model.reactions:
                variants.append((rxn_id, 1))
            elif f"{rxn_id}_reverse" in model.reactions:
                variants.append((f"{rxn_id}_reverse", -1))
            elif rxn_id.endswith("_reverse"):
                base_id = rxn_id[:-8]
                if base_id in model.reactions:
                    variants.append((base_id, -1))
            return variants

        num_rxns_valid = []
        for rxn_id in num_reactions:
            variants = find_reaction_variants(rxn_id, model)
            if variants:
                num_rxns_valid.extend(variants)
            else:
                print(f"Warning: Reaction {rxn_id} not found in model (numerator for {pathway})")

        if not num_rxns_valid:
            print(f"Skipping constraint {pathway}: no valid numerator reactions")
            continue

        if pd.notna(target_met) and str(target_met).strip():
            metabolite_id = str(target_met).strip()
            print(f"  Using specified target metabolite: {metabolite_id}")
        else:
            metabolite_id = identify_product_metabolite(model, num_reactions, pathway_name=pathway)

        if not metabolite_id:
            print(f"Skipping constraint {pathway}: unable to identify target metabolite")
            continue

        producing_reactions = find_reactions_producing_metabolite(model, metabolite_id)

        if not producing_reactions:
            print(f"Skipping constraint {pathway}: no reactions found that produce {metabolite_id}")
            continue

        print(f"\nProcessing constraint {i+1}: {pathway}")
        print(f"  Target metabolite: {metabolite_id}")
        print(f"  Found {len(producing_reactions)} reactions producing the metabolite")

        num_rxn_ids = set([rxn_id for rxn_id, _ in num_rxns_valid])
        other_rxns_valid = []

        for rxn_id, stoich in producing_reactions:
            if rxn_id in num_rxn_ids:
                continue
            base_rxn_id = rxn_id.replace('_reverse', '')
            if base_rxn_id in num_rxn_ids:
                continue
            other_rxns_valid.append((rxn_id, 1))

        print(f"  Other reactions (excluding numerator-producing reactions): {len(other_rxns_valid)}")

        coefficients = {}

        for rxn_id, direction_coeff in num_rxns_valid:
            rxn = model.reactions.get_by_id(rxn_id)
            coefficients[rxn.forward_variable] = (1 - ratio) * direction_coeff

        for rxn_id, direction_coeff in other_rxns_valid:
            rxn = model.reactions.get_by_id(rxn_id)
            if rxn.forward_variable in coefficients:
                coefficients[rxn.forward_variable] += (-ratio * direction_coeff)
            else:
                coefficients[rxn.forward_variable] = -ratio * direction_coeff

        if constraint_type == "upper":
            lb = None
            ub = 0
            constraint_desc = f"<= {ratio}"
        elif constraint_type == "lower":
            lb = 0
            ub = None
            constraint_desc = f">= {ratio}"
        else:
            lb = 0
            ub = 0
            constraint_desc = f"= {ratio}"

        constraint = model.problem.Constraint(
            0,
            lb=lb,
            ub=ub,
            name=f"flux_ratio_{i+1}_{pathway.replace(' ', '_')[:50]}"
        )

        model.add_cons_vars(constraint)
        model.solver.update()
        constraint.set_linear_coefficients(coefficients=coefficients)

        print(f"  Added constraint [{constraint_type.upper()}]: numerator/(numerator+other) {constraint_desc}")

        num_rxn_names = [f"{rxn_id}{'(rev)' if dir_coef == -1 else ''}"
                        for rxn_id, dir_coef in num_rxns_valid]
        print(f"  Numerator reactions: {', '.join(num_rxn_names)}")

        if other_rxns_valid:
            other_rxn_names = [f"{rxn_id}{'(rev)' if dir_coef == -1 else ''}"
                             for rxn_id, dir_coef in other_rxns_valid[:5]]
            if len(other_rxns_valid) > 5:
                other_rxn_names.append(f"... (total {len(other_rxns_valid)})")
            print(f"  Other reactions: {', '.join(other_rxn_names)}")
        else:
            print(f"  Other reactions: (none)")

        constraints_added += 1

    print("-" * 80)
    print(f"Total number of flux ratio constraints added: {constraints_added}")

    return model, constraints_added


def create_flux_ratio_constraints_temp_file(flux_ratio_constraint_file, flux_ratios_row, output_temp_file):
    """
    Build a temporary flux ratio constraint file based on the flux ratio data.
    """
    constraint_df = None
    for encoding in ['utf-8', 'gb18030', 'gbk', 'latin1']:
        try:
            constraint_df = pd.read_csv(flux_ratio_constraint_file, index_col=0, encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue

    if constraint_df is None:
        raise ValueError(f"Unable to read {flux_ratio_constraint_file}; tried multiple encodings without success")

    with open(output_temp_file, 'w') as f:
        pathway_name_strs = []
        for col_name in constraint_df.columns:
            if ',' in str(col_name):
                pathway_name_strs.append(f'"{col_name}"')
            else:
                pathway_name_strs.append(str(col_name))
        pathway_names = ','.join([''] + pathway_name_strs)
        f.write(pathway_names + '\n')

        ratio_values = []
        for col in constraint_df.columns:
            if col in flux_ratios_row.index:
                ratio_values.append(str(flux_ratios_row[col]))
            else:
                print(f"    Warning: Pathway '{col}' not found in flux_data, using default 0.5")
                ratio_values.append('0.5')
        f.write(',' + ','.join(ratio_values) + '\n')

        reactions = constraint_df.loc['reaction'].values
        reaction_strs = []
        for r in reactions:
            if pd.isna(r):
                reaction_strs.append('')
            elif ',' in str(r):
                reaction_strs.append(f'"{r}"')
            else:
                reaction_strs.append(str(r))
        f.write(',' + ','.join(reaction_strs) + '\n')

        if 'other reaction' in constraint_df.index:
            other_reactions = constraint_df.loc['other reaction'].values
            metabolite_strs = []
            for r in other_reactions:
                if pd.isna(r):
                    metabolite_strs.append('')
                else:
                    r_str = str(r).strip()
                    if ',' in r_str:
                        metabolite_strs.append('')
                    else:
                        metabolite_strs.append(r_str)
            f.write(',' + ','.join(metabolite_strs) + '\n')
        else:
            f.write(',' + ','.join([''] * len(constraint_df.columns)) + '\n')

    return output_temp_file


def get_reaction_flux_reference(model: cobra.Model, dictionary_model: dict) -> Dict[str, float]:
    """
    Get the reference flux upper bound for each reaction in the wild-type model.

    This function estimates the theoretical maximum flux for each reaction based on enzyme constraints:
    flux_max = kcat_MW * E_total

    Args:
    model: COBRA model
    dictionary_model: JSON dictionary for the model (contains kcat_MW and enzyme constraint info)

    Returns:
    Dict[str, float]: {reaction_id: reference_flux_upper_bound}
    """
    reference_flux = {}

    # Get total enzyme constraint
    if 'enzyme_constraint' in dictionary_model:
        E_total = dictionary_model['enzyme_constraint']['upperbound']
    else:
        E_total = 0.227  # default

    # Compute the theoretical maximum flux for each reaction
    for eachr in dictionary_model['reactions']:
        rxn_id = eachr['id']
        kcat_mw = eachr.get('kcat_MW', '')

        if kcat_mw and kcat_mw != '':
            try:
                # Theoretical maximum flux = kcat_MW * E_total
                # This can be large; we just want the relative limitation ratio
                reference_flux[rxn_id] = float(kcat_mw)
            except (ValueError, TypeError):
                pass

    return reference_flux


def apply_tf_regulatory_effects_by_bounds(
    model: cobra.Model,
    tf_name: str,
    glucose_uptake: float = 10.0,
    config: Dict = None
) -> Tuple[cobra.Model, List[str], Dict[str, Tuple[float, float]]]:
    """
    Apply TF knockout regulatory effects by constraining reaction bounds (V2 version).

    Key idea:
    - activation / repression keep their standard biological meanings.
    - effect_strength denotes the remaining target activity/expression fraction after TF knockout.
    - Activation regulation: TF knockout -> loss of positive regulation -> reduce reaction upper bound.
    - Repression regulation: TF knockout -> de-repression -> current implementation keeps bounds unchanged.

    Args:
    model: COBRA model
    tf_name: transcription factor name
    glucose_uptake: glucose uptake rate (used to estimate reference flux)
    config: organism configuration dictionary

    Returns:
    model: modified model
    affected_reactions: list of affected reactions
    bounds_changes: dict recording bound changes {rxn_id: (old_ub, new_ub)}
    """
    # Use the function from the configuration to obtain regulatory rules
    get_regulatory_effects = config.get('get_regulatory_effects') if config else get_regulatory_effects_ecoli
    rules = get_regulatory_effects(tf_name)
    affected_reactions = []
    bounds_changes = {}

    if not rules:
        print(f"    No regulatory rules found for transcription factor {tf_name}")
        return model, affected_reactions, bounds_changes

    print(f"    Applying regulatory effects for TF {tf_name} (bounds-limiting mode)...")

    # Estimate reference flux based on glucose uptake.
    # Most central metabolic reactions have fluxes comparable to or smaller than glucose uptake.
    reference_flux = glucose_uptake * 2  # use twice the glucose uptake as a reference

    for rule in rules:
        if rule.regulation_type == "activation" and rule.effect_strength < 1.0:
            for base_rxn_id in rule.target_reactions:
                matching_rxns = []
                for rxn in model.reactions:
                    rxn_base = rxn.id.replace('_reverse', '').split('_num')[0]
                    if rxn_base == base_rxn_id or rxn.id == base_rxn_id:
                        matching_rxns.append(rxn)

                for rxn in matching_rxns:
                    old_lb, old_ub = rxn.bounds

                    if old_ub > 0:
                        new_ub = min(old_ub, reference_flux * rule.effect_strength)

                        new_ub = max(0, new_ub)

                        rxn.bounds = (old_lb, new_ub)
                        bounds_changes[rxn.id] = (old_ub, new_ub)

                        print(f"      {rxn.id}: upper_bound {old_ub:.2f} -> {new_ub:.2f} "
                              f"(effect: {rule.effect_strength:.0%})")
                        affected_reactions.append(rxn.id)

                    elif old_lb < 0:
                        new_lb = max(old_lb, -reference_flux * rule.effect_strength)

                        rxn.bounds = (new_lb, old_ub)
                        bounds_changes[rxn.id] = (old_lb, new_lb)

                        print(f"      {rxn.id}: lower_bound {old_lb:.2f} -> {new_lb:.2f} "
                              f"(effect: {rule.effect_strength:.0%})")
                        affected_reactions.append(rxn.id)

        elif rule.regulation_type == "repression":
            for base_rxn_id in rule.target_reactions:
                print(f"      {base_rxn_id}: de-repressed (negative-regulation target; TF knockout restores or increases expression, bounds unchanged in current implementation)")
                affected_reactions.append(base_rxn_id)

    return model, affected_reactions, bounds_changes


def apply_tf_regulatory_effects_by_relative_bounds(
    model: cobra.Model,
    dictionary_model: dict,
    tf_name: str,
    config: Dict = None
) -> Tuple[cobra.Model, List[str], Dict[str, Tuple[float, float]]]:
    """
    Apply TF knockout effects based on relative enzyme capacity limits (advanced version).

    This version uses kcat_MW information to estimate each reaction's relative capacity and scales bounds accordingly.

    Key idea:
    - Run wild-type FBA to obtain a reference flux distribution.
    - Restrict affected reaction bounds proportional to effect_strength.

    Args:
    model: COBRA model
    dictionary_model: model JSON dictionary
    tf_name: transcription factor name
    config: organism configuration

    Returns:
    model: modified model
    affected_reactions: affected reaction list
    bounds_changes: dict recording bound changes
    """
    get_regulatory_effects = config.get('get_regulatory_effects') if config else get_regulatory_effects_ecoli
    rules = get_regulatory_effects(tf_name)
    affected_reactions = []
    bounds_changes = {}

    if not rules:
        print(f"    No regulatory rules found for transcription factor {tf_name}")
        return model, affected_reactions, bounds_changes

    print(f"    Applying regulatory effects for TF {tf_name} (relative bounds mode)...")

    # Build a mapping from reaction ID to kcat_MW
    rxn_to_kcat_mw = {}
    for eachr in dictionary_model['reactions']:
        kcat_mw = eachr.get('kcat_MW', '')
        if kcat_mw and kcat_mw != '':
            try:
                rxn_to_kcat_mw[eachr['id']] = float(kcat_mw)
            except (ValueError, TypeError):
                pass

    # Total enzyme constraint
    if 'enzyme_constraint' in dictionary_model:
        E_total = dictionary_model['enzyme_constraint']['upperbound']
    else:
        E_total = 0.227

    for rule in rules:
        if rule.regulation_type == "activation" and rule.effect_strength < 1.0:
            for base_rxn_id in rule.target_reactions:
                matching_rxns = []
                for rxn in model.reactions:
                    rxn_base = rxn.id.replace('_reverse', '').split('_num')[0]
                    if rxn_base == base_rxn_id or rxn.id == base_rxn_id:
                        matching_rxns.append(rxn)

                for rxn in matching_rxns:
                    old_lb, old_ub = rxn.bounds

                    if rxn.id in rxn_to_kcat_mw:
                        kcat_mw = rxn_to_kcat_mw[rxn.id]
                        theoretical_max = kcat_mw * E_total * rule.effect_strength

                        if old_ub > 0:
                            new_ub = min(old_ub, theoretical_max)
                            new_ub = max(0, new_ub)
                            rxn.bounds = (old_lb, new_ub)
                            bounds_changes[rxn.id] = (old_ub, new_ub)

                            print(f"      {rxn.id}: upper_bound {old_ub:.2f} -> {new_ub:.2f} "
                                  f"(kcat_MW={kcat_mw:.2f}, effect: {rule.effect_strength:.0%})")
                            affected_reactions.append(rxn.id)
                    else:
                        # Without kcat_MW use proportional scaling, avoid infinite bounds
                        if old_ub > 0 and old_ub < 1000:
                            new_ub = old_ub * rule.effect_strength
                            rxn.bounds = (old_lb, new_ub)
                            bounds_changes[rxn.id] = (old_ub, new_ub)

                            print(f"      {rxn.id}: upper_bound {old_ub:.2f} -> {new_ub:.2f} "
                                  f"(no kcat_MW, proportional scaling)")
                            affected_reactions.append(rxn.id)

        elif rule.regulation_type == "repression":
            for base_rxn_id in rule.target_reactions:
                print(f"      {base_rxn_id}: de-repressed (negative-regulation target; bounds unchanged in current implementation)")
                affected_reactions.append(base_rxn_id)

    return model, affected_reactions, bounds_changes


def apply_enzyme_knockout(model: cobra.Model, enzyme_name: str, config: Dict = None) -> Tuple[cobra.Model, List[str]]:
    """
    Apply a metabolic enzyme knockout.

    Args:
    model: COBRA model
    enzyme_name: enzyme name
    config: organism configuration dictionary

    Returns:
    model: modified model
    knocked_out_reactions: list of reactions knocked out
    """
    get_enzyme_knockout_reactions = config.get('get_enzyme_knockout_reactions') if config else get_enzyme_knockout_reactions_ecoli
    reactions_to_ko = get_enzyme_knockout_reactions(enzyme_name)
    knocked_out_reactions = []

    if not reactions_to_ko:
        # If the enzyme is not in the predefined list, try to locate it directly in the model
        print(f"    {enzyme_name} not found in predefined list, searching model directly...")
        for rxn in model.reactions:
            rxn_base = rxn.id.replace('_reverse', '').split('_num')[0]
            if rxn_base.lower() == enzyme_name.lower():
                reactions_to_ko.append(rxn.id)

    for base_rxn_id in reactions_to_ko:
        # Knock out every matching reaction
        for rxn in model.reactions:
            rxn_base = rxn.id.replace('_reverse', '').split('_num')[0]
            if rxn_base == base_rxn_id or rxn.id == base_rxn_id:
                rxn.bounds = (0, 0)
                knocked_out_reactions.append(rxn.id)
                print(f"      Disabled reaction: {rxn.id}")

    return model, knocked_out_reactions


def get_enzyme_constraint_model(json_model_file: str) -> cobra.Model:
    """
    Create an ecGEM model with enzyme constraints (kcat_MW untouched).

    Args:
    json_model_file: ecGEM model JSON file

    Returns:
    model: COBRA model with enzyme constraints
    """
    dictionary_model = json_load(json_model_file)
    model = cobra.io.json.load_json_model(json_model_file)

    # Build the enzyme constraint
    coefficients = dict()
    for rxn in model.reactions:
        for eachr in dictionary_model['reactions']:
            if rxn.id == eachr['id']:
                if eachr.get('kcat_MW') and eachr['kcat_MW'] != '':
                    try:
                        coefficients[rxn.forward_variable] = 1 / float(eachr['kcat_MW'])
                    except (ValueError, TypeError):
                        pass
                break

    # Retrieve the enzyme constraint bounds
    if 'enzyme_constraint' in dictionary_model:
        lowerbound = dictionary_model['enzyme_constraint']['lowerbound']
        upperbound = dictionary_model['enzyme_constraint']['upperbound']
    else:
        lowerbound = 0
        upperbound = 0.227  # default

    # Add enzyme constraint to the model
    constraint = model.problem.Constraint(0, lb=lowerbound, ub=upperbound)
    model.add_cons_vars(constraint)
    model.solver.update()
    constraint.set_linear_coefficients(coefficients=coefficients)

    return model, dictionary_model


def generate_ecgem_flux_for_strain(
    model_file: str,
    strain_id: str,
    glucose_uptake: float,
    acetate_secretion: float,
    biomass: float,
    flux_ratio_constraint_file: str,
    flux_ratios_row: pd.Series,
    config: Dict = None,
    regulation_method: str = "bounds"  # "bounds" or "relative_bounds"
) -> Optional[Dict[str, float]]:
    """
    Generate ecGEM flux data for one strain (V2 - applying bounds-based regulation).

    Args:
    model_file: ecGEM model file
    strain_id: strain ID (TF or enzyme name)
    glucose_uptake: glucose uptake rate (mmol/g/h)
    acetate_secretion: acetate secretion rate (mmol/g/h)
    biomass: growth rate (h-1)
    flux_ratio_constraint_file: flux ratio constraint strategy file
    flux_ratios_row: row with flux ratio values for the strain
    config: organism configuration dictionary
    regulation_method: regulation method ("bounds" or "relative_bounds")

    Returns:
    flux_dict: flux dictionary
    """
    # Get organism name and wild-type identifier
    organism = config.get('name', 'E. coli') if config else 'E. coli'
    wild_type_id = config.get('wild_type_id', 'WILD TYPE') if config else 'WILD TYPE'
    biomass_reaction = config.get('biomass_reaction', 'BIOMASS_Ec_iJO1366_core_53p95M') if config else 'BIOMASS_Ec_iJO1366_core_53p95M'

    # Retrieve organism-specific helper functions
    is_transcription_factor = config.get('is_transcription_factor', is_transcription_factor_ecoli) if config else is_transcription_factor_ecoli
    is_metabolic_enzyme = config.get('is_metabolic_enzyme', is_metabolic_enzyme_ecoli) if config else is_metabolic_enzyme_ecoli

    print(f"\n{'='*80}")
    print(f"Processing strain: {strain_id}")
    print(f"  Glucose uptake: {glucose_uptake:.2f} mmol/g/h")
    print(f"  Acetate secretion: {acetate_secretion:.2f} mmol/g/h")
    print(f"  Growth rate: {biomass:.3f} h-1")
    print(f"  Regulation method: {regulation_method}")

    # Create the temporary flux ratio constraint file
    temp_constraint_file = f"./temp_flux_ratio_{strain_id.replace(' ', '_')}.csv"
    create_flux_ratio_constraints_temp_file(
        flux_ratio_constraint_file,
        flux_ratios_row,
        temp_constraint_file
    )

    try:
        # Determine strain type
        is_wild_type = strain_id == wild_type_id
        is_tf = is_transcription_factor(strain_id)
        is_enzyme = is_metabolic_enzyme(strain_id)

        # Load enzyme-constrained model
        model, dictionary_model = get_enzyme_constraint_model(model_file)

        if is_wild_type:
            print(f"  Type: wild type")
            # Wild type needs no extra handling
        elif is_tf:
            print(f"  Type: transcription factor mutant")
            if regulation_method == "relative_bounds":
                model, affected_rxns, bounds_changes = apply_tf_regulatory_effects_by_relative_bounds(
                    model, dictionary_model, strain_id, config
                )
            else:  # default: "bounds"
                model, affected_rxns, bounds_changes = apply_tf_regulatory_effects_by_bounds(
                    model, strain_id, glucose_uptake, config
                )
            print(f"    Adjusted bounds for {len(affected_rxns)} reactions")
        elif is_enzyme:
            print(f"  Type: metabolic enzyme mutant (control)")
            model, knocked_out_rxns = apply_enzyme_knockout(model, strain_id, config)
            print(f"    Disabled {len(knocked_out_rxns)} reactions")
        else:
            # Unknown type, treat as generic mutant
            print(f"  Type: unknown (handling as generic mutant)")
            print(f"    Warning: {strain_id} not defined in regulatory rules")

        # Add flux ratio constraints
        print(f"  Applying flux ratio constraints...")
        model, constraints_added = add_flux_ratio_constraints(model, temp_constraint_file)
        print(f"    Added {constraints_added} flux ratio constraints")

        # Apply in vitro constraints
        try:
            model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)
            model.reactions.get_by_id("EX_glc__D_e").bounds = (0, 0)
        except KeyError:
            # Try alternative reaction IDs
            if "EX_glc__D_e" in model.reactions:
                model.reactions.get_by_id("EX_glc__D_e").bounds = (-glucose_uptake, -glucose_uptake)

        try:
            model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
            model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
        except KeyError:
            if "EX_ac_e" in model.reactions:
                model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)

        # Block alternative carbon sources
        for rxn_id in ["EX_glyc_e", "EX_fru_e", "EX_glyc_e_reverse", "EX_fru_e_reverse"]:
            if rxn_id in model.reactions:
                model.reactions.get_by_id(rxn_id).bounds = (0, 0)

        # Fix Biomass reaction bounds
        model.reactions.get_by_id(biomass_reaction).bounds = (biomass, biomass)

        # Run FBA
        print(f"  Running FBA...")
        solution = model.optimize()

        if solution.status != "optimal":
            print(f"  ✗ Optimization failed, status={solution.status}")
            return None

        print(f"  ✓ Optimization successful, objective={solution.objective_value:.4f}")

        flux_dict = solution.fluxes.to_dict()
        return flux_dict

    except Exception as e:
        print(f"  ✗ Error encountered: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        if os.path.exists(temp_constraint_file):
            os.remove(temp_constraint_file)


def main():
    """Main entry point"""
    """
    1. Bounds method (default):
    - Use glucose uptake rate as a reference
    - new_upper_bound = reference_flux x effect_strength
    2. Relative_bounds method:
        - Use kcat_MW information to compute theoretical max flux
        - theoretical_max = kcat_MW x E_total x effect_strength
    """
    # ========== Configuration ==========
    # Supported organisms: "E. coli", "B. subtilis"
    ORGANISM = "E. coli"  # options: "E. coli", "B. subtilis"
    REGULATION_METHOD = "bounds"  # "bounds" or "relative_bounds"
    # ===================================

    # Validate organism selection
    if ORGANISM not in ORGANISM_CONFIGS:
        print(f"Error: unsupported organism '{ORGANISM}'")
        print(f"Supported organisms: {list(ORGANISM_CONFIGS.keys())}")
        return

    config = ORGANISM_CONFIGS[ORGANISM]

    print("=" * 80)
    print("Generate ecGEM-based 13C flux data (V2 - bounds-limited regulation)")
    print(f"Organism: {config['name']}")
    print(f"Regulation method: {REGULATION_METHOD}")
    print("Improvements:")
    print("  1. TF knockouts modeled via reaction bound restrictions (upper/lower)")
    print("  2. kcat_MW remains unchanged (intrinsic kinetic parameter)")
    print("  3. Activation: TF knockout -> lower upper bound")
    print("  4. Repression: TF knockout -> bounds unchanged (de-repressed)")
    print("=" * 80)

    # Pull file paths from config
    vitro_exp_file = config['vitro_exp_file']
    flux_data_file = config['flux_data_file']
    flux_ratio_constraint_file = config['flux_ratio_constraint_file']
    model_file = config['model_file']
    output_file = config['output_file']
    flux_data_sep = config.get('flux_data_sep', '\t')
    wild_type_id = config['wild_type_id']

    # Organism-specific helpers
    is_transcription_factor = config['is_transcription_factor']
    is_metabolic_enzyme = config['is_metabolic_enzyme']
    regulatory_rules = config['regulatory_rules']
    enzyme_knockouts = config['enzyme_knockouts']

    # Read source data
    print("\n[Step 1] Read data files...")
    vitro_exp_df = pd.read_csv(vitro_exp_file)
    flux_data_df = pd.read_csv(flux_data_file, sep=flux_data_sep)

    print(f"  In vitro data: {len(vitro_exp_df)} strains")
    print(f"  Flux ratio data: {len(flux_data_df)} strains")

    # Merge tables
    merged_df = pd.merge(vitro_exp_df, flux_data_df, on='Id', how='inner')
    print(f"  After merge: {len(merged_df)} valid strains")

    if len(merged_df) == 0:
        print("\nError: no data after merge. Check the 'Id' column in both files.")
        return

    # Regulatory rule summary
    print("\n[Step 1.5] Summary of TF regulatory rules:")
    print(f"  Defined {len(regulatory_rules)} transcription factor rules")
    print(f"  Defined {len(enzyme_knockouts)} metabolic enzyme knockout rules")

    # Count mutant types
    tf_count = 0
    enzyme_count = 0
    unknown_count = 0
    for strain_id in merged_df['Id']:
        if strain_id == wild_type_id:
            continue
        elif is_transcription_factor(strain_id):
            tf_count += 1
        elif is_metabolic_enzyme(strain_id):
            enzyme_count += 1
        else:
            unknown_count += 1
            print(f"    Warning: unrecognized mutant {strain_id}")

    print(f"  TF mutants: {tf_count}")
    print(f"  Enzyme mutants: {enzyme_count}")
    if unknown_count > 0:
        print(f"  Unrecognized: {unknown_count}")

    # Collect flux data per strain
    all_flux_data = []

    # Process each strain
    print(f"\n[Step 2] Generate flux data for each strain...")
    success_count = 0
    fail_count = 0

    for idx, row in merged_df.iterrows():
        strain_id = row['Id']
        glucose_uptake = row['Specific glucose uptake rate(mmol/g/h)']
        acetate_secretion = row['Specific acetate secretion rate(mmol/g/h)']
        biomass = row['Specific growth rate(h-1)']

        vitro_cols = ['Id', 'Specific glucose uptake rate(mmol/g/h)',
                      'Specific acetate secretion rate(mmol/g/h)',
                      'Specific growth rate(h-1)']
        flux_ratio_cols = [col for col in merged_df.columns if col not in vitro_cols]
        flux_ratios = row[flux_ratio_cols]

        flux_dict = generate_ecgem_flux_for_strain(
            model_file,
            strain_id,
            glucose_uptake,
            acetate_secretion,
            biomass,
            flux_ratio_constraint_file,
            flux_ratios,
            config=config,
            regulation_method=REGULATION_METHOD
        )

        if flux_dict is None:
            print(f"  ✗ Skipped strain {strain_id} (optimization failed)\n")
            fail_count += 1
            continue

        flux_dict['Id'] = strain_id
        all_flux_data.append(flux_dict)

        success_count += 1
        print(f"  ✓ Completed strain {strain_id}, generated {len(flux_dict)-1} reaction fluxes\n")

    # Build DataFrame
    print(f"\n[Step 3] Aggregate data...")
    print(f"  Success: {success_count} strains")
    print(f"  Failure: {fail_count} strains")

    if not all_flux_data:
        print("\nError: no flux data generated!")
        return

    flux_df = pd.DataFrame(all_flux_data)

    cols = ['Id'] + [col for col in flux_df.columns if col != 'Id']
    flux_df = flux_df[cols]

    flux_df = flux_df.fillna(0.0)
    flux_df = flux_df.sort_values('Id').reset_index(drop=True)

    # Save output
    print(f"\n[Step 4] Save results...")
    print(f"  Output file: {output_file}")
    flux_df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"✓ Finished!")
    print(f"  - Generated flux data for {len(flux_df)} strains")
    print(f"  - Contains {len(flux_df.columns)-1} reactions")
    print(f"  - Saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

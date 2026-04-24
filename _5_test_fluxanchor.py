import pandas as pd
import os
import cobra
import numpy as np
import math
import json
import sys
import logging
from typing import Dict, List, Optional, Tuple
from analysis.analysis_13C import CENTRAL_METABOLISM_60
from analysis.analysis_13C import EXTEND_REACTIONS, aggregate_ground_truth_flux
from analysis.analysis_13C import project_flux_to_bigg, project_flux_to_bigg_gems, filter_reactions_by_avg_flux_threshold
# Configure logging
logging.getLogger('cobra').setLevel(logging.ERROR)

# Extend sys.path for local modules
sys.path.append('./script/')
sys.path.append('./analysis/')

from sklearn.metrics import mean_squared_error, r2_score
from script.ECMpy_function import get_enzyme_constraint_model

# Import regulatory rule module (V2) for E. coli
from tf_regulatory_rules_v2 import (
    get_regulatory_effects as get_regulatory_effects_ecoli,
    get_enzyme_knockout_reactions as get_enzyme_knockout_reactions_ecoli,
    is_transcription_factor as is_transcription_factor_ecoli,
    is_metabolic_enzyme as is_metabolic_enzyme_ecoli,
    get_affected_reactions_and_effects,
    REGULATORY_RULES_V2 as REGULATORY_RULES_ECOLI,
    ENZYME_KNOCKOUTS as ENZYME_KNOCKOUTS_ECOLI,
    RegulatoryRuleV2
)

# Import regulatory rule module for Bacillus subtilis
# from tf_regulatory_rules_bsubtilis import (
#     get_regulatory_effects_bsubtilis,
#     get_enzyme_knockout_reactions_bsubtilis,
#     is_transcription_factor_bsubtilis,
#     is_metabolic_enzyme_bsubtilis,
#     REGULATORY_RULES_BSUBTILIS,
#     ENZYME_KNOCKOUTS_BSUBTILIS,
# )


class MultiOrganismKnockoutAnalysis:
    """Multi-organism gene knockout analysis framework"""

    def __init__(self, evaluation_mode="central_60", flux_threshold=0.01,
                 train_data_ratio="wildtype", random_seed=42, model_suffix="",
                 customize_samples=None):
        """
        Initialize the analysis framework.

        Args:
        evaluation_mode: evaluation strategy
            - "central_60": fixed set of 60 central carbon reactions (default, recommended)
            - "extend_reactions": extended reaction set (pathway annotations + reverse reactions)
            - "threshold": reactions with ground truth > flux_threshold
            - "all_reactions": every reaction
        flux_threshold: flux cutoff (mmol/g/h), used only when evaluation_mode="threshold"
        train_data_ratio: training data split
            - "wildtype": train on wild type only, evaluate all mutants
            - "customize": use a custom sample list (requires customize_samples)
            - 0.0-1.0: fraction of mutants assigned to the test set
        random_seed: random seed matching the training pipeline
        model_suffix: model filename suffix derived from train_data_ratio
        customize_samples: user-defined sample list (only for train_data_ratio="customize")
        """
        self.evaluation_mode = evaluation_mode
        self.flux_threshold = flux_threshold
        self.train_data_ratio = train_data_ratio
        self.random_seed = random_seed
        self.model_suffix = model_suffix
        self.customize_samples = customize_samples if customize_samples else []

        self.organisms_config = {
            "iECDH1ME8569_1439": {
                "name": "E.coli",
                "glucose_uptake": 10,
                "wild_type_id": "WILD TYPE",
                "biomass_reaction": "BIOMASS_Ec_iJO1366_core_53p95M",
                "vitro_exp": "./predict/iECDH1ME8569_1439/vitro_exp_data.csv",
                "test_file": "./predict/iECDH1ME8569_1439/test_13C.csv",
                "flux_data_file": "./predict/iECDH1ME8569_1439/13C_data.csv",
                "gene_mapping_file": "./predict/iECDH1ME8569_1439/goal_blast.blast",
                "analysis_dir": "./predict/iECDH1ME8569_1439/analysis/",
                "network_file": "./predict/iECDH1ME8569_1439/network.xml",
                "use_reverse_reactions": True,
                # Regulatory function configuration
                "get_regulatory_effects": get_regulatory_effects_ecoli,
                "get_enzyme_knockout_reactions": get_enzyme_knockout_reactions_ecoli,
                "is_transcription_factor": is_transcription_factor_ecoli,
                "is_metabolic_enzyme": is_metabolic_enzyme_ecoli,
                "regulatory_rules": REGULATORY_RULES_ECOLI,
                "enzyme_knockouts": ENZYME_KNOCKOUTS_ECOLI,
            },
            "Bacillus subtilis": {
                "name": "Bacillus subtilis",
                "glucose_uptake": 10,
                "wild_type_id": "Wild-type",
                "biomass_reaction": "BIOMASS_BS_10",
                "vitro_exp": "./predict/Bacillus subtilis/vitro_exp_data.csv",
                "test_file": "./predict/Bacillus subtilis/test_13C.csv",
                "flux_data_file": "./predict/Bacillus subtilis/13C_data.csv",
                "gene_mapping_file": None,  # no mapping file available for B. subtilis
                "analysis_dir": "./predict/Bacillus subtilis/analysis/",
                "network_file": "./predict/Bacillus subtilis/iYO844.xml",
                "use_reverse_reactions": True,
                # Regulatory function configuration
                "get_regulatory_effects": get_regulatory_effects_bsubtilis,
                "get_enzyme_knockout_reactions": get_enzyme_knockout_reactions_bsubtilis,
                "is_transcription_factor": is_transcription_factor_bsubtilis,
                "is_metabolic_enzyme": is_metabolic_enzyme_bsubtilis,
                "regulatory_rules": REGULATORY_RULES_BSUBTILIS,
                "enzyme_knockouts": ENZYME_KNOCKOUTS_BSUBTILIS,
            }
        }

        self.methods = [
            "FluxGen",
            "KinLLM",
            "fba",
            # "TurNup",
            # "Catpred",
            # "AutoPACMEN",
            # "UniKP",
            # "DLKcat"
        ]

    def load_gene_mapping(self, mapping_file: str) -> Dict[str, str]:
        """
        Load gene mapping data (kept for backward compatibility).
        Regulatory rules are the primary mechanism now, but mapping acts as a fallback.
        """
        gene_mapping = {}

        if not mapping_file or not os.path.exists(mapping_file):
            print(f"  Gene mapping file not found: {mapping_file}")
            return gene_mapping

        try:
            with open(mapping_file, 'r') as f:
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
            print(f"  Loaded {len(gene_mapping)} gene mappings")
        except Exception as e:
            print(f"  Warning: unable to read gene mapping file {mapping_file}: {str(e)}")

        return gene_mapping

    def apply_tf_regulatory_knockout(
        self,
        model: cobra.Model,
        tf_name: str,
        glucose_uptake: float = 10.0,
        organism: str = "iECDH1ME8569_1439"
    ) -> Tuple[cobra.Model, List[str], Dict[str, Tuple[float, float]]]:
        """
        Apply TF knockout regulatory effects by constraining reaction bounds.

        Args:
        model: COBRA model
        tf_name: transcription factor name
        glucose_uptake: glucose uptake rate used to estimate reference flux
        organism: organism identifier

        Returns:
        model: modified model
        affected_reactions: list of affected reactions
        bounds_changes: dict recording bound changes {rxn_id: (old_ub, new_ub)}
        """
        # Retrieve organism-specific regulatory rules
        config = self.organisms_config.get(organism, self.organisms_config["iECDH1ME8569_1439"])
        get_regulatory_effects = config.get("get_regulatory_effects", get_regulatory_effects_ecoli)

        rules = get_regulatory_effects(tf_name)
        affected_reactions = []
        bounds_changes = {}

        if not rules:
            print(f"      No regulatory rules defined for TF {tf_name}")
            return model, affected_reactions, bounds_changes

        # Estimate reference flux from glucose uptake
        reference_flux = glucose_uptake * 2

        for rule in rules:
            if rule.regulation_type == "activation" and rule.effect_strength < 1.0:
                # Activation: TF knockout lowers enzyme expression -> constrain upper bound
                for base_rxn_id in rule.target_reactions:
                    # Locate all matching reactions (including isoenzymes)
                    matching_rxns = []
                    for rxn in model.reactions:
                        rxn_base = rxn.id.replace('_reverse', '').split('_num')[0]
                        if rxn_base == base_rxn_id or rxn.id == base_rxn_id:
                            matching_rxns.append(rxn)

                    for rxn in matching_rxns:
                        old_lb, old_ub = rxn.bounds

                        if old_ub > 0:
                            # New UB = min(original UB, reference_flux * effect_strength)
                            new_ub = min(old_ub, reference_flux * rule.effect_strength)
                            new_ub = max(0, new_ub)
                            rxn.bounds = (old_lb, new_ub)
                            bounds_changes[rxn.id] = (old_ub, new_ub)
                            affected_reactions.append(rxn.id)

                        elif old_lb < 0:
                            # Handle reverse direction of reversible reactions
                            new_lb = max(old_lb, -reference_flux * rule.effect_strength)
                            rxn.bounds = (new_lb, old_ub)
                            bounds_changes[rxn.id] = (old_lb, new_lb)
                            affected_reactions.append(rxn.id)

            elif rule.regulation_type == "repression":
                # Repression: knockout de-represses, bounds stay unchanged
                for base_rxn_id in rule.target_reactions:
                    affected_reactions.append(base_rxn_id)

        return model, affected_reactions, bounds_changes

    def apply_enzyme_knockout(
        self,
        model: cobra.Model,
        enzyme_name: str,
        organism: str = "iECDH1ME8569_1439"
    ) -> Tuple[cobra.Model, List[str]]:
        """
        Apply a metabolic-enzyme knockout by disabling reactions directly.

        Args:
        model: COBRA model
        enzyme_name: enzyme identifier
        organism: organism identifier

        Returns:
        model: modified model
        knocked_out_reactions: reactions disabled by the knockout
        """
        # Fetch organism-specific knockout function
        config = self.organisms_config.get(organism, self.organisms_config["iECDH1ME8569_1439"])
        get_enzyme_knockout_reactions = config.get("get_enzyme_knockout_reactions", get_enzyme_knockout_reactions_ecoli)

        reactions_to_ko = get_enzyme_knockout_reactions(enzyme_name)
        knocked_out_reactions = []

        if not reactions_to_ko:
            # Fall back to scanning the model when not pre-defined
            for rxn in model.reactions:
                rxn_base = rxn.id.replace('_reverse', '').split('_num')[0]
                if rxn_base.lower() == enzyme_name.lower():
                    reactions_to_ko.append(rxn.id)

        for base_rxn_id in reactions_to_ko:
            # Disable each matching reaction
            for rxn in model.reactions:
                rxn_base = rxn.id.replace('_reverse', '').split('_num')[0]
                if rxn_base == base_rxn_id or rxn.id == base_rxn_id:
                    rxn.bounds = (0, 0)
                    knocked_out_reactions.append(rxn.id)

        return model, knocked_out_reactions

    def apply_gene_knockout_by_regulation(
        self,
        model: cobra.Model,
        gene_name: str,
        glucose_uptake: float = 10.0,
        organism: str = "iECDH1ME8569_1439"
    ) -> Tuple[str, List[str]]:
        """
        Apply knockout logic according to regulatory rules.

        Args:
        model: COBRA model
        gene_name: gene identifier
        glucose_uptake: glucose uptake rate
        organism: organism identifier

        Returns:
        knockout_type: "transcription_factor", "metabolic_enzyme", or "unknown"
        affected_reactions: list of affected reactions
        """
        # Grab organism-specific helper functions
        config = self.organisms_config.get(organism, self.organisms_config["iECDH1ME8569_1439"])
        is_transcription_factor = config.get("is_transcription_factor", is_transcription_factor_ecoli)
        is_metabolic_enzyme = config.get("is_metabolic_enzyme", is_metabolic_enzyme_ecoli)
        
        # Determine gene type
        if is_transcription_factor(gene_name):
            # TF knockout via bound constraints
            model, affected_rxns, bounds_changes = self.apply_tf_regulatory_knockout(
                model, gene_name, glucose_uptake, organism
            )
            return "transcription_factor", affected_rxns

        elif is_metabolic_enzyme(gene_name):
            # Enzyme knockout by disabling reactions
            model, knocked_out_rxns = self.apply_enzyme_knockout(model, gene_name, organism)
            return "metabolic_enzyme", knocked_out_rxns

        else:
            # Unknown type: return empty list (mapping fallback used later)
            return "unknown", []

    def load_model(self, organism: str, method: str) -> Optional[cobra.Model]:
        """Load the model for a given organism and method."""
        config = self.organisms_config[organism]

        try:
            if method == "fba":
                # model_file = config["network_file"]
                # model = cobra.io.read_sbml_model(model_file)
                # model_file = f"{config['analysis_dir']}/get_kcat_mw_by_KinLLM/ecGEM/Bayesian/best_ecGEM_overflow.json"
                model_file = f"{config['analysis_dir']}/get_kcat_mw_by_KinLLM/ecGEM/ecGEM_irr_enz_constraint.json"
                model = cobra.io.json.load_json_model(model_file)
            elif method == "FluxGen":
                # model_file = f"{config['analysis_dir']}/get_kcat_mw_by_KinLLM/ecGEM/Bayesian/best_ecGEM_overflow.json"
                model_file = f"{config['analysis_dir']}/get_kcat_mw_by_KinLLM/ecGEM/Bayesian/best_ecGEM{self.model_suffix}.json"
                # model_file = f"{config['analysis_dir']}/get_kcat_mw_by_KinLLM/ecGEM/EGA/best_ecGEM{self.model_suffix}.json"
                # model_file = f"{config['analysis_dir']}/get_kcat_mw_by_KinLLM/ecGEM/Bayesian/best_ecGEM_13C_biomass_ace.json"
                model = get_enzyme_constraint_model(model_file)
            else:
                model_file = f"{config['analysis_dir']}/get_kcat_mw_by_{method}/ecGEM/ecGEM_irr_enz_constraint.json"
                model = get_enzyme_constraint_model(model_file)

            # print(f"  Loaded model: {model_file}")
            return model

        except Exception as e:
            print(f"  Error: unable to load model for method {method}: {str(e)}")
            return None

    def set_model_constraints(self, model: cobra.Model, organism: str, method: str):
        """Apply model constraints for the given organism/method."""
        config = self.organisms_config[organism]
        glucose_uptake = config["glucose_uptake"]
        # vitro_exp_df = config["vitro_exp"]
        use_reverse = config["use_reverse_reactions"]

        try:
            # if method == "fba":
            #     # FBA uses negative bounds for uptake
            #     model.reactions.get_by_id("EX_glc__D_e").bounds = (-glucose_uptake, 0)
            #     model.reactions.get_by_id("EX_ac_e").bounds = (0, 0)
            #     if organism == "iECDH1ME8569_1439":
            #         model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
            #         model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
            # else:
            # Other methods rely on reverse reactions
            if use_reverse:
                model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)
                model.reactions.get_by_id("EX_ac_e").bounds = (0, 100)
                model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
                if organism == "iECDH1ME8569_1439":
                    model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
                    model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)
                    # model.reactions.get_by_id("XYLI2").bounds = (0, 0)
                    # model.reactions.get_by_id("XYLI2_reverse").bounds = (0, 0)
                    # model.reactions.get_by_id("HEX1").bounds = (0, 0)
                elif organism == "Bacillus subtilis":
                    model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
                    model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)

        except Exception as e:
            print(f"  Warning: failed to configure reaction bounds: {str(e)}")

    def find_genes_to_knockout(self, model: cobra.Model, gene_name: str,
                              gene_mapping: Dict[str, str], organism: str) -> List:
        """Locate genes that should be knocked out."""
        genes_to_knockout = []

        # Handle multi-gene knockouts (e.g., ytsJ_gapB)
        if '_' in gene_name and organism == "Bacillus subtilis":
            gene_names = gene_name.split('_')
        else:
            gene_names = [gene_name]

        for single_gene in gene_names:
            # Resolve mapped gene ID (if any)
            target_gene_id = None
            gene_name_lower = single_gene.lower()

            if gene_name_lower in gene_mapping:
                target_gene_id = gene_mapping[gene_name_lower]
            else:
                target_gene_id = single_gene

            # Search for gene entry in the model
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

    def analyze_single_knockout(self, model: cobra.Model, gene_name: str,
                               true_values: List[float], bigg_cols: List[str],
                               gene_mapping: Dict[str, str], organism: str, method: str,
                               active_reactions: List[str], glucose_uptake: float = 10.0) -> Optional[Dict]:
        """
        Analyze a single gene knockout using regulatory rules.

        Improvements:
        1. TF knockouts restrict bounds only for positive-regulation targets whose capacity drops after TF loss.
        2. Enzyme knockouts disable reactions directly.
        3. Unknown genes fall back to mapping-based knockouts.
        """
        try:
            # Apply regulatory-rule knockout effects
            knockout_type, affected_reactions = self.apply_gene_knockout_by_regulation(
                model, gene_name, glucose_uptake, organism
            )

            if knockout_type == "transcription_factor":
                # Positive-regulation targets can be down-constrained; negative-regulation targets are only reported as de-repressed.
                if affected_reactions:
                    pass  # effect already applied
                else:
                    # Possibly no target reactions in the core network
                    pass

            elif knockout_type == "metabolic_enzyme":
                # Enzyme knockout path
                if not affected_reactions:
                    return None

            else:
                # Unknown type -> fallback to mapping
                genes_to_knockout = self.find_genes_to_knockout(model, gene_name, gene_mapping, organism)

                if not genes_to_knockout:
                    return None

                # Perform knockout
                for gene in genes_to_knockout:
                    gene.knock_out()

            # Solve FBA
            solution = model.optimize()

            if solution.status != "optimal":
                return None

            # Map predicted fluxes to BiGG IDs
            pred_flux_dict = solution.fluxes.to_dict()
            pred_flux = project_flux_to_bigg_gems(pred_flux_dict, active_reactions)

            # Collect predictions for active reactions
            pred_values = []
            for bigg_id in bigg_cols:
                pred_val = pred_flux.get(bigg_id, 0.0)
                pred_values.append(abs(pred_val))

            # Compute metrics
            mse = mean_squared_error(true_values, pred_values)
            rmse = math.sqrt(mse)
            r2 = r2_score(true_values, pred_values)

            # Build per-reaction breakdown
            reaction_details = {}
            for i, bigg_id in enumerate(bigg_cols):
                reaction_details[f'{bigg_id}_true'] = true_values[i]
                reaction_details[f'{bigg_id}_pred'] = pred_values[i]
                reaction_details[f'{bigg_id}_abs_error'] = abs(true_values[i] - pred_values[i])
                if true_values[i] != 0:
                    reaction_details[f'{bigg_id}_rel_error'] = abs(true_values[i] - pred_values[i]) / abs(true_values[i])
                else:
                    reaction_details[f'{bigg_id}_rel_error'] = 0.0

            result = {
                'Gene': gene_name,
                'Organism': organism,
                'Method': method,
                'Knockout_Type': knockout_type,
                'Affected_Reactions': len(affected_reactions) if affected_reactions else 0,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'Status': 'Success'
            }

            # Append reaction-level metrics
            result.update(reaction_details)

            return result

        except Exception as e:
            return None

    def analyze_organism_method(self, organism: str, method: str) -> Dict:
        """Analyze a specific organism/method pair."""
        print(f"\nStarting analysis: {organism} - {method}")
        print("-" * 60)

        config = self.organisms_config[organism]

        # Select reactions according to evaluation mode
        try:
            flux_data_all = pd.read_csv(config["flux_data_file"])

            print("\n" + "="*80)

            if self.evaluation_mode == "central_60":
                active_reactions = CENTRAL_METABOLISM_60
                print("Evaluation mode: central carbon reactions (central_60)")
                print("Using the fixed set of 60 central carbon reactions")

            elif self.evaluation_mode == "extend_reactions":
                active_reactions = EXTEND_REACTIONS
                print("Evaluation mode: extended reaction set (extend_reactions)")
                print(f"Reactions: {len(active_reactions)}")
                print("Includes glycolysis, PPP, TCA, amino acid, fatty acid, nucleotide pathways, etc.")

            elif self.evaluation_mode == "threshold":
                # # Alternative threshold mode example: using only wild-type flux
                # wild_type_data = flux_data_all[flux_data_all['Id'] == 'WILD TYPE']
                # if wild_type_data.empty:
                #     print(f"  Error: WILD TYPE not found in {config['flux_data_file']}")
                #     return self._empty_result(organism, method)

                # reaction_cols = [col for col in flux_data_all.columns if col != 'Id']
                # gt_fluxes_all = wild_type_data[reaction_cols].iloc[0].abs()

                # # Select reactions whose true flux exceeds the threshold
                # active_reactions_list = gt_fluxes_all[gt_fluxes_all > self.flux_threshold].index.tolist()
                # active_reactions = active_reactions_list
                # print("Evaluation mode: threshold")
                # print(f"Flux threshold: {self.flux_threshold} mmol/g/h")
                # print(f"Active reactions: {len(active_reactions)}")
                # Official threshold mode: compute average flux across all samples
                active_reactions = filter_reactions_by_avg_flux_threshold(flux_data_all, self.flux_threshold)
                print("Evaluation mode: threshold")
                print(f"Flux threshold: {self.flux_threshold} mmol/g/h")
                print(f"Total samples (wild type + mutants): {len(flux_data_all)}")
                print(f"Active reactions: {len(active_reactions)}")

            elif self.evaluation_mode == "all_reactions":
                # Mode: evaluate all reactions
                reaction_cols = [col for col in flux_data_all.columns if col != 'Id']
                active_reactions = reaction_cols
                print("Evaluation mode: all reactions")
                print(f"Reactions: {len(active_reactions)}")
                print("Warning: many inactive reactions included; metrics may be inflated")

            else:
                raise ValueError(f"Unsupported evaluation mode: {self.evaluation_mode}. "
                                 "Please select 'central_60', 'extend_reactions', 'threshold', or 'all_reactions'")

            print("="*80)

        except Exception as e:
            print(f"  Error: unable to load flux dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._empty_result(organism, method)

        # Determine test samples from 13C_data.csv
        try:
            config = self.organisms_config[organism]
            wild_type_id = config["wild_type_id"]

            # ================================================================================
            # Data split logic (must mirror training exactly)
            # ================================================================================
            print(f"\n{'='*80}")
            print("Test data selection strategy")
            print(f"{'='*80}")

            if self.train_data_ratio == "customize":
                # Custom mode: evaluate specified samples only
                print("Mode: custom sample list")
                print(f"Wild type ID: {wild_type_id}")
                print(f"Custom samples: {self.customize_samples}")

                # Validate presence of each custom sample
                all_sample_ids = flux_data_all['Id'].tolist()
                missing_samples = [s for s in self.customize_samples if s not in all_sample_ids]
                if missing_samples:
                    print(f"  Error: custom samples not present in 13C_data.csv: {missing_samples}")
                    print(f"  Available samples: {all_sample_ids}")
                    return self._empty_result(organism, method)

                # Evaluate only custom samples
                mutant_strains = [s for s in self.customize_samples if s != wild_type_id]
                print(f"  Total samples evaluated: {len(self.customize_samples)}")
                print(f"  Details: {self.customize_samples}")
                print(f"  Mutants evaluated: {len(mutant_strains)}")

            else:
                # Wildtype or ratio: evaluate all mutants
                print("Mode: evaluate all mutants (complete analysis)")
                print(f"Wild type ID: {wild_type_id}")
                print("Note: every mutant is evaluated regardless of training split")

                mutant_strains = flux_data_all[flux_data_all['Id'] != wild_type_id]['Id'].tolist()
                print(f"  Loaded {len(mutant_strains)} mutant entries from 13C_data.csv")

                # In ratio mode, also display the train/test split used during training
                if self.train_data_ratio != "wildtype":
                    test_ratio = float(self.train_data_ratio)
                    print(f"\n  Training configuration reference: test share {test_ratio*100:.1f}% (mutants)")
                    print(f"  Random seed: {self.random_seed}")

                    # Separate wild-type from mutant IDs
                    all_sample_ids = flux_data_all['Id'].tolist()
                    if wild_type_id in all_sample_ids:
                        mutant_ids = [sid for sid in all_sample_ids if sid != wild_type_id]
                        n_mutants = len(mutant_ids)
                        n_test = int(n_mutants * test_ratio)

                        # Use the same random seed as training
                        np.random.seed(self.random_seed)
                        import random
                        random.seed(self.random_seed)
                        shuffled_indices = np.random.permutation(n_mutants)

                        test_indices = shuffled_indices[:n_test]
                        test_samples = [mutant_ids[i] for i in test_indices]
                        train_samples = [wild_type_id] + [mutant_ids[i] for i in shuffled_indices[n_test:]]

                        print(f"  (Reference) training set: 1 wild type + {n_mutants - n_test} mutants")
                        print(f"  (Reference) test set: {n_test} mutants")
                        print(f"  (Reference) test sample preview: {test_samples[:5]}{'...' if len(test_samples) > 5 else ''}")
                        print(f"  Actual evaluation covers all {len(mutant_strains)} mutants (train + test)")

            print(f"{'='*80}\n")

        except Exception as e:
            print(f"  Error: unable to extract samples from 13C_data.csv: {str(e)}")
            return self._empty_result(organism, method)

        # Load gene mapping as fallback
        gene_mapping = self.load_gene_mapping(config["gene_mapping_file"])

        # Load in vitro experimental data
        vitro_exp_df = pd.read_csv(config["vitro_exp"], sep=",")

        # Retrieve organism-specific regulatory helpers
        is_transcription_factor = config.get("is_transcription_factor", is_transcription_factor_ecoli)
        is_metabolic_enzyme = config.get("is_metabolic_enzyme", is_metabolic_enzyme_ecoli)
        regulatory_rules = config.get("regulatory_rules", REGULATORY_RULES_ECOLI)
        enzyme_knockouts = config.get("enzyme_knockouts", ENZYME_KNOCKOUTS_ECOLI)

        # Summarize mutant types
        print(f"\n{'='*80}")
        print("Mutant classification summary (based on regulatory rules)")
        print(f"{'='*80}")
        print(f"  Defined {len(regulatory_rules)} TF regulatory rules")
        print(f"  Defined {len(enzyme_knockouts)} enzyme knockout rules")
        tf_count = 0
        enzyme_count = 0
        unknown_count = 0
        for strain_id in mutant_strains:
            if is_transcription_factor(strain_id):
                tf_count += 1
            elif is_metabolic_enzyme(strain_id):
                enzyme_count += 1
            else:
                unknown_count += 1

        print(f"  TF knockouts: {tf_count} (positive-regulation targets may be down-constrained after TF loss)")
        print(f"  Enzyme knockouts: {enzyme_count} (disable reactions)")
        print(f"  Unknown category: {unknown_count} (fallback to mapping)")
        print(f"  Total mutants: {len(mutant_strains)}")
        print(f"{'='*80}\n")

        # Iterate through each gene knockout
        results = []
        failed_knockouts = []

        for idx, gene_name in enumerate(mutant_strains):
            # Reload a fresh model for each gene to avoid state bleed
            model = self.load_model(organism, method)
            if model is None:
                failed_knockouts.append(gene_name)
                continue

            # Apply organism-specific constraints
            self.set_model_constraints(model, organism, method)

            # Ensure we have in vitro data for this mutant
            gene_vitro_data = vitro_exp_df[vitro_exp_df["Id"] == gene_name]
            if gene_vitro_data.empty:
                print(f"  Warning: {gene_name} not found in vitro_exp_data.csv, skipping")
                failed_knockouts.append(gene_name)
                continue

            # Apply strain-specific constraints
            try:
                config = self.organisms_config[organism]
                biomass_reaction = config["biomass_reaction"]

                biomass = float(gene_vitro_data["Specific growth rate(h-1)"].iloc[0])
                glucose_uptake = float(gene_vitro_data["Specific glucose uptake rate(mmol/g/h)"].iloc[0])
                acetate_secretion = float(gene_vitro_data["Specific acetate secretion rate(mmol/g/h)"].iloc[0])
                model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (glucose_uptake, glucose_uptake)
                model.reactions.get_by_id("EX_glc__D_e").bounds = (0, 0)
                model.reactions.get_by_id("EX_ac_e").bounds = (acetate_secretion, acetate_secretion)
                model.reactions.get_by_id(biomass_reaction).bounds = (biomass, biomass)
            except Exception as e:
                print(f"  Warning: failed to set constraints for {gene_name}: {str(e)}; skipping")
                failed_knockouts.append(gene_name)
                continue

            if idx % 10 == 0:  # report progress every 10 genes
                print(f"  Progress: {idx+1}/{len(mutant_strains)} ({(idx+1)/len(mutant_strains)*100:.1f}%)")

            # Extract ground-truth fluxes for this gene (active reactions only)
            gene_flux_data = flux_data_all[flux_data_all['Id'] == gene_name]
            if gene_flux_data.empty:
                print(f"  Warning: flux data for {gene_name} not found in 13C_data.csv, skipping")
                failed_knockouts.append(gene_name)
                continue

            # Pull true flux values directly without additional aggregation
            # true_values = []
            agg_ground_truth = aggregate_ground_truth_flux(gene_flux_data, active_reactions)
            true_values = np.array([agg_ground_truth[rxn] for rxn in active_reactions])
            # for rxn in active_reactions:
            #     if rxn in gene_flux_data.columns:
            #         true_val = abs(float(gene_flux_data[rxn].iloc[0]))
            #     else:
            #         # Default to zero when reaction data are missing
            #         true_val = 0.0
            #     true_values.append(true_val)
            
            # Perform knockout analysis (regulatory rules)
            result = self.analyze_single_knockout(
                model, gene_name, true_values, active_reactions,
                gene_mapping, organism, method, active_reactions,
                glucose_uptake=glucose_uptake
            )
    
            if result:
                results.append(result)
            else:
                failed_knockouts.append(gene_name)

        # Summarize results
        if results:
            results_df = pd.DataFrame(results)
            avg_mse = results_df['MSE'].mean()
            avg_rmse = results_df['RMSE'].mean()
            avg_r2 = results_df['R2'].mean()

            # Count knockout categories
            knockout_type_counts = results_df['Knockout_Type'].value_counts().to_dict()

            summary = {
                'organism': organism,
                'method': method,
                'avg_mse': avg_mse,
                'avg_rmse': avg_rmse,
                'avg_r2': avg_r2,
                'success_count': len(results_df),
                'failed_count': len(failed_knockouts),
                'tf_knockout_count': knockout_type_counts.get('transcription_factor', 0),
                'enzyme_knockout_count': knockout_type_counts.get('metabolic_enzyme', 0),
                'unknown_knockout_count': knockout_type_counts.get('unknown', 0),
                'detailed_results': results_df
            }

            print("  Analysis complete:")
            print(f"    Successful knockouts: {len(results_df)}")
            print(f"      - TF knockouts: {knockout_type_counts.get('transcription_factor', 0)} (positive-regulation targets may be down-constrained)")
            print(f"      - Enzyme knockouts: {knockout_type_counts.get('metabolic_enzyme', 0)}")
            print(f"      - Unknown category: {knockout_type_counts.get('unknown', 0)}")
            print(f"    Failed knockouts: {len(failed_knockouts)}")
            print(f"    Average RMSE: {avg_rmse:.6f}")
            print(f"    Average MSE: {avg_mse:.6f}")
            print(f"    Average R^2: {avg_r2:.6f}")

        else:
            print("  No gene knockouts succeeded")
            summary = self._empty_result(organism, method)

        return summary

    def _empty_result(self, organism: str, method: str) -> Dict:
        """Return an empty result structure."""
        return {
            'organism': organism,
            'method': method,
            'avg_mse': np.nan,
            'avg_rmse': np.nan,
            'avg_r2': np.nan,
            'success_count': 0,
            'failed_count': 0,
            'detailed_results': pd.DataFrame()
        }

    def run_full_analysis(self, organisms: List[str] = None, methods: List[str] = None) -> Dict:
        """Execute the full multi-organism analysis."""
        if organisms is None:
            organisms = list(self.organisms_config.keys())
        if methods is None:
            methods = self.methods

        print("=" * 80)
        print("Starting multi-organism, multi-method knockout analysis")
        print(f"Organisms: {', '.join(organisms)}")
        print(f"Methods: {', '.join(methods)}")
        print("=" * 80)

        all_results = {}

        for organism in organisms:
            if organism not in self.organisms_config:
                print(f"Warning: unsupported organism {organism}")
                continue

            organism_results = {}

            for method in methods:
                result = self.analyze_organism_method(organism, method)
                organism_results[method] = result

            all_results[organism] = organism_results

        return all_results

    def save_results(self, all_results: Dict, output_base_dir: str = "./results"):
        """Save aggregated and detailed results."""
        # Choose output directory based on evaluation mode
        if self.evaluation_mode == "central_60":
            output_dir = output_base_dir  # default path
        elif self.evaluation_mode == "extend_reactions":
            output_dir = f"{output_base_dir}_extend_reactions"
        elif self.evaluation_mode == "threshold":
            output_dir = f"{output_base_dir}_threshold_{self.flux_threshold}"
        elif self.evaluation_mode == "all_reactions":
            output_dir = f"{output_base_dir}_all_reactions"
        else:
            output_dir = output_base_dir

        os.makedirs(output_dir, exist_ok=True)

        # Build summary rows
        summary_data = []
        all_detailed_results = []

        for organism, organism_results in all_results.items():
            for method, result in organism_results.items():
                # Aggregate metrics
                summary_data.append({
                    'Organism': organism,
                    'Method': method,
                    'Average_MSE': result['avg_mse'],
                    'Average_RMSE': result['avg_rmse'],
                    'Average_R2': result['avg_r2'],
                    'Success_Count': result['success_count'],
                    'Failed_Count': result['failed_count'],
                    'TF_Knockout_Count': result.get('tf_knockout_count', 0),
                    'Enzyme_Knockout_Count': result.get('enzyme_knockout_count', 0),
                    'Unknown_Knockout_Count': result.get('unknown_knockout_count', 0),
                    'Success_Rate': result['success_count'] / (result['success_count'] + result['failed_count']) * 100
                                   if (result['success_count'] + result['failed_count']) > 0 else 0
                })

                # Capture detailed records per method
                if not result['detailed_results'].empty:
                    all_detailed_results.append(result['detailed_results'])

        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{output_dir}/multi_organism_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        print(f"Evaluation mode: {self.evaluation_mode}")

        # Save concatenated detailed results
        if all_detailed_results:
            detailed_df = pd.concat(all_detailed_results, ignore_index=True)
            if self.train_data_ratio == "wildtype":
                detailed_path = f"{output_dir}/multi_organism_detailed.csv"
            else:
                detailed_path = f"{output_dir}/multi_organism_detailed_{self.train_data_ratio}.csv"
            detailed_df.to_csv(detailed_path, index=False)
            print(f"Detailed results saved to: {detailed_path}")

        # Save per-organism CSVs
        for organism, organism_results in all_results.items():
            organism_dir = f"{output_dir}/{organism.replace(' ', '_')}"
            os.makedirs(organism_dir, exist_ok=True)

            # Summary for this organism
            organism_summary = summary_df[summary_df['Organism'] == organism]
            if self.train_data_ratio == "wildtype":
                organism_summary.to_csv(f"{organism_dir}/summary.csv", index=False)
            else:
                organism_summary.to_csv(f"{organism_dir}/summary_{self.train_data_ratio}.csv", index=False)
            # Detailed reaction-level results for this organism
            organism_detailed = []
            for method, result in organism_results.items():
                if not result['detailed_results'].empty:
                    organism_detailed.append(result['detailed_results'])

                    # Save per-method breakdown for convenience
                    if self.train_data_ratio == "wildtype":
                        method_detailed_path = f"{organism_dir}/{method}_detailed_reactions.csv"
                    else:
                        method_detailed_path = f"{organism_dir}/{method}_detailed_reactions_{self.train_data_ratio}.csv"

                    result['detailed_results'].to_csv(method_detailed_path, index=False)
                    print(f"  Saved detailed reaction data for {method}: {method_detailed_path}")

            if organism_detailed:
                organism_detailed_df = pd.concat(organism_detailed, ignore_index=True)
                # Name file according to TRAIN_DATA_RATIO
                if self.train_data_ratio == "wildtype":
                    organism_detailed_path = f"{organism_dir}/detailed.csv"
                else:
                    organism_detailed_path = f"{organism_dir}/detailed_{self.train_data_ratio}.csv"
                organism_detailed_df.to_csv(organism_detailed_path, index=False)
                print(f"  Combined detailed results saved to: {organism_detailed_path}")

        return summary_df

    def print_summary_report(self, summary_df: pd.DataFrame):
        """Print a textual summary report."""
        print("\n" + "=" * 100)
        print("Multi-organism, multi-method knockout report")
        print("=" * 100)

        # Show results per organism
        for organism in summary_df['Organism'].unique():
            organism_data = summary_df[summary_df['Organism'] == organism]

            print(f"\n{organism}:")
            print("-" * 80)
            print(f"{'Method':<12} {'Avg MSE':<12} {'Avg RMSE':<12} {'Avg R^2':<12} "
                  f"{'Success':<8} {'Failed':<8} {'Success %':<10}")
            print("-" * 80)

            for _, row in organism_data.iterrows():
                if not pd.isna(row['Average_R2']):
                    print(f"{row['Method']:<12} {row['Average_MSE']:<12.6f} "
                          f"{row['Average_RMSE']:<12.6f} {row['Average_R2']:<12.6f} "
                          f"{row['Success_Count']:<8.0f} {row['Failed_Count']:<8.0f} "
                          f"{row['Success_Rate']:<10.2f}%")
                else:
                    print(f"{row['Method']:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} "
                          f"{row['Success_Count']:<8.0f} {row['Failed_Count']:<8.0f} "
                          f"{row['Success_Rate']:<10.2f}%")

            # Highlight best methods for this organism
            valid_organism_data = organism_data[~organism_data['Average_R2'].isna()]
            if not valid_organism_data.empty:
                best_r2_method = valid_organism_data.loc[valid_organism_data['Average_R2'].idxmax(), 'Method']
                best_mse_method = valid_organism_data.loc[valid_organism_data['Average_MSE'].idxmin(), 'Method']
                print(f"\n  Best methods for {organism}:")
                print(f"    Highest R^2: {best_r2_method}")
                print(f"    Lowest MSE: {best_mse_method}")

        # Overall best-performing methods
        print(f"\nOverall results:")
        print("-" * 80)
        valid_data = summary_df[~summary_df['Average_R2'].isna()]
        if not valid_data.empty:
            overall_best_r2 = valid_data.loc[valid_data['Average_R2'].idxmax()]
            overall_best_mse = valid_data.loc[valid_data['Average_MSE'].idxmin()]

            print(f"Highest R^2: {overall_best_r2['Method']} ({overall_best_r2['Organism']}) "
                  f"- R^2={overall_best_r2['Average_R2']:.6f}")
            print(f"Lowest MSE: {overall_best_mse['Method']} ({overall_best_mse['Organism']}) "
                  f"- MSE={overall_best_mse['Average_MSE']:.6f}")

        print("=" * 100)

def main():
    """Main entry point."""
    # ================================================================================
    # Training data selection hyperparameters (must match training configuration)
    # ================================================================================
    # Options:
    #   "wildtype"       - train on wild type only, evaluate all mutants
    #   "customize"      - use a custom sample list (set CUSTOMIZE_SAMPLES)
    #   numeric (0.0-1.0) - fraction of mutants assigned to the test set
    TRAIN_DATA_RATIO = "wildtype"
    # TRAIN_DATA_RATIO = "customize"
    # TRAIN_DATA_RATIO = 0.8  # keep consistent with training

    # Custom sample list (only when TRAIN_DATA_RATIO == "customize")
    # Names must match the Id column in 13C_data.csv
    CUSTOMIZE_SAMPLES = ["WILD TYPE"]
    # color = "#96CCEA"
    # color = "#B2A3DD"
    # color = "#ED949A"
    # Random seed (match training for identical splits)
    RANDOM_SEED = 42

    # Derive model suffix from TRAIN_DATA_RATIO (must align with training)
    if TRAIN_DATA_RATIO == "wildtype":
        MODEL_SUFFIX = ""
    elif TRAIN_DATA_RATIO == "customize":
        MODEL_SUFFIX = "_customize"
    else:
        MODEL_SUFFIX = f"_{TRAIN_DATA_RATIO}"

    # ================================================================================
    # Evaluation mode hyperparameters
    # ================================================================================
    # Options:
    #   "central_60"       - fixed set of 60 central carbon reactions
    #   "extend_reactions" - extended reaction set (pathways + reverse reactions)
    #   "threshold"        - thresholded reactions (truth > flux_threshold)
    #   "all_reactions"    - all reactions (not recommended)
    # EVALUATION_MODE = "threshold"
    # EVALUATION_MODE = "central_60"
    EVALUATION_MODE = "extend_reactions"
    # EVALUATION_MODE = "threshold"
    FLUX_THRESHOLD = 0.01  # only used in threshold mode

    # Instantiate analyzer
    analyzer = MultiOrganismKnockoutAnalysis(
        evaluation_mode=EVALUATION_MODE,
        flux_threshold=FLUX_THRESHOLD,
        train_data_ratio=TRAIN_DATA_RATIO,
        random_seed=RANDOM_SEED,
        model_suffix=MODEL_SUFFIX,
        customize_samples=CUSTOMIZE_SAMPLES if TRAIN_DATA_RATIO == "customize" else None
    )

    # Choose organisms/methods as needed
    # organisms = ["Bacillus subtilis"]
    organisms = ["iECDH1ME8569_1439"]
    # methods = ["FluxGen"]
    # methods = ["FluxGen", "KinLLM", "fba"]
    methods = None
    try:
        # Run analysis
        all_results = analyzer.run_full_analysis(organisms, methods)

        # Save results
        summary_df = analyzer.save_results(all_results)

        # Print summary report
        analyzer.print_summary_report(summary_df)

        print("\nAnalysis complete! Results stored in ./results/")

    except Exception as e:
        print(f"Execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

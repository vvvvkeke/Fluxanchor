#!/usr/bin/env python
"""
Transcription factor regulatory rules module - V2.

Improvements:
1. effect_strength is automatically computed based on a more reasonable default strategy
2. Default values can be overridden with external configuration files
3. Multiple biological factors are considered when estimating regulation strength

effect_strength calculation strategy:
1. Base effect strength: set by TF category (global vs. local)
2. Regulation tier adjustment: direct regulation vs. indirect regulation
3. Evidence strength: level of experimental validation
4. Metabolic importance: central vs. peripheral metabolism

Data sources:
- RegulonDB: http://regulondb.ccg.unam.mx/
- EcoCyc: https://ecocyc.org/
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import os


class TFCategory(Enum):
    """Transcription factor categories"""
    GLOBAL_REGULATOR = "global"          # Global regulators (Crp, Fnr, ArcA, etc.)
    METABOLIC_REGULATOR = "metabolic"    # Metabolism-specific regulators (Cra, IclR, FadR, etc.)
    STRESS_RESPONSE = "stress"           # Stress-response regulators (OxyR, SoxS, etc.)
    SIGMA_FACTOR = "sigma"               # Sigma factors (RpoS, RpoN, etc.)
    LOCAL_REGULATOR = "local"            # Local regulators
    UNKNOWN = "unknown"                  # Unknown category


class EvidenceLevel(Enum):
    """Regulatory evidence strength"""
    STRONG = "strong"        # Multiple experiments (e.g., ChIP-seq + expression profiling + in vitro binding)
    MODERATE = "moderate"    # Partial experimental validation
    WEAK = "weak"            # Computational prediction or literature inference only
    INFERRED = "inferred"    # Inferred from homologous genes


class MetabolicImportance(Enum):
    """Metabolic pathway importance"""
    CENTRAL = "central"      # Central metabolism (glycolysis, TCA, PPP, etc.)
    AMINO_ACID = "amino_acid"  # Amino acid metabolism
    ENERGY = "energy"        # Energy metabolism (respiratory chain, etc.)
    PERIPHERAL = "peripheral"  # Peripheral metabolism


@dataclass
class RegulatoryRuleV2:
    """Regulatory rule data class - V2"""
    tf_name: str                          # Transcription factor name
    target_genes: List[str]               # List of regulated genes
    target_reactions: List[str]           # Regulated reactions (BiGG IDs)
    regulation_type: str                  # Regulation type: "activation" or "repression"

    # Additional attributes used to compute effect_strength
    tf_category: TFCategory = TFCategory.UNKNOWN
    evidence_level: EvidenceLevel = EvidenceLevel.MODERATE
    metabolic_importance: MetabolicImportance = MetabolicImportance.PERIPHERAL
    is_direct_regulation: bool = True     # Whether the TF directly regulates the target

    # Optional: manually specify effect_strength (overrides the automatic value)
    manual_effect_strength: Optional[float] = None

    notes: str = ""

    @property
    def effect_strength(self) -> float:
        """
        Calculate the effect strength

        Return range: 0.0 - 1.0
        - 0.0: the enzyme is not expressed when the TF is knocked out
        - 1.0: enzyme expression is unchanged when the TF is knocked out

        Activation rules: smaller values mean expression drops more after the knockout
        Repression rules: typically return 1.0 (relief of repression, expression recovers)
        """
        # Return immediately if a manual value is provided
        if self.manual_effect_strength is not None:
            return self.manual_effect_strength

        # Repression rules relieve inhibition after knockout, so expression increases or stays the same
        if self.regulation_type == "repression":
            return 1.0

        # Compute the effect strength for activation rules
        return self._calculate_activation_effect()

    def _calculate_activation_effect(self) -> float:
        """Compute effect strength for activation rules"""

        # 1. Base effect strength (by TF category)
        base_effect = {
            TFCategory.GLOBAL_REGULATOR: 0.15,    # Global regulators have strong effects
            TFCategory.METABOLIC_REGULATOR: 0.20, # Metabolic regulators have substantial effects
            TFCategory.STRESS_RESPONSE: 0.30,     # Stress responses are often condition-dependent
            TFCategory.SIGMA_FACTOR: 0.25,        # Sigma factors typically have sizable effects
            TFCategory.LOCAL_REGULATOR: 0.40,     # Local regulators usually have smaller effects
            TFCategory.UNKNOWN: 0.35,             # Use a conservative value for unknown categories
        }.get(self.tf_category, 0.35)

        # 2. Evidence strength modifier
        evidence_modifier = {
            EvidenceLevel.STRONG: 1.0,      # Strong evidence, no adjustment
            EvidenceLevel.MODERATE: 1.2,    # Moderate evidence, effect may be overestimated
            EvidenceLevel.WEAK: 1.5,        # Weak evidence, be more conservative
            EvidenceLevel.INFERRED: 1.8,    # Inferred only, most conservative
        }.get(self.evidence_level, 1.2)

        # 3. Metabolic importance modifier
        importance_modifier = {
            MetabolicImportance.CENTRAL: 0.8,      # Central metabolism is tightly regulated
            MetabolicImportance.ENERGY: 0.85,      # Energy metabolism
            MetabolicImportance.AMINO_ACID: 0.95,  # Amino acid metabolism
            MetabolicImportance.PERIPHERAL: 1.1,   # Peripheral metabolism is more loosely regulated
        }.get(self.metabolic_importance, 1.0)

        # 4. Direct vs. indirect regulation modifier
        direct_modifier = 1.0 if self.is_direct_regulation else 1.3

        # Calculate the final effect strength
        effect = base_effect * evidence_modifier * importance_modifier * direct_modifier

        # Clamp the value to [0.05, 0.95]
        return max(0.05, min(0.95, effect))


# =============================================================================
# TF category definitions
# =============================================================================

TF_CATEGORIES: Dict[str, TFCategory] = {
    # Global regulators
    "Crp": TFCategory.GLOBAL_REGULATOR,
    "Fnr": TFCategory.GLOBAL_REGULATOR,
    "ArcA": TFCategory.GLOBAL_REGULATOR,
    "ArcB": TFCategory.GLOBAL_REGULATOR,
    "Fis": TFCategory.GLOBAL_REGULATOR,
    "Hns": TFCategory.GLOBAL_REGULATOR,
    "IHFA": TFCategory.GLOBAL_REGULATOR,
    "IHFB": TFCategory.GLOBAL_REGULATOR,
    "Lrp": TFCategory.GLOBAL_REGULATOR,

    # Metabolism-specific regulators
    "Cra": TFCategory.METABOLIC_REGULATOR,
    "IclR": TFCategory.METABOLIC_REGULATOR,
    "FadR": TFCategory.METABOLIC_REGULATOR,
    "PdhR": TFCategory.METABOLIC_REGULATOR,
    "GlpR": TFCategory.METABOLIC_REGULATOR,
    "PurR": TFCategory.METABOLIC_REGULATOR,
    "NagC": TFCategory.METABOLIC_REGULATOR,
    "Mlc": TFCategory.METABOLIC_REGULATOR,
    "GcvA": TFCategory.METABOLIC_REGULATOR,
    "CysB": TFCategory.METABOLIC_REGULATOR,
    "ArgR": TFCategory.METABOLIC_REGULATOR,
    "NtrC": TFCategory.METABOLIC_REGULATOR,

    # Stress-response regulators
    "OxyR": TFCategory.STRESS_RESPONSE,
    "SoxR": TFCategory.STRESS_RESPONSE,
    "SoxS": TFCategory.STRESS_RESPONSE,
    "Fur": TFCategory.STRESS_RESPONSE,
    "GadE": TFCategory.STRESS_RESPONSE,
    "GadW": TFCategory.STRESS_RESPONSE,
    "GadX": TFCategory.STRESS_RESPONSE,
    "MarA": TFCategory.STRESS_RESPONSE,
    "MarR": TFCategory.STRESS_RESPONSE,

    # Sigma factors
    "RpoD": TFCategory.SIGMA_FACTOR,
    "RpoN": TFCategory.SIGMA_FACTOR,
    "RpoS": TFCategory.SIGMA_FACTOR,
    "FliA": TFCategory.SIGMA_FACTOR,
    "FecI": TFCategory.SIGMA_FACTOR,

    # Two-component system regulators
    "NarL": TFCategory.LOCAL_REGULATOR,
    "NarP": TFCategory.LOCAL_REGULATOR,
    "DcuR": TFCategory.LOCAL_REGULATOR,
    "PhoB": TFCategory.LOCAL_REGULATOR,
    "OmpR": TFCategory.LOCAL_REGULATOR,
    "TorR": TFCategory.LOCAL_REGULATOR,
    "BasR": TFCategory.LOCAL_REGULATOR,
    "CpxR": TFCategory.LOCAL_REGULATOR,
    "EvgA": TFCategory.LOCAL_REGULATOR,
    "QseB": TFCategory.LOCAL_REGULATOR,
    "RcsA": TFCategory.LOCAL_REGULATOR,
    "RcsB": TFCategory.LOCAL_REGULATOR,
    "CreB": TFCategory.LOCAL_REGULATOR,
}

# Mapping from reactions to metabolic importance
REACTION_IMPORTANCE: Dict[str, MetabolicImportance] = {
    # Glycolysis
    "PGI": MetabolicImportance.CENTRAL,
    "PFK": MetabolicImportance.CENTRAL,
    "FBA": MetabolicImportance.CENTRAL,
    "TPI": MetabolicImportance.CENTRAL,
    "GAPD": MetabolicImportance.CENTRAL,
    "PGK": MetabolicImportance.CENTRAL,
    "PGM": MetabolicImportance.CENTRAL,
    "ENO": MetabolicImportance.CENTRAL,
    "PYK": MetabolicImportance.CENTRAL,

    # Gluconeogenesis
    "PPS": MetabolicImportance.CENTRAL,
    "PPCK": MetabolicImportance.CENTRAL,
    "FBP": MetabolicImportance.CENTRAL,

    # TCA cycle
    "PDH": MetabolicImportance.CENTRAL,
    "CS": MetabolicImportance.CENTRAL,
    "ACONTa": MetabolicImportance.CENTRAL,
    "ACONTb": MetabolicImportance.CENTRAL,
    "ICDHyr": MetabolicImportance.CENTRAL,
    "AKGDH": MetabolicImportance.CENTRAL,
    "SUCOAS": MetabolicImportance.CENTRAL,
    "SUCDi": MetabolicImportance.CENTRAL,
    "FUM": MetabolicImportance.CENTRAL,
    "MDH": MetabolicImportance.CENTRAL,

    # Glyoxylate cycle
    "ICL": MetabolicImportance.CENTRAL,
    "MALS": MetabolicImportance.CENTRAL,

    # Pentose phosphate pathway
    "G6PDH2r": MetabolicImportance.CENTRAL,
    "PGL": MetabolicImportance.CENTRAL,
    "GND": MetabolicImportance.CENTRAL,
    "RPI": MetabolicImportance.CENTRAL,
    "RPE": MetabolicImportance.CENTRAL,
    "TKT1": MetabolicImportance.CENTRAL,
    "TKT2": MetabolicImportance.CENTRAL,
    "TALA": MetabolicImportance.CENTRAL,

    # Entner-Doudoroff (ED) pathway
    "EDD": MetabolicImportance.CENTRAL,
    "EDA": MetabolicImportance.CENTRAL,

    # Respiratory chain
    "NADH16pp": MetabolicImportance.ENERGY,
    "NADH17pp": MetabolicImportance.ENERGY,
    "CYTBO3_4pp": MetabolicImportance.ENERGY,
    "ATPS4rpp": MetabolicImportance.ENERGY,
    "FRD2": MetabolicImportance.ENERGY,
    "FRD3": MetabolicImportance.ENERGY,

    # Amino acid metabolism
    "GLNS": MetabolicImportance.AMINO_ACID,
    "GLUDy": MetabolicImportance.AMINO_ACID,
    "ACLS": MetabolicImportance.AMINO_ACID,
    "SERAT": MetabolicImportance.AMINO_ACID,
    "CYSS": MetabolicImportance.AMINO_ACID,
    "ARGSS": MetabolicImportance.AMINO_ACID,

    # Transport
    "GLCptspp": MetabolicImportance.CENTRAL,
    "ACS": MetabolicImportance.CENTRAL,
}


def get_tf_category(tf_name: str) -> TFCategory:
    """Get the category of a transcription factor"""
    return TF_CATEGORIES.get(tf_name, TFCategory.UNKNOWN)


def get_reaction_importance(reaction_id: str) -> MetabolicImportance:
    """Get the metabolic importance for a reaction"""
    # Remove suffixes
    base_id = reaction_id.replace('_reverse', '').split('_num')[0]
    return REACTION_IMPORTANCE.get(base_id, MetabolicImportance.PERIPHERAL)


# =============================================================================
# Regulatory rule definitions - V2
# =============================================================================

def create_regulatory_rules_v2() -> Dict[str, List[RegulatoryRuleV2]]:
    """
    Create regulatory rules (V2 version)

    effect_strength is automatically computed from TF category, evidence strength,
    and metabolic importance
    """
    rules = {}

    # =========================================================================
    # Cra (FruR) - carbon catabolite repressor
    # =========================================================================
    rules["Cra"] = [
        # Activated reactions
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["ppsA"],
            target_reactions=["PPS"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra activates ppsA expression for gluconeogenesis"
        ),
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["pckA"],
            target_reactions=["PPCK"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra activates pckA expression for gluconeogenesis"
        ),
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["fbp"],
            target_reactions=["FBP"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra activates fbp expression for gluconeogenesis"
        ),
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["aceB", "aceA"],
            target_reactions=["ICL", "MALS"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra activates the glyoxylate shunt"
        ),
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["icd"],
            target_reactions=["ICDHyr"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra activates icd expression"
        ),
        # Repressed reactions
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["pykF", "pykA"],
            target_reactions=["PYK"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra represses pyruvate kinase"
        ),
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["pfkA", "pfkB"],
            target_reactions=["PFK"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra represses phosphofructokinase"
        ),
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["edd", "eda"],
            target_reactions=["EDD", "EDA"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra represses the ED pathway"
        ),
    ]

    # =========================================================================
    # Crp - cAMP receptor protein (global regulator)
    # =========================================================================
    rules["Crp"] = [
        RegulatoryRuleV2(
            tf_name="Crp",
            target_genes=["aceB", "aceA", "aceK"],
            target_reactions=["ICL", "MALS"],
            regulation_type="activation",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Crp activates the glyoxylate shunt"
        ),
        RegulatoryRuleV2(
            tf_name="Crp",
            target_genes=["sdhC", "sdhD", "sdhA", "sdhB"],
            target_reactions=["SUCDi"],
            regulation_type="activation",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Crp activates the SDH complex"
        ),
        RegulatoryRuleV2(
            tf_name="Crp",
            target_genes=["acs"],
            target_reactions=["ACS"],
            regulation_type="activation",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Crp activates acs expression"
        ),
    ]

    # =========================================================================
    # ArcA/ArcB - two-component anaerobic regulation system
    # =========================================================================
    rules["ArcA"] = [
        RegulatoryRuleV2(
            tf_name="ArcA",
            target_genes=["sdhC", "sdhD", "sdhA", "sdhB"],
            target_reactions=["SUCDi"],
            regulation_type="repression",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="ArcA represses SDH under anaerobic conditions"
        ),
        RegulatoryRuleV2(
            tf_name="ArcA",
            target_genes=["cyoA", "cyoB", "cyoC", "cyoD", "cyoE"],
            target_reactions=["CYTBO3_4pp"],
            regulation_type="repression",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="ArcA represses cytochrome oxidase under anaerobic conditions"
        ),
        RegulatoryRuleV2(
            tf_name="ArcA",
            target_genes=["icd"],
            target_reactions=["ICDHyr"],
            regulation_type="repression",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="ArcA represses TCA-cycle enzymes"
        ),
        RegulatoryRuleV2(
            tf_name="ArcA",
            target_genes=["aceE", "aceF", "lpdA"],
            target_reactions=["PDH"],
            regulation_type="repression",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="ArcA represses the PDH complex"
        ),
        RegulatoryRuleV2(
            tf_name="ArcA",
            target_genes=["gltA"],
            target_reactions=["CS"],
            regulation_type="repression",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="ArcA represses citrate synthase"
        ),
    ]

    rules["ArcB"] = [
        RegulatoryRuleV2(
            tf_name="ArcB",
            target_genes=["sdhC", "sdhD", "sdhA", "sdhB"],
            target_reactions=["SUCDi"],
            regulation_type="repression",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=False,  # Indirect via ArcA
            notes="The ArcB-ArcA system regulates SDH"
        ),
    ]

    # =========================================================================
    # Fnr - anaerobic regulator
    # =========================================================================
    rules["Fnr"] = [
        RegulatoryRuleV2(
            tf_name="Fnr",
            target_genes=["ndh"],
            target_reactions=["NADH16pp", "NADH17pp"],
            regulation_type="repression",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="Fnr regulates the respiratory chain"
        ),
        RegulatoryRuleV2(
            tf_name="Fnr",
            target_genes=["cyoA", "cyoB", "cyoC", "cyoD"],
            target_reactions=["CYTBO3_4pp"],
            regulation_type="repression",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="Fnr represses aerobic respiration"
        ),
        RegulatoryRuleV2(
            tf_name="Fnr",
            target_genes=["frdA", "frdB", "frdC", "frdD"],
            target_reactions=["FRD2", "FRD3"],
            regulation_type="activation",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="Fnr activates anaerobic respiratory enzymes"
        ),
    ]

    # =========================================================================
    # PdhR - pyruvate dehydrogenase regulator
    # =========================================================================
    rules["PdhR"] = [
        RegulatoryRuleV2(
            tf_name="PdhR",
            target_genes=["aceE", "aceF", "lpdA"],
            target_reactions=["PDH"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="PdhR represses expression of the PDH complex"
        ),
        RegulatoryRuleV2(
            tf_name="PdhR",
            target_genes=["ndh"],
            target_reactions=["NADH16pp"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="PdhR represses NADH dehydrogenase"
        ),
    ]

    # =========================================================================
    # IclR - glyoxylate shunt regulator
    # =========================================================================
    rules["IclR"] = [
        RegulatoryRuleV2(
            tf_name="IclR",
            target_genes=["aceB", "aceA", "aceK"],
            target_reactions=["ICL", "MALS"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="IclR represses the glyoxylate shunt"
        ),
    ]

    # =========================================================================
    # FadR - fatty-acid metabolism regulator
    # =========================================================================
    rules["FadR"] = [
        RegulatoryRuleV2(
            tf_name="FadR",
            target_genes=["fadD", "fadE", "fadB", "fadA"],
            target_reactions=["FACOAL60", "FACOAL80", "ACOAD1f", "ACOAD2f"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="FadR represses fatty-acid degradation"
        ),
        RegulatoryRuleV2(
            tf_name="FadR",
            target_genes=["fabA", "fabB"],
            target_reactions=["3OAS60", "3OAS80", "3OAS100"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="FadR activates fatty-acid biosynthesis"
        ),
    ]

    # =========================================================================
    # Fur - iron metabolism regulator
    # =========================================================================
    rules["Fur"] = [
        RegulatoryRuleV2(
            tf_name="Fur",
            target_genes=["sdhC", "sdhD", "sdhA", "sdhB"],
            target_reactions=["SUCDi"],
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=False,  # Indirect through iron availability
            notes="Fur influences expression of iron-containing proteins"
        ),
        RegulatoryRuleV2(
            tf_name="Fur",
            target_genes=["acnA", "acnB"],
            target_reactions=["ACONTa", "ACONTb"],
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=False,
            notes="Fur influences expression of iron-containing proteins"
        ),
    ]

    # =========================================================================
    # NarL/NarP - nitrate respiration regulators
    # =========================================================================
    rules["NarL"] = [
        RegulatoryRuleV2(
            tf_name="NarL",
            target_genes=["narG", "narH", "narJ", "narI"],
            target_reactions=["NO3R1pp", "NO3R2pp"],
            regulation_type="activation",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="NarL activates nitrate respiration"
        ),
        RegulatoryRuleV2(
            tf_name="NarL",
            target_genes=["frdA", "frdB", "frdC", "frdD"],
            target_reactions=["FRD2", "FRD3"],
            regulation_type="repression",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="NarL represses fumarate respiration"
        ),
    ]

    rules["NarP"] = [
        RegulatoryRuleV2(
            tf_name="NarP",
            target_genes=["narG", "narH", "narJ", "narI"],
            target_reactions=["NO3R1pp", "NO3R2pp"],
            regulation_type="activation",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="NarP activates nitrate respiration"
        ),
    ]

    # =========================================================================
    # GadE/GadW/GadX - acid resistance regulators
    # =========================================================================
    for tf in ["GadE", "GadW", "GadX"]:
        rules[tf] = [
            RegulatoryRuleV2(
                tf_name=tf,
                target_genes=["gadA", "gadB", "gadC"],
                target_reactions=["GLUDy"],
                regulation_type="activation",
                tf_category=TFCategory.STRESS_RESPONSE,
                evidence_level=EvidenceLevel.STRONG,
                metabolic_importance=MetabolicImportance.AMINO_ACID,
                is_direct_regulation=True,
                notes=f"{tf} activates the acid resistance system"
            ),
        ]

    # =========================================================================
    # Lrp - leucine-responsive regulatory protein
    # =========================================================================
    rules["Lrp"] = [
        RegulatoryRuleV2(
            tf_name="Lrp",
            target_genes=["ilvB", "ilvN", "ilvH", "ilvI"],
            target_reactions=["ACLS"],
            regulation_type="activation",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="Lrp activates branched-chain amino-acid synthesis"
        ),
        RegulatoryRuleV2(
            tf_name="Lrp",
            target_genes=["gltB", "gltD"],
            target_reactions=["GLUDy", "GLNS"],
            regulation_type="activation",
            tf_category=TFCategory.GLOBAL_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="Lrp activates nitrogen metabolism"
        ),
    ]

    # =========================================================================
    # SoxR/SoxS - oxidative stress response
    # =========================================================================
    rules["SoxS"] = [
        RegulatoryRuleV2(
            tf_name="SoxS",
            target_genes=["sodA"],
            target_reactions=["SPODM"],
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="SoxS activates antioxidant enzymes"
        ),
        RegulatoryRuleV2(
            tf_name="SoxS",
            target_genes=["zwf"],
            target_reactions=["G6PDH2r"],
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="SoxS activates NADPH-producing enzymes"
        ),
    ]

    rules["SoxR"] = [
        RegulatoryRuleV2(
            tf_name="SoxR",
            target_genes=["soxS"],
            target_reactions=[],
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="SoxR senses oxidative stress and activates SoxS"
        ),
    ]

    # =========================================================================
    # OxyR - peroxide stress response
    # =========================================================================
    rules["OxyR"] = [
        RegulatoryRuleV2(
            tf_name="OxyR",
            target_genes=["katG", "ahpC", "ahpF"],
            target_reactions=["CAT"],
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="OxyR activates peroxide-detoxifying enzymes"
        ),
    ]

    # =========================================================================
    # CysB - cysteine metabolism regulation
    # =========================================================================
    rules["CysB"] = [
        RegulatoryRuleV2(
            tf_name="CysB",
            target_genes=["cysE", "cysK"],
            target_reactions=["SERAT", "CYSS"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="CysB activates cysteine biosynthesis"
        ),
    ]

    # =========================================================================
    # GlpR - glycerol metabolism regulation
    # =========================================================================
    rules["GlpR"] = [
        RegulatoryRuleV2(
            tf_name="GlpR",
            target_genes=["glpK", "glpD", "glpF"],
            target_reactions=["GLYK", "G3PD2"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="GlpR represses glycerol utilization"
        ),
    ]

    # =========================================================================
    # Mlc - PTS system regulation
    # =========================================================================
    rules["Mlc"] = [
        RegulatoryRuleV2(
            tf_name="Mlc",
            target_genes=["ptsG", "ptsH", "ptsI"],
            target_reactions=["GLCptspp"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Mlc represses the PTS system"
        ),
    ]

    # =========================================================================
    # PurR - purine metabolism regulation
    # =========================================================================
    rules["PurR"] = [
        RegulatoryRuleV2(
            tf_name="PurR",
            target_genes=["purA", "purB", "purC", "purD"],
            target_reactions=["ADSS", "ADSL1r", "PRFGS"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="PurR represses de novo purine biosynthesis"
        ),
    ]

    # =========================================================================
    # NtrC - nitrogen metabolism regulation
    # =========================================================================
    rules["NtrC"] = [
        RegulatoryRuleV2(
            tf_name="NtrC",
            target_genes=["glnA", "glnL", "glnG"],
            target_reactions=["GLNS"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="NtrC activates nitrogen metabolism"
        ),
    ]

    # =========================================================================
    # GcvA - glycine cleavage system regulation
    # =========================================================================
    rules["GcvA"] = [
        RegulatoryRuleV2(
            tf_name="GcvA",
            target_genes=["gcvT", "gcvH", "gcvP"],
            target_reactions=["GLYCL"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="GcvA activates the glycine cleavage system"
        ),
    ]

    # =========================================================================
    # Sigma factors
    # =========================================================================
    rules["RpoN"] = [
        RegulatoryRuleV2(
            tf_name="RpoN",
            target_genes=["glnA", "nifA"],
            target_reactions=["GLNS"],
            regulation_type="activation",
            tf_category=TFCategory.SIGMA_FACTOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="RpoN (sigma54) regulates nitrogen metabolism"
        ),
    ]

    rules["RpoS"] = [
        RegulatoryRuleV2(
            tf_name="RpoS",
            target_genes=["katE", "osmC", "dps"],
            target_reactions=["CAT"],
            regulation_type="activation",
            tf_category=TFCategory.SIGMA_FACTOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="RpoS (sigma38) regulates stationary-phase responses"
        ),
    ]
    
    # ArgR - arginine repressor
    rules["ArgR"] = [
        RegulatoryRuleV2(
            tf_name="ArgR",
            target_genes=["argA", "argB", "argC", "argD"],
            target_reactions=["ARGSL", "AGPR", "ACODA"], # Upper arginine biosynthesis steps
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="ArgR represses the arginine biosynthetic operon"
        ),
        RegulatoryRuleV2(
            tf_name="ArgR",
            target_genes=["argF", "argI"],
            target_reactions=["OCBT"], # Ornithine carbamoyltransferase
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True
        )
    ]

    # TrpR - tryptophan repressor
    rules["TrpR"] = [
        RegulatoryRuleV2(
            tf_name="TrpR",
            target_genes=["trpE", "trpD"],
            target_reactions=["ANS", "ANPRT"], # Anthranilate synthase (key steps)
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="TrpR represses tryptophan biosynthesis"
        )
    ]

    # TyrR - aromatic amino-acid regulation
    rules["TyrR"] = [
        RegulatoryRuleV2(
            tf_name="TyrR",
            target_genes=["aroF", "tyrA"],
            target_reactions=["DDPA", "PPND"], # Shared aromatic amino-acid biosynthesis steps
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="TyrR mainly represses tyrosine and phenylalanine synthesis"
        ),
        # TyrR activates mtr (tryptophan permease)
        RegulatoryRuleV2(
            tf_name="TyrR",
            target_genes=["mtr"],
            target_reactions=["TRPt2rpp"], 
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True
        )
    ]

    # MetJ - methionine repressor
    rules["MetJ"] = [
        RegulatoryRuleV2(
            tf_name="MetJ",
            target_genes=["metA", "metB", "metC", "metE"],
            target_reactions=["HSST", "CYSS", "HCYS"], # Methionine biosynthesis pathway
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="MetJ represses methionine biosynthesis"
        )
    ]

    # =========================================================================
    # Additional coverage: secondary carbon sources and transport (from default_tfs)
    # =========================================================================

    # NagC - N-acetylglucosamine repressor
    rules["NagC"] = [
        RegulatoryRuleV2(
            tf_name="NagC",
            target_genes=["nagE", "nagB", "nagA"],
            target_reactions=["NAGt2pp", "ACGAMPM", "ACGAM6P"], # NAG utilization
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL, # NAG serves as a carbon source
            is_direct_regulation=True,
            notes="NagC represses N-acetylglucosamine utilization"
        )
    ]

    # DcuR - C4-dicarboxylate transport (fumarate/succinate)
    rules["DcuR"] = [
        RegulatoryRuleV2(
            tf_name="DcuR",
            target_genes=["dctA"],
            target_reactions=["SUCCt2_2pp", "FUMt2_2pp"], # Succinate/fumarate transport
            regulation_type="activation",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="DcuR activates C4-dicarboxylate transport under anaerobic conditions"
        )
    ]

    # BetI - choline/betaine regulator
    rules["BetI"] = [
        RegulatoryRuleV2(
            tf_name="BetI",
            target_genes=["betA", "betB"],
            target_reactions=["CHOLD", "BETALDHx"], # Choline oxidation
            regulation_type="repression",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="BetI represses conversion of choline to betaine"
        )
    ]

    # PhoB - phosphate starvation response
    rules["PhoB"] = [
        RegulatoryRuleV2(
            tf_name="PhoB",
            target_genes=["pstS", "pstC", "pstA", "pstB"],
            target_reactions=["PIt2rpp"], # Inorganic phosphate transport (approximation)
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="PhoB activates high-affinity phosphate transport during phosphate starvation"
        )
    ]

    # =========================================================================
    # Additional coverage: anaerobic and specialized metabolism (from default_tfs)
    # =========================================================================

    # FhlA - formate hydrogen lyase regulation
    rules["FhlA"] = [
        RegulatoryRuleV2(
            tf_name="FhlA",
            target_genes=["fdhF", "hycE"],
            target_reactions=["FHL"], # Formate hydrogen lyase (formate -> CO2 + H2)
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="FhlA activates hydrogen production from formate"
        )
    ]

    # AppY - acid phosphatase/anaerobic regulator
    rules["AppY"] = [
        RegulatoryRuleV2(
            tf_name="AppY",
            target_genes=["hyaA", "hyaB"],
            target_reactions=["HYD1pp"], # Hydrogenase 1
            regulation_type="activation",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="AppY activates anaerobic hydrogenase"
        )
    ]

    # FecI - ferric citrate transport
    rules["FecI"] = [
        RegulatoryRuleV2(
            tf_name="FecI",
            target_genes=["fecA", "fecB"],
            target_reactions=["FECR_1", "FECR_2"], # Ferric citrate reduction/transport (verify model IDs)
            regulation_type="activation",
            tf_category=TFCategory.SIGMA_FACTOR, # ECF sigma factor
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="FecI activates import of ferric citrate"
        )
    ]


    # MarA - multiple antibiotic resistance regulator
    # MarA is notable: to resist oxidative stress/antibiotics it activates the PPP for NADPH and closes porins
    rules["MarA"] = [
        RegulatoryRuleV2(
            tf_name="MarA",
            target_genes=["zwf"], 
            target_reactions=["G6PDH2r"], # Activates G6P dehydrogenase to boost NADPH supply
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="MarA activates Zwf to provide NADPH for antioxidation/detoxification"
        ),
        RegulatoryRuleV2(
            tf_name="MarA",
            target_genes=["ompF"],
            target_reactions=["OMP_DIFFUSION"], # Assuming an OmpF-like outer-membrane diffusion reaction exists
            # Note: if the model lacks explicit porin reactions, this rule may have no effect but remains conceptually important
            regulation_type="repression",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="MarA downregulates OmpF to limit antibiotic influx"
        )
    ]
    # Rob largely overlaps with MarA
    rules["Rob"] = rules["MarA"]

    # KdgR - hexuronate/pectin metabolism (ED branch)
    rules["KdgR"] = [
        RegulatoryRuleV2(
            tf_name="KdgR",
            target_genes=["eda", "edd"],
            target_reactions=["EDA", "EDD"], # Key ED-pathway enzymes
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="KdgR represses the ED pathway (cooperating/competing with Cra)"
        ),
        RegulatoryRuleV2(
            tf_name="KdgR",
            target_genes=["kdgK"],
            target_reactions=["K2GK"], # 2-dehydro-3-deoxygluconokinase
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True
        )
    ]

    # CytR - nucleoside catabolism
    # CytR blocks nucleoside utilization when glucose is available
    rules["CytR"] = [
        RegulatoryRuleV2(
            tf_name="CytR",
            target_genes=["udp"],
            target_reactions=["URIDK2r"], # Uridine kinase
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="CytR represses nucleoside degradation"
        ),
        RegulatoryRuleV2(
            tf_name="CytR",
            target_genes=["cdd"],
            target_reactions=["CYTD"], # Cytidine deaminase
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True
        )
    ]

    # PrpR - propionate metabolism (methylcitrate cycle)
    rules["PrpR"] = [
        RegulatoryRuleV2(
            tf_name="PrpR",
            target_genes=["prpC", "prpB"],
            target_reactions=["MCITL2", "MCITS"], # Methylcitrate synthase/lyase
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL, # Links into the TCA cycle
            is_direct_regulation=True,
            notes="PrpR activates propionate utilization"
        )
    ]

    # LldR - L-lactate utilization
    rules["LldR"] = [
        RegulatoryRuleV2(
            tf_name="LldR",
            target_genes=["lldD"],
            target_reactions=["L_LACD2"], # L-lactate dehydrogenase (quinone)
            regulation_type="repression", 
            # Note: LldR is a repressor that is relieved when lactate is present.
            # Defining it as a repressor means deleting LldR derestricts lactate utilization.
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="LldR represses lactate utilization; deletion causes constitutive expression"
        )
    ]

    # GlcC - glycolate utilization
    rules["GlcC"] = [
        RegulatoryRuleV2(
            tf_name="GlcC",
            target_genes=["glcD", "glcE", "glcF"],
            target_reactions=["GLYCTO2", "GLYCTO4"], # Glycolate oxidase
            regulation_type="activation", # GlcC acts as an activator
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="GlcC activates glycolate oxidation"
        )
    ]

    # TdcA / TdcR - anaerobic threonine metabolism
    # Critical for anaerobic fermentation that produces propionate
    tdc_rule = [
        RegulatoryRuleV2(
            tf_name="TdcA", # TdcA is the primary activator
            target_genes=["tdcB"],
            target_reactions=["THRD_L"], # Threonine deaminase (catabolic)
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="TdcA activates anaerobic threonine degradation"
        )
    ]
    rules["TdcA"] = tdc_rule
    rules["TdcR"] = tdc_rule # TdcR cooperates with TdcA
    
    # TorR - trimethylamine N-oxide (TMAO) respiration
    # Under anaerobic conditions with TMAO present, TorR activates the TMAO reductase system
    rules["TorR"] = [
        RegulatoryRuleV2(
            tf_name="TorR",
            target_genes=["torA", "torC"],
            target_reactions=["TMAOR1pp", "TMAOR2pp"], # TMAO reductase
            regulation_type="activation",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="TorR activates anaerobic respiration using TMAO as the electron acceptor"
        )
    ]

    # ModE - molybdate transport and metabolism
    # Molybdenum is essential for Nar and Fdh enzymes
    rules["ModE"] = [
        RegulatoryRuleV2(
            tf_name="ModE",
            target_genes=["modA", "modB", "modC"],
            target_reactions=["MOBd"], # Molybdate transport
            regulation_type="repression", 
            # ModE suppresses uptake at high molybdate (saves energy) and derepresses at low levels
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="ModE regulates molybdate uptake and affects anaerobic respiratory enzymes"
        )
    ]

    # UxuR - hexuronate metabolism
    # Controls utilization of glucuronate and galacturonate
    rules["UxuR"] = [
        RegulatoryRuleV2(
            tf_name="UxuR",
            target_genes=["uxuA", "uxuB"],
            target_reactions=["MAN6PI", "ALTRH"], # Isomerase pathway (verify BiGG IDs in the model)
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="UxuR represses hexuronate utilization pathways"
        )
    ]

    # IdnR - L-idonate metabolism
    rules["IdnR"] = [
        RegulatoryRuleV2(
            tf_name="IdnR",
            target_genes=["idnD", "idnO", "idnK"],
            target_reactions=["IDOND", "IDOND2"], # Idonate dehydrogenase
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="IdnR activates L-idonate utilization"
        )
    ]

    # RcsB - colanic acid capsule synthesis
    # This process consumes energy and precursors (e.g., fucose) and is usually stress-induced
    rules["RcsB"] = [
        RegulatoryRuleV2(
            tf_name="RcsB",
            target_genes=["wcaA", "wcaB", "wcaC"], # colanic acid synthesis operon
            target_reactions=["UAGDP", "UAGPT3"], # UDP-sugar metabolic pathways
            # If the model has a capsule/biomass reaction, prefer that (e.g., "COLANIC_ACID_SINK")
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="RcsB activates capsule polysaccharide synthesis, drawing on cell-wall precursors"
        )
    ]

    # =========================================================================
    # Add TFs without explicit metabolic targets (using defaults)
    # =========================================================================
    default_tfs = [
        "AlsR", "BolA", "CpxR", "CsgD", "FlhD", "Fis", "Hns",
        "IHFA", "IHFB", "OmpR", "AscG", "BasR", "ChpB", "CreB",
        "Crl", "CspA",  "EvgA", "FliZ", "FrvR",
        "HdfR", "LrhA", "MarR",
        "NsrR", "OgrK", "PerR", "PspC",
        "QseB", "RcsA", "RutR", "SdiA", "SspA", 
        "UspA", "UvrY", "YbhD", "YdeO", "YdjF",
        "YfhA", "ElbB", "FlgM", "FliA", "RpoD", "Rsd", "RseA", "RseB"
    ]

    for tf in default_tfs:
        if tf not in rules:
            rules[tf] = [
                RegulatoryRuleV2(
                    tf_name=tf,
                    target_genes=[],
                    target_reactions=[],
                    regulation_type="activation",
                    tf_category=get_tf_category(tf),
                    evidence_level=EvidenceLevel.WEAK,
                    metabolic_importance=MetabolicImportance.PERIPHERAL,
                    is_direct_regulation=True,
                    notes=f"{tf} has minimal or unknown impact on central metabolism"
                ),
            ]

    return rules


# Build the global rule dictionary
REGULATORY_RULES_V2 = create_regulatory_rules_v2()


# =============================================================================
# Functional enzyme knockouts (direct enzyme deletions)
# =============================================================================

ENZYME_KNOCKOUTS: Dict[str, List[str]] = {
    "Pgi": ["PGI"],
    "Zwf": ["G6PDH2r"],
    "SdhC": ["SUCDi"],
}


# =============================================================================
# API functions
# =============================================================================

def get_regulatory_effects(tf_name: str) -> List[RegulatoryRuleV2]:
    """Get the regulatory effects for a transcription factor"""
    tf_name_normalized = tf_name.strip()

    for key in REGULATORY_RULES_V2.keys():
        if key.lower() == tf_name_normalized.lower():
            return REGULATORY_RULES_V2[key]

    return []


def get_enzyme_knockout_reactions(enzyme_name: str) -> List[str]:
    """Get the reactions affected by a metabolic enzyme knockout"""
    for key in ENZYME_KNOCKOUTS.keys():
        if key.lower() == enzyme_name.lower():
            return ENZYME_KNOCKOUTS[key]
    return []


def is_transcription_factor(gene_name: str) -> bool:
    """Determine whether a gene encodes a transcription factor"""
    gene_name_lower = gene_name.lower()
    for key in REGULATORY_RULES_V2.keys():
        if key.lower() == gene_name_lower:
            return True
    return False


def is_metabolic_enzyme(gene_name: str) -> bool:
    """Determine whether a gene encodes a metabolic enzyme"""
    gene_name_lower = gene_name.lower()
    for key in ENZYME_KNOCKOUTS.keys():
        if key.lower() == gene_name_lower:
            return True
    return False


def get_affected_reactions_and_effects(
    tf_name: str,
    effect_type: str = "activation"
) -> Dict[str, float]:
    """Get the reactions and effect factors influenced by knocking out a TF"""
    rules = get_regulatory_effects(tf_name)
    affected_reactions = {}

    for rule in rules:
        if effect_type == "all" or rule.regulation_type == "activation":
            for rxn_id in rule.target_reactions:
                if rxn_id not in affected_reactions:
                    affected_reactions[rxn_id] = rule.effect_strength
                else:
                    affected_reactions[rxn_id] = min(
                        affected_reactions[rxn_id],
                        rule.effect_strength
                    )

    return affected_reactions


def print_regulatory_summary():
    """Print a summary of all regulatory rules"""
    print("=" * 80)
    print("TF regulatory rules summary (V2 - automatic effect_strength)")
    print("=" * 80)

    for tf_name, rules in REGULATORY_RULES_V2.items():
        activation_rules = [r for r in rules if r.regulation_type == "activation" and r.target_reactions]
        repression_rules = [r for r in rules if r.regulation_type == "repression" and r.target_reactions]

        if not activation_rules and not repression_rules:
            continue

        print(f"\n{tf_name} ({get_tf_category(tf_name).value}):")

        if activation_rules:
            print("  Activated reactions (TF knockout lowers expression):")
            for rule in activation_rules:
                reactions_str = ", ".join(rule.target_reactions)
                print(f"    - {reactions_str}")
                print(f"      effect_strength: {rule.effect_strength:.2f}")
                print(f"      (Category: {rule.tf_category.value}, "
                      f"Evidence: {rule.evidence_level.value}, "
                      f"Importance: {rule.metabolic_importance.value})")

        if repression_rules:
            print("  Repressed reactions (TF knockout relieves repression):")
            for rule in repression_rules:
                reactions_str = ", ".join(rule.target_reactions)
                print(f"    - {reactions_str} (effect_strength: {rule.effect_strength:.2f})")

    print("\n" + "=" * 80)
    print("Enzyme controls:")
    for enzyme, reactions in ENZYME_KNOCKOUTS.items():
        print(f"  {enzyme}: {', '.join(reactions)}")
    print("=" * 80)


def print_effect_strength_calculation_example():
    """Print example effect_strength calculations"""
    print("\n" + "=" * 80)
    print("effect_strength calculation examples")
    print("=" * 80)

    examples = [
        ("Cra", "PPS"),
        ("Crp", "ICL"),
        ("ArcA", "SUCDi"),
        ("Fur", "ACONTa"),
    ]

    for tf_name, rxn_id in examples:
        rules = get_regulatory_effects(tf_name)
        for rule in rules:
            if rxn_id in rule.target_reactions:
                print(f"\n{tf_name} -> {rxn_id}:")
                print(f"  TF category: {rule.tf_category.value}")
                print(f"  Evidence level: {rule.evidence_level.value}")
                print(f"  Metabolic importance: {rule.metabolic_importance.value}")
                print(f"  Direct regulation: {rule.is_direct_regulation}")
                print(f"  Calculated effect_strength: {rule.effect_strength:.3f}")
                break


if __name__ == "__main__":
    print_regulatory_summary()
    print_effect_strength_calculation_example()

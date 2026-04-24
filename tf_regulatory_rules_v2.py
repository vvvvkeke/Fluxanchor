#!/usr/bin/env python
"""
转录因子调控规则模块 - V2版本

改进：
1. effect_strength 基于更合理的默认策略自动计算
2. 支持从外部配置文件覆盖默认值
3. 考虑多种生物学因素来估算调控效应强度

effect_strength 计算策略：
1. 基础效应强度：根据调控因子类型（全局/局部）设定
2. 调控层级修正：直接调控 vs 间接调控
3. 调控证据强度：实验验证程度
4. 代谢重要性：中心代谢 vs 外周代谢

数据来源：
- RegulonDB: http://regulondb.ccg.unam.mx/
- EcoCyc: https://ecocyc.org/
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import os


class TFCategory(Enum):
    """转录因子类别"""
    GLOBAL_REGULATOR = "global"          # 全局调控因子 (Crp, Fnr, ArcA等)
    METABOLIC_REGULATOR = "metabolic"    # 代谢特异性调控因子 (Cra, IclR, FadR等)
    STRESS_RESPONSE = "stress"           # 应激响应调控因子 (OxyR, SoxS等)
    SIGMA_FACTOR = "sigma"               # Sigma因子 (RpoS, RpoN等)
    LOCAL_REGULATOR = "local"            # 局部调控因子
    UNKNOWN = "unknown"                  # 未知类别


class EvidenceLevel(Enum):
    """调控证据强度"""
    STRONG = "strong"        # 多种实验验证（如ChIP-seq + 表达谱 + 体外结合）
    MODERATE = "moderate"    # 部分实验验证
    WEAK = "weak"            # 仅计算预测或文献推断
    INFERRED = "inferred"    # 从同源基因推断


class MetabolicImportance(Enum):
    """代谢途径重要性"""
    CENTRAL = "central"      # 中心代谢（糖酵解、TCA、PPP等）
    AMINO_ACID = "amino_acid"  # 氨基酸代谢
    ENERGY = "energy"        # 能量代谢（呼吸链等）
    PERIPHERAL = "peripheral"  # 外周代谢


@dataclass
class RegulatoryRuleV2:
    """
    调控规则数据类 - V2版本

    注意区分两个概念：
    1. regulation_type 保持标准生物学含义：
       - "activation" = TF 正向调控目标基因表达
       - "repression" = TF 负向调控目标基因表达
    2. effect_strength 不是“激活强度”分数，而是 TF 敲除后的剩余活性/表达比例。
       - activation 规则 + TF敲除 -> 失去正调控 -> 目标表达/容量下降 -> 可能收紧反应边界
       - repression 规则 + TF敲除 -> 去抑制 -> 当前实现中不额外收紧边界
    """
    tf_name: str                          # 转录因子名称
    target_genes: List[str]               # 被调控的基因列表
    target_reactions: List[str]           # 被调控的反应列表（BiGG ID）
    regulation_type: str                  # 调控方向: "activation"=正调控, "repression"=负调控

    # 新增：用于计算 effect_strength 的属性
    tf_category: TFCategory = TFCategory.UNKNOWN
    evidence_level: EvidenceLevel = EvidenceLevel.MODERATE
    metabolic_importance: MetabolicImportance = MetabolicImportance.PERIPHERAL
    is_direct_regulation: bool = True     # 是否直接调控

    # 可选：手动指定 effect_strength（覆盖自动计算）
    manual_effect_strength: Optional[float] = None

    notes: str = ""

    @property
    def effect_strength(self) -> float:
        """
        计算 TF 敲除后的剩余活性/表达比例。

        返回值范围: 0.0 - 1.0
        - 0.0: TF敲除后，目标酶/反应容量完全丧失
        - 1.0: TF敲除后，目标酶/反应容量不下降

        这里的 regulation_type 与 effect_strength 含义不同：
        - activation: 生物学上表示 TF 对目标是正调控；在敲除模拟里，数值越小表示失去该 TF 后剩余容量越低
        - repression: 生物学上表示 TF 对目标是负调控；在敲除模拟里通常视为去抑制，因此当前实现返回 1.0，且不进一步收紧边界
        """
        # 如果手动指定了，直接返回
        if self.manual_effect_strength is not None:
            return self.manual_effect_strength

        # 对于 repression 类型，生物学上是负调控；在 TF 敲除模拟里视为去抑制，因此保持 1.0
        if self.regulation_type == "repression":
            return 1.0

        # 对于 activation 类型，计算 TF 敲除后的剩余活性/表达比例
        return self._calculate_activation_effect()

    def _calculate_activation_effect(self) -> float:
        """计算激活型调控的效应强度"""

        # 1. 基础效应强度（基于TF类别）
        base_effect = {
            TFCategory.GLOBAL_REGULATOR: 0.15,    # 全局调控因子影响大
            TFCategory.METABOLIC_REGULATOR: 0.20, # 代谢调控因子影响较大
            TFCategory.STRESS_RESPONSE: 0.30,     # 应激响应通常条件依赖
            TFCategory.SIGMA_FACTOR: 0.25,        # Sigma因子影响较大
            TFCategory.LOCAL_REGULATOR: 0.40,     # 局部调控因子影响较小
            TFCategory.UNKNOWN: 0.35,             # 未知类别使用保守估计
        }.get(self.tf_category, 0.35)

        # 2. 证据强度修正
        evidence_modifier = {
            EvidenceLevel.STRONG: 1.0,      # 强证据，不修正
            EvidenceLevel.MODERATE: 1.2,    # 中等证据，效应可能被高估
            EvidenceLevel.WEAK: 1.5,        # 弱证据，更保守
            EvidenceLevel.INFERRED: 1.8,    # 推断的，最保守
        }.get(self.evidence_level, 1.2)

        # 3. 代谢重要性修正
        importance_modifier = {
            MetabolicImportance.CENTRAL: 0.8,      # 中心代谢，调控更严格
            MetabolicImportance.ENERGY: 0.85,      # 能量代谢
            MetabolicImportance.AMINO_ACID: 0.95,  # 氨基酸代谢
            MetabolicImportance.PERIPHERAL: 1.1,   # 外周代谢，调控较松
        }.get(self.metabolic_importance, 1.0)

        # 4. 直接/间接调控修正
        direct_modifier = 1.0 if self.is_direct_regulation else 1.3

        # 计算最终效应强度
        effect = base_effect * evidence_modifier * importance_modifier * direct_modifier

        # 限制在 [0.05, 0.95] 范围内
        return max(0.05, min(0.95, effect))


# =============================================================================
# TF 类别定义
# =============================================================================

TF_CATEGORIES: Dict[str, TFCategory] = {
    # 全局调控因子
    "Crp": TFCategory.GLOBAL_REGULATOR,
    "Fnr": TFCategory.GLOBAL_REGULATOR,
    "ArcA": TFCategory.GLOBAL_REGULATOR,
    "ArcB": TFCategory.GLOBAL_REGULATOR,
    "Fis": TFCategory.GLOBAL_REGULATOR,
    "Hns": TFCategory.GLOBAL_REGULATOR,
    "IHFA": TFCategory.GLOBAL_REGULATOR,
    "IHFB": TFCategory.GLOBAL_REGULATOR,
    "Lrp": TFCategory.GLOBAL_REGULATOR,

    # 代谢特异性调控因子
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

    # 应激响应调控因子
    "OxyR": TFCategory.STRESS_RESPONSE,
    "SoxR": TFCategory.STRESS_RESPONSE,
    "SoxS": TFCategory.STRESS_RESPONSE,
    "Fur": TFCategory.STRESS_RESPONSE,
    "GadE": TFCategory.STRESS_RESPONSE,
    "GadW": TFCategory.STRESS_RESPONSE,
    "GadX": TFCategory.STRESS_RESPONSE,
    "MarA": TFCategory.STRESS_RESPONSE,
    "MarR": TFCategory.STRESS_RESPONSE,

    # Sigma因子
    "RpoD": TFCategory.SIGMA_FACTOR,
    "RpoN": TFCategory.SIGMA_FACTOR,
    "RpoS": TFCategory.SIGMA_FACTOR,
    "FliA": TFCategory.SIGMA_FACTOR,
    "FecI": TFCategory.SIGMA_FACTOR,

    # 双组分系统响应调控因子
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

# 反应到代谢重要性的映射
REACTION_IMPORTANCE: Dict[str, MetabolicImportance] = {
    # 糖酵解
    "PGI": MetabolicImportance.CENTRAL,
    "PFK": MetabolicImportance.CENTRAL,
    "FBA": MetabolicImportance.CENTRAL,
    "TPI": MetabolicImportance.CENTRAL,
    "GAPD": MetabolicImportance.CENTRAL,
    "PGK": MetabolicImportance.CENTRAL,
    "PGM": MetabolicImportance.CENTRAL,
    "ENO": MetabolicImportance.CENTRAL,
    "PYK": MetabolicImportance.CENTRAL,

    # 糖异生
    "PPS": MetabolicImportance.CENTRAL,
    "PPCK": MetabolicImportance.CENTRAL,
    "FBP": MetabolicImportance.CENTRAL,

    # TCA循环
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

    # 乙醛酸循环
    "ICL": MetabolicImportance.CENTRAL,
    "MALS": MetabolicImportance.CENTRAL,

    # 磷酸戊糖途径
    "G6PDH2r": MetabolicImportance.CENTRAL,
    "PGL": MetabolicImportance.CENTRAL,
    "GND": MetabolicImportance.CENTRAL,
    "RPI": MetabolicImportance.CENTRAL,
    "RPE": MetabolicImportance.CENTRAL,
    "TKT1": MetabolicImportance.CENTRAL,
    "TKT2": MetabolicImportance.CENTRAL,
    "TALA": MetabolicImportance.CENTRAL,

    # ED途径
    "EDD": MetabolicImportance.CENTRAL,
    "EDA": MetabolicImportance.CENTRAL,

    # 呼吸链
    "NADH16pp": MetabolicImportance.ENERGY,
    "NADH17pp": MetabolicImportance.ENERGY,
    "CYTBO3_4pp": MetabolicImportance.ENERGY,
    "ATPS4rpp": MetabolicImportance.ENERGY,
    "FRD2": MetabolicImportance.ENERGY,
    "FRD3": MetabolicImportance.ENERGY,

    # 氨基酸代谢
    "GLNS": MetabolicImportance.AMINO_ACID,
    "GLUDy": MetabolicImportance.AMINO_ACID,
    "ACLS": MetabolicImportance.AMINO_ACID,
    "SERAT": MetabolicImportance.AMINO_ACID,
    "CYSS": MetabolicImportance.AMINO_ACID,
    "ARGSS": MetabolicImportance.AMINO_ACID,

    # 转运
    "GLCptspp": MetabolicImportance.CENTRAL,
    "ACS": MetabolicImportance.CENTRAL,
}


def get_tf_category(tf_name: str) -> TFCategory:
    """获取转录因子的类别"""
    return TF_CATEGORIES.get(tf_name, TFCategory.UNKNOWN)


def get_reaction_importance(reaction_id: str) -> MetabolicImportance:
    """获取反应的代谢重要性"""
    # 移除后缀
    base_id = reaction_id.replace('_reverse', '').split('_num')[0]
    return REACTION_IMPORTANCE.get(base_id, MetabolicImportance.PERIPHERAL)


# =============================================================================
# 调控规则定义 - V2版本
# =============================================================================

def create_regulatory_rules_v2() -> Dict[str, List[RegulatoryRuleV2]]:
    """
    创建调控规则（V2版本）

    effect_strength 将基于 TF 类别、证据强度、代谢重要性自动计算
    """
    rules = {}

    # =========================================================================
    # Cra (FruR) - 碳代谢抑制蛋白
    # =========================================================================
    rules["Cra"] = [
        # 激活的反应
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["ppsA"],
            target_reactions=["PPS"],
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra激活ppsA表达，参与糖异生"
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
            notes="Cra激活pckA表达，参与糖异生"
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
            notes="Cra激活fbp表达，参与糖异生"
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
            notes="Cra激活乙醛酸循环"
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
            notes="Cra激活icd表达"
        ),
        # 抑制的反应
        RegulatoryRuleV2(
            tf_name="Cra",
            target_genes=["pykF", "pykA"],
            target_reactions=["PYK"],
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="Cra抑制丙酮酸激酶"
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
            notes="Cra抑制磷酸果糖激酶"
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
            notes="Cra抑制ED途径"
        ),
    ]

    # =========================================================================
    # Crp - cAMP受体蛋白 (全局调控因子)
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
            notes="Crp激活乙醛酸循环"
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
            notes="Crp激活SDH复合物"
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
            notes="Crp激活acs表达"
        ),
    ]

    # =========================================================================
    # ArcA/ArcB - 双组分厌氧调控系统
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
            notes="ArcA在厌氧条件下抑制SDH"
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
            notes="ArcA在厌氧条件下抑制细胞色素氧化酶"
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
            notes="ArcA抑制TCA循环酶"
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
            notes="ArcA抑制PDH复合物"
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
            notes="ArcA抑制柠檬酸合酶"
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
            is_direct_regulation=False,  # 通过ArcA间接调控
            notes="ArcB-ArcA系统调控SDH"
        ),
    ]

    # =========================================================================
    # Fnr - 厌氧调控因子
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
            notes="Fnr调控呼吸链"
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
            notes="Fnr抑制好氧呼吸"
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
            notes="Fnr激活厌氧呼吸酶"
        ),
    ]

    # =========================================================================
    # PdhR - 丙酮酸脱氢酶调控因子
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
            notes="PdhR抑制PDH复合物表达"
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
            notes="PdhR抑制NADH脱氢酶"
        ),
    ]

    # =========================================================================
    # IclR - 乙醛酸循环调控因子
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
            notes="IclR抑制乙醛酸循环"
        ),
    ]

    # =========================================================================
    # FadR - 脂肪酸代谢调控因子
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
            notes="FadR抑制脂肪酸降解"
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
            notes="FadR激活脂肪酸合成"
        ),
    ]

    # =========================================================================
    # Fur - 铁代谢调控因子
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
            is_direct_regulation=False,  # 间接通过铁可用性
            notes="Fur影响含铁蛋白表达"
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
            notes="Fur影响含铁蛋白表达"
        ),
    ]

    # =========================================================================
    # NarL/NarP - 硝酸盐呼吸调控因子
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
            notes="NarL激活硝酸盐呼吸"
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
            notes="NarL抑制延胡索酸呼吸"
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
            notes="NarP激活硝酸盐呼吸"
        ),
    ]

    # =========================================================================
    # GadE/GadW/GadX - 酸抗性调控因子
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
                notes=f"{tf}激活酸抗性系统"
            ),
        ]

    # =========================================================================
    # Lrp - 亮氨酸响应调控蛋白
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
            notes="Lrp激活支链氨基酸合成"
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
            notes="Lrp激活氮代谢"
        ),
    ]

    # =========================================================================
    # SoxR/SoxS - 氧化应激响应
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
            notes="SoxS激活抗氧化酶"
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
            notes="SoxS激活NADPH产生酶"
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
            notes="SoxR感应氧化应激，激活SoxS"
        ),
    ]

    # =========================================================================
    # OxyR - 过氧化物应激响应
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
            notes="OxyR激活抗氧化应激酶"
        ),
    ]

    # =========================================================================
    # CysB - 半胱氨酸代谢调控
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
            notes="CysB激活半胱氨酸合成"
        ),
    ]

    # =========================================================================
    # GlpR - 甘油代谢调控
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
            notes="GlpR抑制甘油利用"
        ),
    ]

    # =========================================================================
    # Mlc - PTS系统调控
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
            notes="Mlc抑制PTS系统"
        ),
    ]

    # =========================================================================
    # PurR - 嘌呤代谢调控
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
            notes="PurR抑制嘌呤从头合成"
        ),
    ]

    # =========================================================================
    # NtrC - 氮代谢调控
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
            notes="NtrC激活氮代谢"
        ),
    ]

    # =========================================================================
    # GcvA - 甘氨酸裂解系统调控
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
            notes="GcvA激活甘氨酸裂解系统"
        ),
    ]

    # =========================================================================
    # Sigma因子
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
            notes="RpoN(sigma54)调控氮代谢"
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
            notes="RpoS(sigma38)调控稳定期响应"
        ),
    ]
    
    # ArgR - 精氨酸抑制子
    rules["ArgR"] = [
        RegulatoryRuleV2(
            tf_name="ArgR",
            target_genes=["argA", "argB", "argC", "argD"],
            target_reactions=["ARGSL", "AGPR", "ACODA"], # 精氨酸合成上游
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="ArgR抑制精氨酸合成操纵子"
        ),
        RegulatoryRuleV2(
            tf_name="ArgR",
            target_genes=["argF", "argI"],
            target_reactions=["OCBT"], # 鸟氨酸氨甲酰转移酶
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True
        )
    ]

    # TrpR - 色氨酸抑制子
    rules["TrpR"] = [
        RegulatoryRuleV2(
            tf_name="TrpR",
            target_genes=["trpE", "trpD"],
            target_reactions=["ANS", "ANPRT"], # 邻氨基苯甲酸合酶 (关键步)
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="TrpR抑制色氨酸合成"
        )
    ]

    # TyrR - 芳香族氨基酸调控
    rules["TyrR"] = [
        RegulatoryRuleV2(
            tf_name="TyrR",
            target_genes=["aroF", "tyrA"],
            target_reactions=["DDPA", "PPND"], # 芳香族合成通用步骤
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="TyrR主要抑制酪氨酸和苯丙氨酸合成"
        ),
        # TyrR 对 mtr (色氨酸透过酶) 是激活作用
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

    # MetJ - 甲硫氨酸抑制子
    rules["MetJ"] = [
        RegulatoryRuleV2(
            tf_name="MetJ",
            target_genes=["metA", "metB", "metC", "metE"],
            target_reactions=["HSST", "CYSS", "HCYS"], # 甲硫氨酸合成路径
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="MetJ抑制甲硫氨酸合成"
        )
    ]

    # =========================================================================
    # 补全：次级碳源与转运调控 (原本在 default_tfs 中)
    # =========================================================================

    # NagC - N-乙酰葡萄糖胺抑制子
    rules["NagC"] = [
        RegulatoryRuleV2(
            tf_name="NagC",
            target_genes=["nagE", "nagB", "nagA"],
            target_reactions=["NAGt2pp", "ACGAMPM", "ACGAM6P"], # NAG利用
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL, # 它是糖源
            is_direct_regulation=True,
            notes="NagC抑制N-乙酰葡萄糖胺的利用"
        )
    ]

    # DcuR - C4-二羧酸转运 (延胡索酸/琥珀酸)
    rules["DcuR"] = [
        RegulatoryRuleV2(
            tf_name="DcuR",
            target_genes=["dctA"],
            target_reactions=["SUCCt2_2pp", "FUMt2_2pp"], # 琥珀酸/延胡索酸转运
            regulation_type="activation",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="DcuR在厌氧条件下激活C4-二羧酸转运"
        )
    ]

    # BetI - 胆碱/甜菜碱调控
    rules["BetI"] = [
        RegulatoryRuleV2(
            tf_name="BetI",
            target_genes=["betA", "betB"],
            target_reactions=["CHOLD", "BETALDHx"], # 胆碱氧化
            regulation_type="repression",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="BetI抑制胆碱向甜菜碱的转化"
        )
    ]

    # PhoB - 磷酸盐饥饿响应
    rules["PhoB"] = [
        RegulatoryRuleV2(
            tf_name="PhoB",
            target_genes=["pstS", "pstC", "pstA", "pstB"],
            target_reactions=["PIt2rpp"], # 无机磷酸转运 (近似)
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="PhoB在缺磷时激活高亲和力磷酸转运"
        )
    ]

    # =========================================================================
    # 补全：厌氧与特殊代谢 (原本在 default_tfs 中)
    # =========================================================================

    # FhlA - 甲酸氢解酶调控
    rules["FhlA"] = [
        RegulatoryRuleV2(
            tf_name="FhlA",
            target_genes=["fdhF", "hycE"],
            target_reactions=["FHL"], # 甲酸氢解酶 (Formate -> CO2 + H2)
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="FhlA激活甲酸分解产氢"
        )
    ]

    # AppY - 酸性磷酸酶/厌氧调控
    rules["AppY"] = [
        RegulatoryRuleV2(
            tf_name="AppY",
            target_genes=["hyaA", "hyaB"],
            target_reactions=["HYD1pp"], # 氢化酶1
            regulation_type="activation",
            tf_category=TFCategory.LOCAL_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.ENERGY,
            is_direct_regulation=True,
            notes="AppY激活厌氧氢化酶"
        )
    ]

    # FecI - 柠檬酸铁转运
    rules["FecI"] = [
        RegulatoryRuleV2(
            tf_name="FecI",
            target_genes=["fecA", "fecB"],
            target_reactions=["FECR_1", "FECR_2"], # 柠檬酸铁还原/转运 (需核对模型ID)
            regulation_type="activation",
            tf_category=TFCategory.SIGMA_FACTOR, # 它是ECF Sigma因子
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="FecI激活外源柠檬酸铁的转运"
        )
    ]


    # MarA - 多重抗生素抗性调控子
    # MarA 很有趣，它为了抗氧化/抗药，会激活 PPP 途径产生 NADPH，同时关闭孔蛋白
    rules["MarA"] = [
        RegulatoryRuleV2(
            tf_name="MarA",
            target_genes=["zwf"], 
            target_reactions=["G6PDH2r"], # 激活G6P脱氢酶，增加NADPH供应
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="MarA激活Zwf以提供NADPH用于抗氧化/解毒"
        ),
        RegulatoryRuleV2(
            tf_name="MarA",
            target_genes=["ompF"],
            target_reactions=["OMP_DIFFUSION"], # 假设模型中有外膜扩散反应 (如 OmpF 介导的运输)
            # 注意：如果模型没有显式的孔蛋白反应，这条规则可能无效，但在概念上很重要
            regulation_type="repression",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="MarA下调OmpF以减少抗生素内流"
        )
    ]
    # Rob 的功能与 MarA 高度重叠
    rules["Rob"] = rules["MarA"]

    # KdgR - 糖醛酸/果胶代谢 (ED途径分支)
    rules["KdgR"] = [
        RegulatoryRuleV2(
            tf_name="KdgR",
            target_genes=["eda", "edd"],
            target_reactions=["EDA", "EDD"], # ED途径关键酶
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="KdgR抑制ED途径(与Cra协同/竞争)"
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

    # CytR - 核苷分解代谢
    # 当葡萄糖存在时，CytR 抑制核苷的利用
    rules["CytR"] = [
        RegulatoryRuleV2(
            tf_name="CytR",
            target_genes=["udp"],
            target_reactions=["URIDK2r"], # 尿苷磷酸化酶
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="CytR抑制核苷分解"
        ),
        RegulatoryRuleV2(
            tf_name="CytR",
            target_genes=["cdd"],
            target_reactions=["CYTD"], # 胞苷脱氨酶
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True
        )
    ]

    # PrpR - 丙酸代谢 (甲基柠檬酸循环)
    rules["PrpR"] = [
        RegulatoryRuleV2(
            tf_name="PrpR",
            target_genes=["prpC", "prpB"],
            target_reactions=["MCITL2", "MCITS"], # 甲基柠檬酸合酶/裂解酶
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL, # 连接TCA
            is_direct_regulation=True,
            notes="PrpR激活丙酸利用途径"
        )
    ]

    # LldR - L-乳酸利用
    rules["LldR"] = [
        RegulatoryRuleV2(
            tf_name="LldR",
            target_genes=["lldD"],
            target_reactions=["L_LACD2"], # L-lactate dehydrogenase (quinone)
            regulation_type="repression", 
            # 注：LldR本身是阻遏蛋白，乳酸存在时解除阻遏。
            # 这里定义为 repressor 意味着：敲除 LldR -> 去抑制 -> 乳酸利用能力增强
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.CENTRAL,
            is_direct_regulation=True,
            notes="LldR抑制乳酸利用，敲除后组成型表达"
        )
    ]

    # GlcC - 乙醇酸利用
    rules["GlcC"] = [
        RegulatoryRuleV2(
            tf_name="GlcC",
            target_genes=["glcD", "glcE", "glcF"],
            target_reactions=["GLYCTO2", "GLYCTO4"], # Glycolate oxidase
            regulation_type="activation", # GlcC是激活蛋白
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="GlcC激活乙醇酸氧化"
        )
    ]

    # TdcA / TdcR - 厌氧苏氨酸代谢
    # 这对厌氧发酵非常重要，产生丙酸
    tdc_rule = [
        RegulatoryRuleV2(
            tf_name="TdcA", # TdcA是主激活子
            target_genes=["tdcB"],
            target_reactions=["THRD_L"], # Threonine deaminase (catabolic)
            regulation_type="activation",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.AMINO_ACID,
            is_direct_regulation=True,
            notes="TdcA激活厌氧苏氨酸降解"
        )
    ]
    rules["TdcA"] = tdc_rule
    rules["TdcR"] = tdc_rule # TdcR 协同作用
    
    # TorR - 氧化三甲胺(TMAO)呼吸
    # 在厌氧条件下，如果有TMAO，TorR激活TMAO还原酶系统
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
            notes="TorR激活TMAO作为电子受体的厌氧呼吸"
        )
    ]

    # ModE - 钼酸盐转运与代谢
    # 钼是硝酸盐还原酶(Nar)和甲酸脱氢酶(Fdh)的必需辅因子
    rules["ModE"] = [
        RegulatoryRuleV2(
            tf_name="ModE",
            target_genes=["modA", "modB", "modC"],
            target_reactions=["MOBd"], # Molybdate transport
            regulation_type="repression", 
            # ModE在钼浓度高时抑制转运（节约能量），在低浓度时解除抑制
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="ModE调控钼酸盐摄取，影响无氧呼吸酶活性"
        )
    ]

    # UxuR - 己糖醛酸代谢 (Hexuronate)
    # 控制葡萄糖醛酸(Glucuronate)和半乳糖醛酸(Galacturonate)的利用
    rules["UxuR"] = [
        RegulatoryRuleV2(
            tf_name="UxuR",
            target_genes=["uxuA", "uxuB"],
            target_reactions=["MAN6PI", "ALTRH"], # 异构酶路径 (BiGG ID需根据具体路径核对)
            regulation_type="repression",
            tf_category=TFCategory.METABOLIC_REGULATOR,
            evidence_level=EvidenceLevel.STRONG,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="UxuR抑制己糖醛酸利用路径"
        )
    ]

    # IdnR - L-艾杜糖酸代谢 (L-Idonate)
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
            notes="IdnR激活L-艾杜糖酸利用"
        )
    ]

    # RcsB - 荚膜多糖合成 (Colanic Acid)
    # 这是一个消耗大量能量和代谢前体（如Fucose）的过程，通常在应激下激活
    rules["RcsB"] = [
        RegulatoryRuleV2(
            tf_name="RcsB",
            target_genes=["wcaA", "wcaB", "wcaC"], # colanic acid synthesis operon
            target_reactions=["UAGDP", "UAGPT3"], # UDP-sugar metabolic pathways
            # 如果模型有一个总的生物量或荚膜反应，用那个最好，例如 "COLANIC_ACID_SINK"
            regulation_type="activation",
            tf_category=TFCategory.STRESS_RESPONSE,
            evidence_level=EvidenceLevel.MODERATE,
            metabolic_importance=MetabolicImportance.PERIPHERAL,
            is_direct_regulation=True,
            notes="RcsB激活荚膜多糖合成，竞争细胞壁前体"
        )
    ]

    # =========================================================================
    # 添加其他没有明确代谢反应的TF（使用默认值）
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
                    notes=f"{tf}对中心代谢影响小或未知"
                ),
            ]

    return rules


# 创建全局规则字典
REGULATORY_RULES_V2 = create_regulatory_rules_v2()


# =============================================================================
# 功能酶对照（直接敲除代谢酶）
# =============================================================================

ENZYME_KNOCKOUTS: Dict[str, List[str]] = {
    "Pgi": ["PGI"],
    "Zwf": ["G6PDH2r"],
    "SdhC": ["SUCDi"],
}


# =============================================================================
# API 函数
# =============================================================================

def get_regulatory_effects(tf_name: str) -> List[RegulatoryRuleV2]:
    """获取转录因子的调控效应"""
    tf_name_normalized = tf_name.strip()

    for key in REGULATORY_RULES_V2.keys():
        if key.lower() == tf_name_normalized.lower():
            return REGULATORY_RULES_V2[key]

    return []


def get_enzyme_knockout_reactions(enzyme_name: str) -> List[str]:
    """获取功能酶敲除对应的反应"""
    for key in ENZYME_KNOCKOUTS.keys():
        if key.lower() == enzyme_name.lower():
            return ENZYME_KNOCKOUTS[key]
    return []


def is_transcription_factor(gene_name: str) -> bool:
    """判断基因是否是转录因子"""
    gene_name_lower = gene_name.lower()
    for key in REGULATORY_RULES_V2.keys():
        if key.lower() == gene_name_lower:
            return True
    return False


def is_metabolic_enzyme(gene_name: str) -> bool:
    """判断基因是否是代谢酶"""
    gene_name_lower = gene_name.lower()
    for key in ENZYME_KNOCKOUTS.keys():
        if key.lower() == gene_name_lower:
            return True
    return False


def get_affected_reactions_and_effects(
    tf_name: str,
    effect_type: str = "activation"
) -> Dict[str, float]:
    """获取TF敲除后会下调容量的反应及其剩余活性比例。"""
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
    """打印所有调控规则的摘要，并明确区分调控方向与敲除后的容量变化。"""
    print("=" * 80)
    print("转录因子调控规则摘要 (V2版本 - 自动计算effect_strength)")
    print("  说明: activation/repression 保持标准生物学含义；effect_strength 表示 TF 敲除后的剩余活性比例")
    print("=" * 80)

    for tf_name, rules in REGULATORY_RULES_V2.items():
        activation_rules = [r for r in rules if r.regulation_type == "activation" and r.target_reactions]
        repression_rules = [r for r in rules if r.regulation_type == "repression" and r.target_reactions]

        if not activation_rules and not repression_rules:
            continue

        print(f"\n{tf_name} ({get_tf_category(tf_name).value}):")

        if activation_rules:
            print("  正调控的反应 (activation; TF敲除后这些目标的表达/容量会下降):")
            for rule in activation_rules:
                reactions_str = ", ".join(rule.target_reactions)
                print(f"    - {reactions_str}")
                print(f"      effect_strength: {rule.effect_strength:.2f}")
                print(f"      (类别: {rule.tf_category.value}, "
                      f"证据: {rule.evidence_level.value}, "
                      f"重要性: {rule.metabolic_importance.value})")

        if repression_rules:
            print("  负调控的反应 (repression; TF敲除后这些目标去抑制，当前实现不额外收紧边界):")
            for rule in repression_rules:
                reactions_str = ", ".join(rule.target_reactions)
                print(f"    - {reactions_str} (effect_strength: {rule.effect_strength:.2f})")

    print("\n" + "=" * 80)
    print("功能酶对照:")
    for enzyme, reactions in ENZYME_KNOCKOUTS.items():
        print(f"  {enzyme}: {', '.join(reactions)}")
    print("=" * 80)


def print_effect_strength_calculation_example():
    """打印effect_strength计算示例"""
    print("\n" + "=" * 80)
    print("effect_strength 计算示例")
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
                print(f"  TF类别: {rule.tf_category.value}")
                print(f"  证据强度: {rule.evidence_level.value}")
                print(f"  代谢重要性: {rule.metabolic_importance.value}")
                print(f"  直接调控: {rule.is_direct_regulation}")
                print(f"  计算的 effect_strength: {rule.effect_strength:.3f}")
                break


if __name__ == "__main__":
    print_regulatory_summary()
    print_effect_strength_calculation_example()

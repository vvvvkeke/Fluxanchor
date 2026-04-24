#!/usr/bin/env python
"""
分析FluxGen、KinLLM和FBA方法预测的flux ratio与真实值的差异

本脚本：
1. 加载三种方法的模型（FluxGen, KinLLM, fba）
2. 对每个菌株进行基因敲除和FBA模拟
3. 根据flux_ratio_constrain.csv定义计算预测的flux ratio
4. 与真实的flux ratio（flux_data.csv）比较
5. 计算累计作商（ratio of ratios）和累计作差（difference）两种评估指标

输出:
- 详细结果CSV文件（包含每个菌株每个pathway的预测值、真实值、误差）
- 汇总结果CSV文件（包含每种方法的平均性能指标）
"""

import sys
sys.path.append(r'./')
sys.path.append(r'../')
sys.path.append(r'./script/')
sys.path.append(r'../script/')

import pandas as pd
import numpy as np
import cobra
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from script.ECMpy_function import get_enzyme_constraint_model

# ================================================================================
# 辅助函数：基因映射和基因敲除
# ================================================================================


def aggregate_isoenzyme_fluxes(flux_dict: Dict[str, float]) -> Dict[str, float]:
    """
    合并同工酶反应的流量，保留反应方向信息

    例如: FBA_num1, FBA_num2, FBA_num3 -> FBA
          PGK_num1, PGK_num2 -> PGK
          PGK_reverse_num1, PGK_reverse_num2 -> PGK_reverse
          MDH, MDH2, MDH3 -> MDH

    参数:
    flux_dict: 原始流量字典 {reaction_id: flux_value}

    返回:
    aggregated_dict: 合并后的流量字典 {reaction_name: flux_value}
    """
    aggregated = defaultdict(list)

    for rxn_id, flux_value in flux_dict.items():
        # 提取反应的基础名称（只去掉同工酶编号，保留 _reverse 标记）
        base_name = rxn_id

        # 去掉 _num1, _num2, _num3 等后缀（同工酶）
        base_name = re.sub(r'_num\d+$', '', base_name)

        # 特殊处理：合并 MDH2, MDH3 -> MDH
        if re.match(r'^MDH\d$', base_name):
            base_name = 'MDH'
        # 特殊处理：合并 MDH2_reverse, MDH3_reverse -> MDH_reverse
        elif re.match(r'^MDH\d_reverse$', base_name):
            base_name = 'MDH_reverse'

        # 收集所有变体的流量（取绝对值）
        aggregated[base_name].append(abs(flux_value))

    # 对每个反应取所有同工酶的和
    result = {}
    for base_name, flux_values in aggregated.items():
        result[base_name] = sum(flux_values)

    return result


# ================================================================================
# Flux Ratio计算函数
# ================================================================================

def find_reactions_producing_metabolite(model, metabolite_id: str) -> List[Tuple[str, float]]:
    """
    查找代谢网络模型中所有生成指定代谢物的反应

    参数:
    model: cobra模型
    metabolite_id: 代谢物ID

    返回:
    List[Tuple[reaction_id, coefficient]]: 生成该代谢物的反应列表
    """
    producing_reactions = []

    # 查找代谢物
    metabolite = None
    try:
        metabolite = model.metabolites.get_by_id(metabolite_id)
    except KeyError:
        return producing_reactions

    # 遍历所有涉及该代谢物的反应
    for reaction in metabolite.reactions:
        # 获取反应中该代谢物的化学计量系数
        stoich_coeff = reaction.metabolites[metabolite]

        # 如果系数为正，说明该反应生��该代谢物
        if stoich_coeff > 0:
            producing_reactions.append((reaction.id, stoich_coeff))

    return producing_reactions


def identify_product_metabolite(model, numerator_reaction_ids: List[str], pathway_name: str = None) -> str:
    """
    识别numerator反应生成的关键代谢物

    参数:
    model: cobra模型
    numerator_reaction_ids: numerator反应ID列表
    pathway_name: pathway名称（用于特殊情况识别）

    返回:
    metabolite_id: 关键代谢物ID，如果无法识别则返回None
    """
    # 定义反应模式与代谢物的映射关系
    reaction_patterns = {
        'PYK': 'pyr_c',           # 丙酮酸激酶 -> 丙���酸
        'PPCK': 'pep_c',          # PEP羧激酶 -> PEP
        'PEPCK': 'oaa_c',         # PEP羧激酶 -> 草酰乙酸
        'PC': 'oaa_c',            # 丙酮酸羧化酶 -> 草酰乙酸
        'GHMT2r': 'ser__L_c',     # 甘氨酸羟甲基转移酶 -> 丝氨酸
        'SHMT': 'ser__L_c',       # 丝氨酸羟甲基转移酶 -> 丝氨酸
        'FBA': 'g3p_c',           # 果糖二磷酸醛缩酶 -> 甘油醛-3-磷酸
        'EDA': 'pyr_c',           # ED途径 -> 丙酮酸
        'TKT': 'g3p_c',           # 转酮醇酶 -> 甘油醛-3-磷酸
        'TALA': 'g3p_c',          # 转醛醇酶 -> 甘油醛-3-磷酸
        'MDH': 'oaa_c',           # 苹果酸脱氢酶 -> 草酰乙酸
        'ME': 'pyr_c',            # 苹果酸酶 -> 丙酮酸
    }

    # 首先检查pathway名称中的特殊模式
    if pathway_name:
        pathway_lower = str(pathway_name).lower()
        if "glycolysis" in pathway_lower or "serine through glycolysis" in pathway_lower:
            if 'g3p_c' in model.metabolites:
                return 'g3p_c'
        if "ed pathway" in pathway_lower:
            if 'pyr_c' in model.metabolites:
                return 'pyr_c'
        if "pp pathway" in pathway_lower:
            if 'pep_c' in model.metabolites:
                return 'pep_c'
        if "tca cycle" in pathway_lower:
            if 'oaa_c' in model.metabolites:
                return 'oaa_c'
        if "gluco-neogenesis" in pathway_lower:
            if "pep from" in pathway_lower:
                if 'pep_c' in model.metabolites:
                    return 'pep_c'
            if "pyr from" in pathway_lower:
                if 'pyr_c' in model.metabolites:
                    return 'pyr_c'

    # 遍历所有numerator反应，尝试匹配模式
    for rxn_id in numerator_reaction_ids:
        # 移除可能的_reverse后缀
        base_rxn_id = rxn_id.replace('_reverse', '').replace('_num1', '').replace('_num2', '').replace('_num3', '')

        # 检查是否匹配已知模式
        for pattern, metabolite_id in reaction_patterns.items():
            if pattern in base_rxn_id.upper():
                # 验证该代谢物在模型中存在
                if metabolite_id in model.metabolites:
                    return metabolite_id

    return None


def calculate_flux_ratio(model, flux_dict: Dict[str, float],
                         numerator_reactions: List[str],
                         pathway_name: str = None) -> float:
    """
    根据flux_ratio_constrain.csv的定义计算flux ratio

    公式: ratio = sum(numerator) / sum(all_producing_reactions)

    参数:
    model: cobra模型（用于查找生成代谢物的所有反应）
    flux_dict: 流量字典 {reaction_id: flux_value}，键是带有后缀的反应id（例如FBA_num1）
    numerator_reactions: 分子反应列表（来自flux_ratio_constrain.csv，已经包含后缀）
    pathway_name: pathway名称（用于识别目标代谢物）

    返回:
    flux_ratio: 计算的flux ratio值
    """
    # 不进行同工酶合并，直接使用原始flux_dict

    # 计算numerator的总流量
    numerator_flux = 0.0
    aggregate_isoenzyme_fluxes(numerator_reactions)
    for rxn_id in numerator_reactions:
        rxn_id = rxn_id.strip()
        # 直接匹配带后缀的反应id
        if rxn_id in flux_dict:
            numerator_flux += abs(flux_dict[rxn_id])

    # 识别目标代谢物
    metabolite_id = identify_product_metabolite(model, numerator_reactions, pathway_name)

    if metabolite_id is None:
        # 如果无法识别目标代谢物，返回NaN
        return np.nan

    # 查找所有生成该代谢物的反应
    producing_reactions = find_reactions_producing_metabolite(model, metabolite_id)

    if not producing_reactions:
        return np.nan

    # 计算所有生成反应的总流量
    # 注意：producing_reactions中的反应id是模型中的原始id，可能带有_num后缀
    total_flux = 0.0

    for rxn_id, stoich in producing_reactions:
        # 直接使用反应id（包括后缀）从flux_dict中获取流量
        if rxn_id in flux_dict:
            total_flux += abs(flux_dict[rxn_id])

    # 计算flux ratio
    if total_flux > 1e-10:
        return numerator_flux / total_flux
    else:
        return np.nan


def normalize_reaction_name(reaction_name: str) -> str:
    """
    标准化反应名称，去掉同工酶后缀

    例如:
    - FBA_num1 -> FBA
    - FBA_num2 -> FBA
    - MDH2 -> MDH
    - MDH3 -> MDH
    - PGI_reverse_num1 -> PGI_reverse

    参数:
    reaction_name: 原始反应名称

    返回:
    标准化后的反应名称
    """
    base_name = reaction_name

    # 去掉 _num1, _num2, _num3 等后缀（同工酶）
    base_name = re.sub(r'_num\d+$', '', base_name)

    # 特殊处理：合并 MDH2, MDH3 -> MDH
    if re.match(r'^MDH\d$', base_name):
        base_name = 'MDH'
    # 特殊处理：合并 MDH2_reverse, MDH3_reverse -> MDH_reverse
    elif re.match(r'^MDH\d_reverse$', base_name):
        base_name = 'MDH_reverse'

    return base_name


def calculate_flux_ratios_from_detailed(model: cobra.Model,
                                        detailed_row: pd.Series,
                                        flux_ratio_constraint_file: str) -> Dict[str, float]:
    """
    从detailed.csv中的一行数据计算所有flux ratio值

    参数:
    model: cobra模型（用于识别生成代谢物的反应）
    detailed_row: detailed.csv中的一行数据（包含所有反应的预测值）
    flux_ratio_constraint_file: flux ratio约束文件路径

    返回:
    flux_ratios: {pathway_name: ratio_value}
    """
    try:
        # 从detailed_row中提取所有反应的预测值
        # 列名格式为：<reaction_name>_pred
        # 注意：detailed.csv中的反应名称已经合并了同工酶后缀
        flux_dict_merged = {}
        for col in detailed_row.index:
            if col.endswith('_pred'):
                reaction_name = col[:-5]  # 去掉'_pred'后缀
                flux_value = detailed_row[col]
                if not pd.isna(flux_value):
                    flux_dict_merged[reaction_name] = float(flux_value)

        # 读取flux ratio约束定义
        constraint_df = pd.read_csv(flux_ratio_constraint_file, index_col=0)

        # 计算每个pathway的flux ratio
        flux_ratios = {}
        for pathway_name in constraint_df.columns:
            numerator_str = constraint_df.loc['reaction', pathway_name]

            if pd.isna(numerator_str):
                continue

            # 解析numerator反应列表（这些反应id可能包含后缀，如FBA_num1）
            numerator_reactions_raw = [r.strip() for r in str(numerator_str).split(',') if r.strip()]

            # 标准化反应名称以匹配detailed.csv中的格式
            # flux_ratio_constrain.csv: FBA_num1, FBA_num2 -> 标准化为 FBA
            # detailed.csv: FBA_pred -> 提取为 FBA
            normalized_reactions = set()
            for rxn in numerator_reactions_raw:
                normalized_name = normalize_reaction_name(rxn)
                normalized_reactions.add(normalized_name)

            # 计算numerator的总流量
            numerator_flux = 0.0
            for normalized_rxn in normalized_reactions:
                if normalized_rxn in flux_dict_merged:
                    numerator_flux += abs(flux_dict_merged[normalized_rxn])

            # 识别目标代谢物（使用原始反应名称）
            metabolite_id = identify_product_metabolite(model, numerator_reactions_raw, pathway_name)

            if metabolite_id is None:
                flux_ratios[pathway_name] = np.nan
                continue

            # 查找所有生成该代谢物的反应
            producing_reactions = find_reactions_producing_metabolite(model, metabolite_id)

            if not producing_reactions:
                flux_ratios[pathway_name] = np.nan
                continue

            # 计算所有生成反应的总流量
            # 注意：producing_reactions中的反应id是模型中的原始id
            # 需要标准化后与flux_dict_merged匹配
            total_flux = 0.0
            for rxn_id, stoich in producing_reactions:
                normalized_rxn_id = normalize_reaction_name(rxn_id)
                if normalized_rxn_id in flux_dict_merged:
                    total_flux += abs(flux_dict_merged[normalized_rxn_id])

            # 计算flux ratio
            if total_flux > 1e-10:
                flux_ratios[pathway_name] = numerator_flux / total_flux
            else:
                flux_ratios[pathway_name] = np.nan

        return flux_ratios

    except Exception as e:
        print(f"    错误: 计算flux ratio失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


# ================================================================================
# 性能评估函数
# ================================================================================

def calculate_performance_metrics(true_ratios: pd.Series,
                                  pred_ratios: pd.Series) -> Dict[str, float]:
    """
    计算性能指标

    两种评估方式：
    1. 累计作商（Cumulative Ratio）: sum(pred_i / true_i) / n
       - 衡量预测值与真实值的倍数关系
       - 理想值为1.0（完全匹配）

    2. 累计作差（Cumulative Difference）: sum(|pred_i - true_i|) / n
       - 衡量预测值与真实值的绝对差异
       - 理想值为0.0（无差异）

    参数:
    true_ratios: 真实flux ratio值（Series）
    pred_ratios: 预测flux ratio值（Series）

    返回:
    metrics: 性能指标字典
    """
    # 确保索引对齐
    common_pathways = true_ratios.index.intersection(pred_ratios.index)
    true_vals = true_ratios[common_pathways].values
    pred_vals = pred_ratios[common_pathways].values

    # 过滤掉NaN值
    valid_mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    true_vals = true_vals[valid_mask]
    pred_vals = pred_vals[valid_mask]

    if len(true_vals) == 0:
        return {
            'cumulative_ratio': np.nan,
            'cumulative_diff': np.nan,
            'mean_absolute_error': np.nan,
            'mean_relative_error': np.nan,
            'rmse': np.nan,
            'n_pathways': 0
        }

    # 累计作商（避免除以0）
    ratios = []
    for t, p in zip(true_vals, pred_vals):
        if abs(t) > 1e-10:  # 避免除以接近0的数
            ratios.append(p / t)
    cumulative_ratio = np.mean(ratios) if ratios else np.nan

    # 累计作差
    cumulative_diff = np.mean(np.abs(pred_vals - true_vals))

    # 平均绝对误差（MAE）
    mae = np.mean(np.abs(pred_vals - true_vals))

    # 平均相对误差（MRE）
    relative_errors = []
    for t, p in zip(true_vals, pred_vals):
        if abs(t) > 1e-10:
            relative_errors.append(abs(p - t) / abs(t))
    mre = np.mean(relative_errors) if relative_errors else np.nan

    # 均方根误差（RMSE）
    rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))

    return {
        'cumulative_ratio': cumulative_ratio,
        'cumulative_diff': cumulative_diff,
        'mean_absolute_error': mae,
        'mean_relative_error': mre,
        'rmse': rmse,
        'n_pathways': len(true_vals)
    }


# ================================================================================
# 主分析类
# ================================================================================

class FluxRatioAnalysis:
    """Flux Ratio性能分析框架"""

    def __init__(self, organism_config: Dict):
        """
        初始化分析框架

        参数:
        organism_config: 物种配置字典，包含模型路径、数据路径等信息
        """
        self.config = organism_config
        self.methods = ["FluxGen", "KinLLM", "fba"]

    def load_model(self, method: str) -> Optional[cobra.Model]:
        """加载指定方法的模型"""
        try:
            if method == "fba":
                model_file = f"{self.config['analysis_dir']}/get_kcat_mw_by_KinLLM/ecGEM/ecGEM_irr_enz_constraint.json"
                model = cobra.io.json.load_json_model(model_file)
            elif method == "FluxGen":
                model_file = f"{self.config['analysis_dir']}/get_kcat_mw_by_KinLLM/ecGEM/Bayesian/best_ecGEM.json"
                model = get_enzyme_constraint_model(model_file)
            else:  # KinLLM
                model_file = f"{self.config['analysis_dir']}/get_kcat_mw_by_{method}/ecGEM/ecGEM_irr_enz_constraint.json"
                model = get_enzyme_constraint_model(model_file)

            return model

        except Exception as e:
            print(f"  错误: 无法加载方法 {method} 的模型: {str(e)}")
            return None

    def analyze_method(self, method: str, detailed_csv_path: str) -> pd.DataFrame:
        """
        分析单个方法的flux ratio预测性能
        直接从detailed.csv文件读取预测结果

        参数:
        method: 方法名称（FluxGen, KinLLM, fba）
        detailed_csv_path: detailed.csv文件的路径

        返回:
        results_df: 结果DataFrame，包含每个菌株的预测值和真实值
        """
        print(f"\n{'='*80}")
        print(f"分析方法: {method}")
        print(f"{'='*80}")

        # 加载真实flux ratio数据
        flux_data_df = pd.read_csv(self.config['flux_data_file'], sep='\t')
        flux_data_df = flux_data_df.set_index('Id')

        print(f"加载详细结果文件: {detailed_csv_path}")
        detailed_df = pd.read_csv(detailed_csv_path)

        # 筛选当前方法的数据
        method_data = detailed_df[detailed_df['Method'] == method]
        print(f"找到 {len(method_data)} 条方法 {method} 的记录")

        # 加载模型（仅用于识别生成代谢物的反应）
        model = self.load_model(method)
        if model is None:
            print(f"  警告: 无法加载方法 {method} 的模型")
            return pd.DataFrame()

        # 存储结果
        results = []

        print(f"开始计算flux ratio...")

        for idx, row in method_data.iterrows():
            gene_id = row['Gene']

            if idx % 10 == 0:
                print(f"  进度: {idx+1}/{len(method_data)} ({(idx+1)/len(method_data)*100:.1f}%)")

            # 检查该基因在真实flux ratio数据中是否存在
            if gene_id not in flux_data_df.index:
                print(f"    警告: {gene_id} 在flux_data.csv中未找到，跳过")
                continue

            # 计算预测的flux ratios
            pred_ratios = calculate_flux_ratios_from_detailed(
                model, row,
                self.config['flux_ratio_constraint_file']
            )

            if not pred_ratios:
                print(f"    警告: {gene_id} 无法计算flux ratio，跳过")
                continue

            # 获取真实值
            true_ratios = flux_data_df.loc[gene_id]

            # 构建结果记录
            result = {'Strain': gene_id, 'Method': method}

            # 添加每个pathway的真实值和预测值
            for pathway_name in true_ratios.index:
                result[f'{pathway_name}_true'] = true_ratios[pathway_name]
                result[f'{pathway_name}_pred'] = pred_ratios.get(pathway_name, np.nan)

                # 计算单个pathway的误差
                true_val = true_ratios[pathway_name]
                pred_val = pred_ratios.get(pathway_name, np.nan)

                if not np.isnan(true_val) and not np.isnan(pred_val):
                    result[f'{pathway_name}_abs_error'] = abs(pred_val - true_val)
                    if abs(true_val) > 1e-10:
                        result[f'{pathway_name}_ratio'] = pred_val / true_val
                        result[f'{pathway_name}_rel_error'] = abs(pred_val - true_val) / abs(true_val)
                    else:
                        result[f'{pathway_name}_ratio'] = np.nan
                        result[f'{pathway_name}_rel_error'] = np.nan
                else:
                    result[f'{pathway_name}_abs_error'] = np.nan
                    result[f'{pathway_name}_ratio'] = np.nan
                    result[f'{pathway_name}_rel_error'] = np.nan

            # 计算该菌株的整体性能指标
            pred_series = pd.Series(pred_ratios)
            metrics = calculate_performance_metrics(true_ratios, pred_series)
            result.update(metrics)

            results.append(result)

        results_df = pd.DataFrame(results)

        print(f"\n方法 {method} 分析完成:")
        print(f"  成功分析菌株数: {len(results_df)}")

        if len(results_df) > 0:
            print(f"  平均累计作商: {results_df['cumulative_ratio'].mean():.4f}")
            print(f"  平均累计作差: {results_df['cumulative_diff'].mean():.4f}")
            print(f"  平均绝对误差: {results_df['mean_absolute_error'].mean():.4f}")
            print(f"  平均相对误差: {results_df['mean_relative_error'].mean():.4f}")
            print(f"  RMSE: {results_df['rmse'].mean():.4f}")

        return results_df

    def run_full_analysis(self, detailed_csv_path: str,
                          output_dir: str = "./results/flux_ratio_analysis"):
        """
        运行完整分析（所有方法）
        直接从detailed.csv文件读取预测结果

        参数:
        detailed_csv_path: detailed.csv文件的路径
        output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("Flux Ratio性能分析")
        print("="*80)
        print(f"物种: {self.config['name']}")
        print(f"方法: {', '.join(self.methods)}")
        print(f"详细结果文件: {detailed_csv_path}")
        print(f"输出目录: {output_dir}")
        print("="*80)

        all_results = {}

        for method in self.methods:
            results_df = self.analyze_method(method, detailed_csv_path)
            all_results[method] = results_df

            # 保存详细结果
            if len(results_df) > 0:
                detail_file = f"{output_dir}/{method}_flux_ratio_detailed.csv"
                results_df.to_csv(detail_file, index=False)
                print(f"  详细结果已保存到: {detail_file}")

        # 创建汇总报告
        self.create_summary_report(all_results, output_dir)

        return all_results

    def create_summary_report(self, all_results: Dict[str, pd.DataFrame], output_dir: str):
        """创建汇总报告"""
        print(f"\n{'='*80}")
        print("生成汇总报告")
        print(f"{'='*80}")

        summary_data = []

        for method, results_df in all_results.items():
            if len(results_df) == 0:
                continue

            summary = {
                'Method': method,
                'N_Strains': len(results_df),
                'Avg_Cumulative_Ratio': results_df['cumulative_ratio'].mean(),
                'Std_Cumulative_Ratio': results_df['cumulative_ratio'].std(),
                'Avg_Cumulative_Diff': results_df['cumulative_diff'].mean(),
                'Std_Cumulative_Diff': results_df['cumulative_diff'].std(),
                'Avg_MAE': results_df['mean_absolute_error'].mean(),
                'Avg_MRE': results_df['mean_relative_error'].mean(),
                'Avg_RMSE': results_df['rmse'].mean()
            }
            summary_data.append(summary)

        summary_df = pd.DataFrame(summary_data)

        # 保存整体汇总结果
        summary_file = f"{output_dir}/summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"整体汇总结果已保存到: {summary_file}")

        # ========================================================================
        # 新增：为每个pathway生成汇总报告
        # ========================================================================
        print(f"\n{'='*80}")
        print("生成各pathway的汇总报告")
        print(f"{'='*80}")

        pathway_summary_data = []

        # 获取所有pathway名称（从第一个方法的结果中提取）
        first_method_results = list(all_results.values())[0]
        if len(first_method_results) > 0:
            # 提取pathway名称（去掉_true/_pred等后缀）
            pathway_cols = [col for col in first_method_results.columns
                           if col.endswith('_true')]
            pathway_names = [col.replace('_true', '') for col in pathway_cols]

            print(f"发现 {len(pathway_names)} 个pathways:")
            for p in pathway_names:
                print(f"  - {p}")

            # 为每个方法和每个pathway计算平均指标
            for method, results_df in all_results.items():
                if len(results_df) == 0:
                    continue

                for pathway in pathway_names:
                    true_col = f'{pathway}_true'
                    pred_col = f'{pathway}_pred'
                    abs_error_col = f'{pathway}_abs_error'
                    ratio_col = f'{pathway}_ratio'
                    rel_error_col = f'{pathway}_rel_error'

                    # 确保列存在
                    if (true_col in results_df.columns and
                        pred_col in results_df.columns):

                        # 计算该pathway的平均指标
                        pathway_summary = {
                            'Method': method,
                            'Pathway': pathway,
                            'N_Strains': len(results_df),
                            'Avg_True_Value': results_df[true_col].mean(),
                            'Avg_Pred_Value': results_df[pred_col].mean(),
                            'Avg_Abs_Error': results_df[abs_error_col].mean(),
                            'Avg_Ratio': results_df[ratio_col].mean(),
                            'Avg_Rel_Error': results_df[rel_error_col].mean(),
                            'Std_Abs_Error': results_df[abs_error_col].std(),
                            'Std_Ratio': results_df[ratio_col].std()
                        }
                        pathway_summary_data.append(pathway_summary)

        if pathway_summary_data:
            pathway_summary_df = pd.DataFrame(pathway_summary_data)

            # 保存pathway级别的汇总结果
            pathway_summary_file = f"{output_dir}/pathway_summary.csv"
            pathway_summary_df.to_csv(pathway_summary_file, index=False)
            print(f"\nPathway级别汇总结果已保存到: {pathway_summary_file}")

            # 为每个pathway单独保存一个文件（方便查看）
            for pathway in pathway_names:
                pathway_data = pathway_summary_df[pathway_summary_df['Pathway'] == pathway]
                if len(pathway_data) > 0:
                    # 使用安全的文件名
                    safe_pathway_name = pathway.replace('/', '_').replace('(', '').replace(')', '').replace(',', '').replace(' ', '_')
                    pathway_file = f"{output_dir}/pathway_{safe_pathway_name}.csv"
                    pathway_data.to_csv(pathway_file, index=False)

            # 打印pathway级别的汇总表格（显示所有pathways）
            print(f"\n{'='*80}")
            print("Pathway级别性能汇总")
            print(f"{'='*80}")
            print(pathway_summary_df.to_string(index=False))
            print(f"{'='*80}")

        # 打印整体汇总表格
        print(f"\n{'='*80}")
        print("整体性能汇总")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))
        print(f"{'='*80}")

        # 找出最佳方法
        if len(summary_df) > 0:
            best_ratio_method = summary_df.loc[summary_df['Avg_Cumulative_Ratio'].sub(1.0).abs().idxmin(), 'Method']
            best_diff_method = summary_df.loc[summary_df['Avg_Cumulative_Diff'].idxmin(), 'Method']

            print(f"\n最佳方法:")
            print(f"  累计作商最接近1.0: {best_ratio_method}")
            print(f"  累计作差最小: {best_diff_method}")


# ================================================================================
# 主函数
# ================================================================================

def main():
    """主函数"""

    # 物种配置
    organism_config = {
        "name": "E.coli",
        "organism": "E. coli",
        "glucose_uptake": 10,
        "wild_type_id": "WILD TYPE",
        "biomass_reaction": "BIOMASS_Ec_iJO1366_core_53p95M",
        "vitro_exp": "/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/vitro_exp_data.csv",
        "flux_data_file": "/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/flux_data.csv",
        "flux_ratio_constraint_file": "/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/flux_ratio_constrain.csv",
        "gene_mapping_file": "/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/goal_blast.blast",
        "analysis_dir": "/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/",
        "network_file": "/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/network.xml"
    }

    # detailed.csv文件路径
    detailed_csv_path = "/home/zhangyangyu/kcat_km_predict/results_extend_reactions/iECDH1ME8569_1439/detailed.csv"

    # 创建分析器
    analyzer = FluxRatioAnalysis(organism_config)

    # 运行完整分析
    output_dir = "./results/flux_ratio_analysis"
    all_results = analyzer.run_full_analysis(detailed_csv_path, output_dir)

    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"所有结果已保存到: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


    #!/usr/bin/env python
    """
    可视化Flux Ratio误差对比
    参考 biological_analysis_supplement.py 的 2_pathway_error_comparison.pdf 风格
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # ============================================================================
    # 配置参数
    # ============================================================================

    # 设置字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    # 方法和配色（与biological_analysis_supplement.py保持一致）
    methods = ['FluxGen', 'KinLLM', 'fba']
    methods_color = ['#96CCEA', '#B2A3DD', '#ED949A']

    # Pathway配色（根据flux ratio数据调整）
    # pathway_colors_full = {
    #     'Glycolysis': '#e74c3c',
    #     'ED pathway': '#3498db',
    #     'PP pathway': '#2ecc71',
    #     'Gluconeogenesis': '#feca57',
    #     'TCA': '#f39c12',
    #     'Other': '#95a5a6'
    # }
    pathway_colors_full = {
        'Glycolysis': "#2E9914",  'PP': "#db349b", 'ED': "#F2B342", 'TCA': '#925A44',
        'Gluconeogenesis': '#CD61D7', 'Amino acid': '#f39c12', 
        'Fatty acid': '#9b59b6', 
        'Other': '#95a5a6',
        # 'Nucleotide': '#ee5a6f', 'Fermentation': '#48dbfb', 'Respiration': "#7D1B88",
        # 'Exchange': '#1abc9c', 'Energy': '#e67e22'
    }


    # 输入输出路径
    results_dir = './results/flux_ratio_analysis'
    output_dir = './analysis/flux_ratio_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    print("="*100)
    print("Flux Ratio误差可视化")
    print("="*100)

    # ============================================================================
    # 读取数据
    # ============================================================================
    print("\n正在读取pathway汇总数据...")
    pathway_summary = pd.read_csv(f'{results_dir}/pathway_summary.csv')

    # 简化pathway名称
    pathway_name_map = {
        'Glycolysis(Serine through glycolysis)': 'Glycolysis',
        'PP pathway(PEP through PP pathway, upper bound)  ': 'PP',
        'ED pathway(PYR through ED) ': 'ED',
        'Gluco-neogenesis(PEP from oxaloacetate)': 'Gluconeo-PEP',
        'Gluco-neogenesis(PYR from MAL, upper bound)': 'Gluconeo-PYR(U)',
        'Gluco-neogenesis(PYR from MAL, lower bound)': 'Gluconeo-PYR(L)',
        'TCA cycle(Oxaloacetate through TCA cycle)': 'TCA'
    }

    pathway_summary['Pathway_Short'] = pathway_summary['Pathway'].map(pathway_name_map)

    # 定义pathway顺序和颜色
    pathway_order = ['Glycolysis', 'PP', 'ED', 
                    'Gluconeo-PEP', 'Gluconeo-PYR(U)', 'Gluconeo-PYR(L)',
                    'TCA']

    pathway_colors = {
        'Glycolysis': "#2E9914",
        'PP': "#db349b",
        'ED': "#F2B342",
        'Gluconeo-PEP': '#CD61D7',
        'Gluconeo-PYR(U)': '#CD61D7',
        'Gluconeo-PYR(L)': '#CD61D7',
        'TCA': '#925A44',
    }

    print(f"加载数据: {len(pathway_summary)} 条记录")
    print(f"Methods: {methods}")
    print(f"Pathways: {pathway_order}")

    # ============================================================================
    # 图1: Pathway级别的平均绝对误差对比（主图）- 纵向排列
    # ============================================================================
    print("\n生成图1: Pathway级别的平均绝对误差对比（纵向排列）...")

    fig, ax = plt.subplots(figsize=(10, 12))

    # 准备绘图数据
    y = np.arange(len(pathway_order))
    height = 0.25

    for i, method in enumerate(methods):
        method_data = pathway_summary[pathway_summary['Method'] == method]

        means = []
        stds = []
        for pathway in pathway_order:
            pathway_row = method_data[method_data['Pathway_Short'] == pathway]
            if len(pathway_row) > 0:
                means.append(pathway_row['Avg_Abs_Error'].values[0])
                stds.append(pathway_row['Std_Abs_Error'].values[0])
            else:
                means.append(0)
                stds.append(0)

        bars = ax.barh(y + i*height, means, height, label=method, alpha=0.8,
            xerr=stds, color=methods_color[i],
            error_kw={'ecolor': methods_color[i],
                        'capsize': 3,
                        'capthick': 1.2,
                        'elinewidth': 1.2})

    ax.tick_params(axis='both', direction='in', width=4)

    # 图例
    legend = ax.legend(fontsize=20, loc='upper right')
    for i, text in enumerate(legend.get_texts()):
        if i < len(methods_color):
            text.set_color(methods_color[i])

    # 坐标轴设置
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel('MAE', fontsize=25, fontweight='bold')
    ax.set_yticks(y + height)
    ax.set_yticklabels(pathway_order, fontsize=18)

    # 为每个标签设置对应的pathway颜色
    for tick_label, pathway in zip(ax.get_yticklabels(), pathway_order):
        tick_label.set_color(pathway_colors.get(pathway, '#000000'))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pathway_error_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {output_path}")

    output_path_png = os.path.join(output_dir, 'pathway_error_comparison.png')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {output_path_png}")
    plt.close()

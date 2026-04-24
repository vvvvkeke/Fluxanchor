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
from tqdm import tqdm
from scipy.integrate import solve_ivp
sys.path.append(r'../script/')
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from script.AutoPACMEN_function import *
from script.ECMpy_function import *
from ast import literal_eval

# def add_dynamic_bounds(model, y):
#     """Use external concentrations to bound the uptake flux of glucose."""
#     biomass, glucose, acetate, co2 = y  # 扩展状态变量
#     glucose_max_import = -10 * glucose / (5 + glucose)
#     # 修改为reverse反应作为摄取
#     model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (0, -glucose_max_import)

def add_dynamic_bounds(model, y):
    biomass, glucose, co2 = y
    glucose_max_import = -10 * glucose / (5 + glucose)  # v = Vmax * [S] / (Km + [S])
    model.reactions.get_by_id("EX_glyc_e_reverse").bounds = (0, -glucose_max_import)
    # 添加乙酸的边界条件
    model.reactions.get_by_id("EX_ac_e").bounds = (0, 1000)  # 允许乙酸产生
    model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)  # 禁止乙酸摄取

def dynamic_system(t, y):
    biomass, glucose, co2 = y

    with model:
        add_dynamic_bounds(model, y)
        cobra.util.add_lp_feasibility(model)
        feasibility = cobra.util.fix_objective_as_constraint(model)

        # 设置目标为生物量并执行PFBA
        model.objective = obj  # 确保目标反应正确设置
        solution = cobra.flux_analysis.pfba(model)

        reactions_to_track = [
            f'{obj}',
            'EX_glyc_e_reverse',
            # 'ANS',
            'EX_co2_e'
        ]
        fluxes = [solution.fluxes[rxn] for rxn in reactions_to_track]

    # 调整通量符号（与原逻辑一致）
    fluxes[1] = -fluxes[1]  # 葡萄糖摄取转为负值

    # 应用生物量缩放
    fluxes = np.array(fluxes) * biomass

    # 进度条更新逻辑（保持不变）
    if dynamic_system.pbar is not None:
        current_progress = int(t * 100 / dynamic_system.t_max)
        if current_progress > dynamic_system.last_progress:
            dynamic_system.pbar.update(current_progress - dynamic_system.last_progress)
            dynamic_system.last_progress = current_progress
            dynamic_system.pbar.set_description(f't = {t:.3f}')

    return fluxes

# 添加进度条所需的属性
dynamic_system.pbar = None
dynamic_system.last_progress = 0
dynamic_system.t_max = 8  # 与 ts.max() 保持一致

def infeasible_event(t, y):
    """
    Determine solution feasibility.

    Avoiding infeasible solutions is handled by solve_ivp's built-in event detection.
    This function re-solves the LP to determine whether or not the solution is feasible
    (and if not, how far it is from feasibility). When the sign of this function changes
    from -epsilon to positive, we know the solution is no longer feasible.

    """

    with model:

        add_dynamic_bounds(model, y)

        cobra.util.add_lp_feasibility(model)
        feasibility = cobra.util.fix_objective_as_constraint(model)

    return feasibility - infeasible_event.epsilon

infeasible_event.epsilon = 1E-6
infeasible_event.direction = 1
infeasible_event.terminal = True

if __name__ == '__main__':
    methods = [
               "dfba",
               # "EkiLLm_normal",
               # "EkiLLm_etc",
               # "Catpred", "UniKP", "TurNup", "DLKcat",
               # "AutoPACMEN"
               ]
    obj = 'BIOMASS_Ec_iJO1366_core_53p95M'
    for method in methods:
        if method == "EkiLLm_normal":
            sbml_file = f"/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint.json"
            output_csv = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/dfba_normal.csv"
            model = get_enzyme_constraint_model_for_dfba(sbml_file)
        elif method == "dfba":
            sbml_file = f"/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint_etc.json"
            output_csv = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/dfba_without_ec.csv"
            model = cobra.io.json.load_json_model(sbml_file)
        elif method == "EkiLLm_etc":
            sbml_file = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint_etc.json"
            output_csv = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/dfba_etc.csv"
            model = get_enzyme_constraint_model_for_dfba(sbml_file)
        else:
            sbml_file = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_{method}/ecGEM/ecGEM_irr_enz_constraint.json"
            output_csv = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_{method}/dfba.csv"
            model = get_enzyme_constraint_model_for_dfba(sbml_file)
        model.reactions.get_by_id("EX_glc__D_e_reverse").bounds = (0, 0)  # glucose uptake
        model.reactions.get_by_id("EX_glc__D_e").bounds = (0, 0)  # 禁止葡萄糖释放
        model.reactions.get_by_id("EX_ac_e").bounds = (0, 1000)
        model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0)
        model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)
        model.reactions.get_by_id("EX_glyc_e_reverse").bounds = (0, 0)
        # 验证酶约束是否正确加载
        # if hasattr(model, '_temp_enzyme_constraint'):
        #     print(f"{method}模型的酶约束范围: {model._temp_enzyme_constraint.lb} - {model._temp_enzyme_constraint.ub}")
        #     print(f"酶约束系数数量: {len(model._temp_enzyme_coefficients)}")
        # else:
        #     print(f"警告: {method}模型没有酶约束!")
        # model.reactions.get_by_id("TRPS1").bounds = (0, 0)
        # model.reactions.get_by_id("TRPS2").bounds = (0, 0)
        # model.reactions.get_by_id("TRPS3").bounds = (0, 0)
        # model.reactions.get_by_id("TRPTRS").bounds = (0, 0)
        # solution = cobra.flux_analysis.pfba(model)
        # print(solution)
        # print(f"{method + str(solution.fluxes['EX_glc__D_e_reverse'])}")
        # continue
        print(f"开始{method}的dfba计算")
        # ts = np.linspace(0, 8, 100)  # 时间点：0到15小时，100个采样点 Desired integration resolution and interval
        ts = np.array([0,3,6,9,12,15,18,21,24,27,30,33,36,42,48])
        # 修改初始条件
        y0 = [0.05,  # 初始生物量
              2.76,  # 初始葡萄糖浓度
              # 0.0,  # 初始乙酸浓度
              0.0]  # 初始CO2浓度
        # 求解系统
        with tqdm() as pbar:
            dynamic_system.pbar = pbar
            sol = solve_ivp(
                fun=dynamic_system,
                events=[infeasible_event],
                t_span=(ts.min(), ts.max()),
                y0=y0,
                t_eval=ts,
                rtol=1e-6,
                atol=1e-8,
                method='BDF'
            )
        results_df = pd.DataFrame({
            'Time': sol.t,
            'Biomass': sol.y[0],
            'Glucose': sol.y[1],
            # 'Anthranilate': sol.y[2],
            'CO2': sol.y[2]
        })
        results_df.to_csv(output_csv, index=False)
    # results_df.to_csv('/home/zhangyangyu/kcat_km_predict/predict/dynamic_profiles.csv', index=False)

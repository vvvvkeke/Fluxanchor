import sys
import escher
import numpy as np
import pandas as pd
from escher import Builder
import cobra
from cobra.io import read_sbml_model
sys.path.append(r'./')
sys.path.append(r'../')
sys.path.append(r'./script/')
sys.path.append(r'../script/')
from script.ECMpy_function import get_enzyme_constraint_model
from cobra.io import read_sbml_model, save_json_model

if __name__ == '__main__':
    # ecModel_file = "predict/ENGRO2/Recon3D_301.xml"
    # ecModel_file = "predict/ENGRO2/network.xml"
    # model = cobra.io.read_sbml_model(ecModel_file)
    # use_substrate = 'EX_glc__D_e'
    # model.reactions.get_by_id(use_substrate).bounds = (-10,0)
    # solution = cobra.flux_analysis.pfba(model)
    # print(solution)
    # print(solution.fluxes)

    # ecModel_file = "../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint_eki2vivo.json"
    # ecModel_file = "../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_Catpred/ecGEM/ecGEM_irr_enz_constraint.json"
    # ecModel_file = "../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_AutoPACMEN/ecGEM/ecGEM_irr_enz_constraint.json"
    # ecModel_file = "../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint.json"
    # ecModel_file = "../predict/Bacillus subtilis/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint.json"
    ecModel_file = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/Bayesian/best_ecGEM.json"
    #
    # ecModel_file = f"/home/zhangyangyu/kcat_km_predict/predict/Bacillus subtilis/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint.json"
    # ecGEM_path = f"../predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/"
    # ecModel_file = f"{ecGEM_path}/Bayesian/best_ecGEM.json"
    model = get_enzyme_constraint_model(ecModel_file)
    # model = cobra.io.load_json_model(ecModel_file)
    # use_substrate = 'EX_glc__D_e'
    # # model.reactions.get_by_id('EX_lac__D_e').bounds = (-800, 0)
    # substrate_name = 'glucose'
    # glc_concentration_list = np.arange(1, 100, 10)
    # model.reactions.get_by_id(use_substrate).bounds = (0, 0)  # 完全禁止底物的分泌
    model.reactions.get_by_id('EX_glc__D_e_reverse').bounds = (4.01, 4.01)  # 设置底物吸收量  579
    model.reactions.get_by_id('EX_glc__D_e').bounds = (0, 0)  # 设置底物吸收量  579
    model.reactions.get_by_id("EX_fru_e").bounds = (0, 0)  # 697
    model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, 0) # 327
    # model.reactions.get_by_id("EX_glyc_e").bounds = (0, 0)  #988
    model.reactions.get_by_id("EX_ac_e").bounds = (1.51, 1.51)
    model.reactions.get_by_id("PGI").bounds = (0, 0)
    model.reactions.get_by_id("PGI_reverse").bounds = (0, 0)
    model.reactions.get_by_id("GLCt2pp").bounds = (0, 0)
    # model.reactions.get_by_id("XYLI2").bounds = (0, 0)
    # model.reactions.get_by_id("HEX1").bounds = (0, 0)
    model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").bounds = (0.44, 0.44)
    # model.reactions.get_by_id("BIOMASS_BS_10").bounds = (0.49, 0.49)
    # model.reactions.get_by_id("EX_o2_e_reverse").bounds = (0, 10)

    # # obj = 'BIOMASS_Ec_iJO1366_core_53p95M'
    # # model.objective = "BIOMASS_Ec_iJO1366_core_53p95M"
    # # model.objective = obj
    # # model.reactions.get_by_id("GLCpts").bounds = (0, 0) 
    # solution1 = cobra.flux_analysis.pfba(model)  # pFBA 的解 = 在“最优”前提下再挑“最经济”
    solution1 = model.optimize() # FBA 返回的只是 其中任意一个，经常包含：大量 循环通量（能耗高、生理上不合理）。多条平行路径同时活跃（细胞其实只选一条）
    print(solution1.objective_value) # 所有反应通量平方和，单位是 (mmol/gDW/h)²，毫无生理意义
    print("Glucose uptake (mmol/gDW/h):", solution1.fluxes['EX_glc__D_e_reverse'])
    # print("Growth rate (h⁻¹):", solution1.fluxes['BIOMASS_Ec_iJO1366_core_53p95M'])
    print("Biomass reaction ID:", model.objective.expression)
    print("EX_ac_e", solution1.fluxes['EX_ac_e'])
    print(solution1.fluxes['GLCDpp_num1'])
    print(solution1.fluxes['GLCDpp_num2'])
    print(solution1.fluxes['G6PDH2r'])
    # print(solution1.fluxes['GLCptspp_num3'])
    # print("PTAr_num1", solution1.fluxes['PTAr_num1'])
    # print("PTAr_num2", solution1.fluxes['PTAr_num2'])
    # for i in range(1, 101, 10):
    #     model.reactions.get_by_id("EX_ac_e_reverse").bounds = (0, i)
    #     solution1 = cobra.flux_analysis.pfba(model) 
    #     print(solution1.objective_value) # 所有反应通量平方和，单位是 (mmol/gDW/h)²，毫无生理意义
    #     print("acela uptake (mmol/gDW/h):", solution1.fluxes['EX_ac_e_reverse'])
    #     print("acela output (mmol/gDW/h):", solution1.fluxes['EX_ac_e'])
    #     print("Growth rate (h⁻¹):", solution1.fluxes['BIOMASS_Ec_iJO1366_core_53p95M'])

    # print(solution1)
    # print(solution1["PGI"]) 
    # model.reactions.get_by_id("PFK_num2").kcat
    # model.reactions.get_by_id("PFK_num2").kcat_MW 酶蛋白分子量
    # model.reactions.get_by_id("PFK_num2").km

    # r["pfk"] = max(pred_flux.get("PFK_num1", 0.0), pred_flux.get("PFK_num2", 0.0))
    # r["fba"] = max(pred_flux.get("FBA_num1", 0.0), pred_flux.get("FBA_num2", 0.0), pred_flux.get("FBA_num3", 0.0))
    # 将通量解转换为DataFrame
    # flux_df = pd.DataFrame({
    #     'reaction_id': solution1.fluxes.index,
    #     'flux': solution1.fluxes.values
    # })
    #

    # # 保存为CSV文件
    # csv_path = r"testttttt2.csv"
    # flux_df.to_csv(csv_path, index=False)
    # print(f"通量解已保存到: {csv_path}")

### ————————————————————————————————————————————————————————————————————————————————————————

    # ecGEM_path = f"../predict/Synechocystis sp/analysis/get_kcat_mw_by_EkiLLm/ecGEM/"
    # ecModel_file = f"{ecGEM_path}/Bayesian/best_ecGEM_GLGC.json"
    # model = get_enzyme_constraint_model(ecModel_file)
    # co2_uptake = 6
    # photon_uptake = 1000
    # # if m == "pfba":
    # #     model.reactions.get_by_id("EX_co2_e").bounds = (-co2_uptake, 0)
    # #     model.reactions.get_by_id("EX_photon_e").bounds = (-photon_uptake, 0)
    # #     model.reactions.get_by_id("EX_ac_e").bounds = (0, 0)
    # #     model.reactions.get_by_id("EX_succ_e").bounds = (0, 0)
    # #     obj = "BIOMASS_Ec_SynAuto_1"      
    # #     model.objective = obj    
    # # else:
    # # model.reactions.get_by_id("EX_hco3_e").bounds = (-10, 0)
    # model.reactions.get_by_id("EX_co2_e_reverse").bounds = (0, co2_uptake)
    # model.reactions.get_by_id("EX_photon_e_reverse").bounds = (0, photon_uptake)
    # # model.reactions.get_by_id("EX_ac_e").bounds = (0, 0)
    # model.reactions.get_by_id("GLGC").bounds = (0, 0)
    # model.reactions.get_by_id("EX_nh4_e").bounds = (-10, 0)
    # model.reactions.get_by_id("EX_no3_e_reverse").bounds = (0, 10)
    # model.reactions.get_by_id("EX_akg_e").bounds = (0.1, 10)
    # model.reactions.get_by_id("EX_pyr_e").bounds = (0.1, 10)
    # # model.reactions.EX_nh4_e.lower_bound = -10.0 # 允许摄取铵盐

    # model.reactions.get_by_id("EX_succ_e").bounds = (0, 1000)
    # obj = "BIOMASS_Ec_SynAuto_1"
    
    # model.objective = obj
    # # model.objective = {model.reactions.EX_pyr_e: 0.1, model.reactions.EX_akg_e: 0.2, model.reactions.BIOMASS_Ec_SynAuto_1: 2}
    # solution1 = cobra.flux_analysis.pfba(model)
    # # 重点观察：有机酸分泌情况
    # print(f"突变体丙酮酸分泌通量 (EX_pyr_e): {solution1.fluxes.get('EX_pyr_e', 0):.4f}")
    # print(f"突变体α-酮戊二酸分泌通量 (EX_akg_e): {solution1.fluxes.get('EX_akg_e', 0):.4f}")
    # print(f"突变体琥珀酸分泌通量 (EX_succ_e): {solution1.fluxes.get('EX_succ_e', 0):.4f}")
    # print("Growth rate (h⁻¹):", solution1.fluxes[obj])
    # print(":", solution1)

    builder = Builder()
    builder.reaction_data = solution1.fluxes
    builder.save_html("ecoli.html")

    # from cobra.flux_analysis import moma
    # wt_solution = model.optimize()  # 野生型
    # model.reactions.GLGC.knock_out()
    # solution = moma(model, wt_solution)
    # print(solution)
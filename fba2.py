import sys
import escher
import numpy as np
import pandas as pd
from escher import Builder
import cobra
from cobra.io import read_sbml_model
sys.path.append(r'./script/')
from script.ECMpy_function import get_enzyme_constraint_model
from cobra.io import read_sbml_model, save_json_model

if __name__ == '__main__':
    # ecModel_file = "predict/ENGRO2/Recon3D_301.xml"
    ecModel_file = "predict/ENGRO2/network.xml"
    model = cobra.io.read_sbml_model(ecModel_file)
    use_substrate = 'EX_glc__D_e'
    model.reactions.get_by_id(use_substrate).bounds = (-10,0)
    solution = cobra.flux_analysis.pfba(model)
    print(solution)
    print(solution.fluxes)

    # ecModel_file = "predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint_etc.json"
    #
    # obj = 'BIOMASS_Ec_iJO1366_core_53p95M'
    # model = get_enzyme_constraint_model(ecModel_file)
    # use_substrate = 'EX_glc__D_e'
    # # model.reactions.get_by_id('EX_lac__D_e').bounds = (-800, 0)
    # substrate_name = 'glucose'
    # glc_concentration_list = np.arange(1, 100, 10)
    # model.reactions.get_by_id(use_substrate).bounds = (0, 0)  # 完全禁止底物的分泌
    # model.reactions.get_by_id(use_substrate + '_reverse').bounds = (0, 70)  # 设置底物吸收量
    #
    # solution1 = cobra.flux_analysis.pfba(model)
    # print(solution1)
    # print(solution1["EX_ac_e"])
    # print(solution1["EX_lac__L_e"])

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
    # builder = Builder()
    # builder.reaction_data = solution1.fluxes
    # builder.save_html("final333.html")
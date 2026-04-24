import sys
import escher
import pandas as pd
from cobra import Reaction
from escher import Builder
import cobra
from cobra.io import read_sbml_model
sys.path.append(r'./script/')
from script.ECMpy_function import get_enzyme_constraint_model
from cobra.io import read_sbml_model, save_json_model
from cobra.flux_analysis import flux_variability_analysis

if __name__ == '__main__':
    # ecModel_file = "/home/zhangyangyu/kcat_km_predict/predict/Synechocystis sp/network.json"
    # model = cobra.io.json.load_json_model(ecModel_file)
    # cobra.io.write_sbml_model(
    #     model,
    #     "/home/zhangyangyu/kcat_km_predict/predict/Synechocystis sp/network.xml",
    # )
    #
    # quit()
    ecModel_file = "predict/Synechocystis sp/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint_delta_glgc.json"
    model = get_enzyme_constraint_model(ecModel_file)
    # model = cobra.io.json.load_json_model(ecModel_file)
    # model.solver = 'scipy'  # 需要安装 gurobipy  glpk

    print(model.solver.interface.__name__)
    # BIOMASS_Ec_SynAuto_1
    # BIOMASS_Ec_SynMixo_1
    # BIOMASS_Ec_SynHetero_1
    obj = "BIOMASS_Ec_SynAuto_1"
    model.objective = obj
    model.objective_direction = "max"
    # 设置求解精度
    # model.solver.configuration.tolerances.feasibility = 1e-13
    # model.solver.configuration.tolerances.integrality = 1e-13
    # model.solver.configuration.tol_bnd = 1e-13  # Boundary tolerance
    # model.solver.configuration.tol_dj = 1e-13  # Dual tolerance

    # model.objective = model.problem.Objective(
    #     10 * model.reactions.BIOMASS_Ec_SynAuto_1.flux_expression +
    #     0.27 * model.reactions.PYK_num1.flux_expression +
    #     0.24 * model.reactions.ICDHyr.flux_expression,
    #     direction='max'
    # )

    # model.objective = model.problem.Objective(
    #     0.999 * model.reactions.BIOMASS_Ec_SynAuto_1.flux_expression -
    #     0.001 * model.reactions.ATPM.flux_expression,
    #     direction='max'
    # )

    # model.objective = model.problem.Objective(
    #     0.892 * model.reactions.BIOMASS_Ec_SynAuto_1.flux_expression +
    #     0.0647235 * model.reactions.EX_pyr_e.flux_expression +
    #     0.110783474 * model.reactions.EX_akg_e.flux_expression,
    #     direction='max'
    # )
    # 0.0642867275
    # model.objective = model.problem.Objective(
    #     0.89 * model.reactions.BIOMASS_Ec_SynAuto_1.flux_expression +
    #     0.0642866 * model.reactions.EX_pyr_e.flux_expression +
    #     0.110783474 * model.reactions.EX_akg_e.flux_expression,
    #     direction='max'
    # )

    # model.objective = model.problem.Objective(
    #     1.3 * model.reactions.BIOMASS_Ec_SynAuto_1.flux_expression +
    #     0.06949 * model.reactions.EX_pyr_e.flux_expression +
    #     0.11814 * model.reactions.EX_akg_e.flux_expression,
    #     direction='max'
    # )  before

    # model.objective = model.problem.Objective(
    #     1.3 * model.reactions.BIOMASS_Ec_SynAuto_1.flux_expression +
    #     0.00000002 * model.reactions.PFK_num1.flux_expression,
    #     direction='max'
    # )

    # model.objective = model.problem.Objective(
    #     0.97 * model.reactions.BIOMASS_Ec_SynAuto_1.flux_expression +
    #     # 0.0426023 * model.reactions.EX_pyr_e.flux_expression +
    #     0.072424 * model.reactions.EX_akg_e.flux_expression,
    #     direction='max'
    # )

    model.reactions.get_by_id("EX_glc__D_e").bounds = (-8, 0)
    # model.reactions.get_by_id("EX_pyr_e").bounds = (0, 999)
    # model.reactions.get_by_id("EX_akg_e").bounds = (0, 999)
    # model.reactions.get_by_id("GLGC").bounds = (0, 0)
    # model.reactions.get_by_id("EX_co2_e_reverse").bounds = (0, 200)
    # model.reactions.get_by_id("EX_co2_e").bounds = (0, 2000)
    # # model.reactions.get_by_id("EX_photon_e_reverse").bounds = (0, 20)
    # model.reactions.get_by_id("EX_photon_e_reverse").bounds = (0, 800)  # 80才能排除1 1的pyr和akg
    model.reactions.get_by_id("EX_o2_e").bounds = (0, 3000)
    model.reactions.get_by_id("EX_o2_e_reverse").bounds = (0, 3000)
    # model.reactions.get_by_id("EX_no3_e_reverse").bounds = (0, 0)
    # model.reactions.get_by_id("EX_nh4_e").bounds = (-3, 3)  # 极度碳饥饿，不会出现PDH
    # model.reactions.get_by_id("PYK_num1").bounds = (15, 68)
    # model.reactions.EX_akg_e.lower_bound = 1
    # model.reactions.EX_pyr_e.lower_bound = 1
    # reaction = cobra.Reaction("NADH_OX_STRESS")
    # reaction.add_metabolites({
    #     model.metabolites.nadh_c: -1,
    #     model.metabolites.nad_c: 1,
    #     model.metabolites.akg_e: -1  # 假设 akg_e 排出带走一个 NADH
    # })
    # reaction.lower_bound = 5  # 模拟 NADH 积累需要排出一定量
    # model.add_reactions([reaction])
    #
    # reaction = cobra.Reaction("REDOX_DRAIN")
    # reaction.name = "Redox pressure relief"
    # reaction.lower_bound = 0
    # reaction.upper_bound = 1000

    # 假设 pyr_e 带走一个 NADPH，虚拟消耗 NADPH
    # reaction.add_metabolites({
    #     model.metabolites.nadph_c: -1,
    #     model.metabolites.nadp_c: 1,
    #     model.metabolites.pyr_e: -1
    # })
    # model.add_reactions([reaction])
    # model.reactions.REDOX_DRAIN.lower_bound = 10

    #
    # model.reactions.get_by_id("ACLSa_1").bounds = (0, 0.337)
    # model.reactions.get_by_id("ACLSb_1").bounds = (0, 0.251)
    # model.reactions.get_by_id("ALAD_L2").bounds = (0, 0.14)
    # model.reactions.get_by_id("DHDPS").bounds = (0, 0.0654)
    # model.reactions.get_by_id("ANSN").bounds = (0, 0.0214)
    # model.reactions.get_by_id("SHCHCS2").bounds = (0, 0.000493)
    # model.reactions.get_by_id("ADCL").bounds = (3, 268)


    # print(model.reactions.get_by_id("EX_co2_e_reverse").bounds)
    # print(model.reactions.get_by_id("EX_photon_e_reverse").bounds)

    # # print(model.reactions)
    # # biomass_reaction = model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M")
    # # model.objective = biomass_reaction
    # print(model.objective)
    # print("当前的优化目标是:", model.objective.variables)
    # solution = model.optimize()
    # print("fba目标函数值（生物量通量）:", solution.objective_value)
    # solution1 = model.optimize()
    # model.reactions.get_by_id("PKETX_num2").bounds = (0, 0)
    # model.reactions.get_by_id("PKETX_num1").bounds = (0, 0)
    # model.reactions.get_by_id("PKETF_num2").bounds = (0, 0)
    # model.reactions.get_by_id("PKETF_num1").bounds = (0, 0)
    # model.reactions.get_by_id("PEPC").bounds = (0, 0)
    # model.reactions.get_by_id("GLYTA").bounds = (0, 0)
    # model.reactions.get_by_id("RBFSa").bounds = (0, 0)
    # model.reactions.get_by_id("EX_ac_e").bounds = (0, 0)
    # model.reactions.get_by_id("EX_succ_e").bounds = (0, 0)

    model.reactions.get_by_id("EX_ac_e").bounds = (0, 0)
    model.reactions.get_by_id("POR_syn").bounds = (0, 0)
    # model.reactions.get_by_id("PSIa_num1").bounds = (0, 100)
    # model.reactions.get_by_id("PSIa_num2").bounds = (0, 0)
    # model.reactions.get_by_id("CYO1b_syn_1_num1").bounds = (0, 0)
    # model.reactions.get_by_id("CYO1b_syn_1_num2").bounds = (0, 0)
    # model.reactions.get_by_id("CYO1b_syn_1_num3").bounds = (0, 0)
    # model.reactions.get_by_id("CYO1b_syn_1_num4").bounds = (0, 0)
    # model.reactions.get_by_id("CYO1b2_syn_1_num1").bounds = (0, 0)
    # model.reactions.get_by_id("CYO1b2_syn_1_num2").bounds = (0, 0)
    # model.reactions.get_by_id("CYO1b2_syn_1_num3").bounds = (0, 0)
    # model.reactions.get_by_id("CYO1b2_syn_1_num4").bounds = (0, 0)
    # model.reactions.get_by_id("ATPS4rpp_1").bounds = (0, 0)
    # model.reactions.get_by_id("EX_succ_e").bounds = (0, 0)
    # model.reactions.get_by_id("PSTA").bounds = (0, 99)
    # model.reactions.get_by_id("GHMT2r").bounds = (0, 0)
    # model.reactions.get_by_id("TALA_reverse").bounds = (0, 1)
    # model.reactions.get_by_id("PTAr_reverse").bounds = (0, 2)
    # model.reactions.get_by_id("ACS").bounds = (0, 1)

    # model.reactions.get_by_id("ORNTA").bounds = (0, 0)  # 消耗akg_c
    # model.reactions.get_by_id("OXGDC").bounds = (0, 0)  # 消耗akg_c
    # model.reactions.get_by_id("OXGDC2").bounds = (0, 0)  # 消耗akg_c
    # model.reactions.get_by_id("ABTA").bounds = (0, 3)  # 消耗akg_c
    # model.reactions.get_by_id("GLUDy_1_reverse").bounds = (0, 11)  # 消耗akg_c
    # # model.reactions.get_by_id("ASPTA_num1").bounds = (0, 2)  # 消耗akg_c
    # # model.reactions.get_by_id("ASPTA_num2").bounds = (0, 2)  # 消耗akg_c
    # model.reactions.get_by_id("GLMS_syn").bounds = (0, 0)  # 消耗akg_c
    # model.reactions.get_by_id("GLUSx").bounds = (0, 0)  # 消耗akg_c,使用nadh

    # model.reactions.get_by_id("GLNS_1_num1").bounds = (0, 2)  # 消耗glu__L_c
    # model.reactions.get_by_id("GLNS_1_num2").bounds = (0, 0)  # 消耗glu__L_c


    # #
    # model.reactions.get_by_id("ACLSa_1").bounds = (0, 1)  # 消耗pyr_c
    # model.reactions.get_by_id("ACLSb_1").bounds = (0, 1)  # 消耗pyr_c
    # model.reactions.get_by_id("AGTi_reverse").bounds = (0, 2)  # 消耗pyr_c
    # model.reactions.get_by_id("DHDPS").bounds = (0, 2)  # 消耗pyr_c
    # # model.reactions.get_by_id("DXPS").bounds = (0, 2)  # 消耗pyr_c
    # # model.reactions.get_by_id("ALAD_L2").bounds = (0, 2)  # 消耗pyr_c
    # model.reactions.get_by_id("POR_syn").bounds = (0, 0)  # 消耗pyr_c  1
    # model.reactions.get_by_id("PPS").bounds = (0, 0)  # 消耗pyr_c  1
    # model.reactions.get_by_id("PDH_num1").bounds = (0, 7)  # 消耗pyr_c
    # model.reactions.get_by_id("PDH_num2").bounds = (0, 0)  # 消耗pyr_c
    # model.reactions.get_by_id("PDH_num3").bounds = (0, 0)  # 消耗pyr_c
    # model.reactions.get_by_id("PDHa").bounds = (0,0)  # 消耗pyr_c
    # model.reactions.get_by_id("ATPM").bounds = (0, 0)
    # model.reactions.get_by_id("PHETA1_reverse_num1").bounds = (0, 0)



    # model.reactions.get_by_id("DXPS").bounds = (0, 0)
    # model.reactions.get_by_id("GLXO3r").bounds = (0, 0)
    # model.reactions.get_by_id("GLMS_syn").bounds = (0, 0)
    # model.reactions.get_by_id("GLUSx").bounds = (0, 0)

    #################
    ### 限制这两个，出现PDH
    # model.reactions.get_by_id("PTAr_reverse").bounds = (0, 2)
    # model.reactions.get_by_id("ACS").bounds = (0, 1)
    #################

    # model.reactions.get_by_id("GLCS1_num1").bounds = (0, 0)
    # model.reactions.get_by_id("GLCS1_num2").bounds = (0, 0)

    # model.reactions.get_by_id("GLBRAN2").bounds = (0, 0)
    # model.reactions.get_by_id("GLCP_num1").bounds = (0, 0)
    # model.reactions.get_by_id("GLCP_num2").bounds = (0, 0)

    # model.reactions.get_by_id("GALUi").bounds = (0, 0.1)
    # model.reactions.get_by_id("G1PCTYT").bounds = (0, 0.5)
    # model.reactions.get_by_id("PDH_num1").bounds = (0, 1.8)
    # model.reactions.get_by_id("PDH_num2").bounds = (0, 1.8)
    # model.reactions.get_by_id("PDH_num3").bounds = (0, 1.8)
    # model.reactions.get_by_id("EX_akg_e").bounds = (0.5, 999)
    # model.reactions.get_by_id("EX_pyr_e").bounds = (0.5, 999)
    # "glycogen_c": -0.21031,
    # model.reactions.get_by_id("PGMT").bounds = (0, 0)
    # model.reactions.get_by_id("GLGC").bounds = (0, 0.1)
    # model.reactions.get_by_id("GALUi").bounds = (0, 0.5)

    # model.reactions.get_by_id("PKETF_num1").bounds = (0, 1)
    # model.reactions.get_by_id("PKETF_num2").bounds = (0, 0)

    # model.reactions.get_by_id("GLYCLc").bounds = (0, 0)
    # model.reactions.get_by_id("ASPTA_reverse_num1").bounds = (0, 0)
    # model.reactions.get_by_id("ASPTA_reverse_num2").bounds = (0, 0)

    # solution1 = model.optimize()
    solution1 = cobra.flux_analysis.pfba(model)
    # solution1 = flux_variability_analysis(model, fraction_of_optimum=1.0)
    # print(solution1.loc[["EX_akg_e", "EX_pyr_e", "BIOMASS_Ec_SynAuto_1"]])
    # quit()
    print(solution1)
    print(solution1["GLGC"])
    # quit()
    # solution1 = model.optimize()

    # print(solution1['PKETX_num2'])
    # print(solution1['PKETX_num1'])
    # print(solution1['PKETF_num2'])
    # print(solution1['PKETF_num1'])
    # print(solution1['SBP'])
    # print(solution1['FBA3_num2'])

    # print(solution1['GLYCK2'])
    # print(solution1['GLYCK'])
    # print(solution1['ME2'])
    print(f"PDH: {solution1['PDH_num1']}")
    print(f"PDH: {solution1['PDH_num2']}")
    print(f"PDH: {solution1['PDH_num3']}")
    # print(solution1['MDH'])
    # print(solution1['ABTA'])
    # print(solution1['SSALy'])
    # print(solution1['GLXCL'])
    # print(solution1['GLYCLTDx_reverse_num2'] or solution1['GLYCLTDx_reverse_num1'] )
    # print(solution1['PGLYCP_num3'] or solution1['PGLYCP_num2'] or solution1['PGLYCP_num1'])
    # print(solution1['RBCh_2'])


    # print(solution1["GLCP_num1"])
    # print(solution1["GLCP_num2"])
    # print(solution1["PGMT"])
    # print(solution1["PGI"])
    # print(solution1["PYK2_num1"])
    # print(solution1["PYK2_num2"])
    # print(solution1["GLGC"])
    for reaction_id, flux in solution1.fluxes.items():
        # if reaction_id.startswith('EX') and not reaction_id.endswith("reverse"):
        if reaction_id.startswith('EX'):
                print(f"{reaction_id}: {flux}")
    # # 打印结果
    # print("pfba目标函数值（生物量通量）:", solution1.objective_value)
    # print("求解状态:", solution1.status)
    builder = Builder()
    builder.reaction_data = solution1.fluxes
    builder.save_html("final.html")

    flux_df = pd.DataFrame({
        'reaction_id': solution1.fluxes.index,
        'flux': solution1.fluxes.values
    })

    # 保存为CSV文件
    csv_path = r"lanzao4.csv"
    flux_df.to_csv(csv_path, index=False)
    print(f"通量解已保存到: {csv_path}")
    print(solution1)
    print(f"biomass: {solution1[obj]}")
    print(f"EX_pyr: {solution1['EX_pyr_e']}")
    print(f"EX_akg: {solution1['EX_akg_e']}")
    print(f"ATPM: {solution1['ATPM']}")
    print(f"PYK: {solution1['PYK_num1'] or solution1['PYK_num2']}")
    print(f"ICDHyr: {solution1['ICDHyr']}")

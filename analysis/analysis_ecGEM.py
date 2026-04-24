import argparse
import datetime
import sys

import matplotlib.pyplot as plt
import pandas as pd


sys.path.append(r'./')
sys.path.append(r'../')
sys.path.append(r'./script/')
sys.path.append(r'../script/')

from script.ECMpy_function import *

parser = argparse.ArgumentParser()


def drwa_kcat_distribution(ecGEM_path, ecModel_file, method):
    figure_path = ecGEM_path
    reaction_kcat_MW_file = f"../analysis/get_kcat_mw_by_%s/reaction_change_by_enzuse.csv" % method
    reaction_kcat_MW = pd.read_csv(reaction_kcat_MW_file)
    reaction_kcat_MW = round(reaction_kcat_MW, 3)
    reaction_kcat_dis_file = '../analysis/reaction_kcat_distrbution.png'
    reaction_mw_dis_file = '../analysis/reaction_mw_distrbution.png'
    reaction_kcat_select = reaction_kcat_MW[reaction_kcat_MW['data_type'] != 'fill']
    # Sort values
    """kcat分布图"""
    sorted_data = reaction_kcat_select.sort_values('kcat')
    sorted_data = sorted_data.reset_index(drop=True)
    y_index = sorted_data.index / (sorted_data.shape[0] - 1)
    data_cdf_data = sorted_data['kcat']
    x_name = "<b>kcat(1/s)<b>"
    y_name = "<b>Cummulative distribution<b>"
    nticks = 1000
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    fig = draw_cdf_fig(data_cdf_data, reaction_kcat_dis_file, x_name, y_name, y_index, nticks)
    print("kcat分布图画完了")


def draw_phpp(figure_path, ecModel_output_file, obj):
    # if is_auto:
    #     obj = 'BIOMASS_Ec_SynAuto_1'  # BIOMASS_Ec_SynAuto_1 BIOMASS_Ec_SynMixo_1  BIOMASS_Ec_SynHetero_1 obj是反应，一般指向带有BIOMASS的
    # else:
    #     obj = 'BIOMASS_Ecoli_core_w_GAM'

    starttime = datetime.datetime.now()
    """phpp图,非常非常慢"""
    if is_auto:
        z_id = 'EX_co2_e_reverse'
        x_id = 'EX_photon_e_reverse'
        substrate_bound = 300  # 这个用的是x
        o2_bound = 10  # 这个用的是z
        GEM_output_file = f'{figure_path}/normal_{x_id}_{z_id}_df.csv'
        ecGEM_output_file = f'{figure_path}/ecGEM_{x_id}_{z_id}_df.csv'
        PhPP_output_fig_file = f'{figure_path}/PhPP_combine_{x_id}_{z_id}.pdf'
        ec_PhPP_output_fig_file = f'{figure_path}/ec_PhPP_combine_{x_id}_{z_id}.pdf'
        GEM_glc_o2_df = get_PhPP_data(ecModel_output_file, 'GEM', obj,
                                      substrate_bound, o2_bound, 11, GEM_output_file, x_id, z_id)
        GEM_glc_o2_df = pd.read_csv(GEM_output_file, sep=",", encoding="utf-8")
        GEM_glc_o2_df = GEM_glc_o2_df.fillna(0)
        # GEM_glc_o2_df.drop(0, axis=0, inplace=True)
        # GEM_glc_o2_df.drop('0', axis=1, inplace=True)
        ecGEM_glc_o2_df = get_PhPP_data(ecModel_output_file, 'ecGEM', obj,
                                        substrate_bound, o2_bound, 11, ecGEM_output_file, x_id, z_id)
        ecGEM_glc_o2_df = pd.read_csv(ecGEM_output_file, sep=",", encoding="utf-8")
        ecGEM_glc_o2_df = ecGEM_glc_o2_df.fillna(0)
        # ecGEM_glc_o2_df.drop(0, axis=0, inplace=True)
        # ecGEM_glc_o2_df.drop('0', axis=1, inplace=True)
        fig = draw_3d_rbas2(GEM_glc_o2_df, substrate_bound, o2_bound, 0.5, 11, PhPP_output_fig_file,
                           x_label="Photon", z_label="CO2")
        fig = draw_3d_rbas2(ecGEM_glc_o2_df, substrate_bound, o2_bound, 0.5, 11, ec_PhPP_output_fig_file,
                           x_label="Photon", z_label="CO2")
        print("phpp图画完了")
        endtime = datetime.datetime.now()
        print(endtime - starttime)
    else:
        z_id = 'EX_o2_e'
        x_id = 'EX_glc__D_e'
        substrate_bound = 20  # 这个用的是x
        o2_bound = 20  # 这个用的是z
        GEM_output_file = f'{figure_path}/normal_{x_id}_{z_id}_df.csv'
        ecGEM_output_file = f'{figure_path}/ecGEM_{x_id}_{z_id}_df.csv'
        PhPP_output_fig_file = f'{figure_path}/PhPP_combine_{x_id}_{z_id}.pdf'
        ec_PhPP_output_fig_file = f'{figure_path}/ec_PhPP_combine_{x_id}_{z_id}.pdf'
        # GEM_glc_o2_df = get_PhPP_data(ecModel_output_file, 'GEM', obj,
        #                               substrate_bound, o2_bound, 11, GEM_output_file, x_id, z_id)
        GEM_glc_o2_df = pd.read_csv(f'{figure_path}/normal_{x_id}_{z_id}_df.csv', sep=",", encoding="utf-8")
        # GEM_glc_o2_df = GEM_glc_o2_df.fillna(0)
        # GEM_glc_o2_df = GEM_glc_o2_df.drop('Unnamed: 0', axis=1)
        # GEM_glc_o2_df = GEM_glc_o2_df.drop(0, axis=0)
        # GEM_glc_o2_df.drop(0, axis=0, inplace=True)
        # GEM_glc_o2_df.drop(0, axis=1, inplace=True)
        # GEM_glc_o2_df = GEM_glc_o2_df.drop('0', axis=1)
        # ecGEM_glc_o2_df = get_PhPP_data(ecModel_output_file, 'ecGEM', obj,
        #                                 substrate_bound, o2_bound, 11, ecGEM_output_file, x_id, z_id)

        # ecGEM_glc_o2_df.drop(0, axis=0, inplace=True)
        # ecGEM_glc_o2_df.drop(0, axis=1, inplace=True)
        ecGEM_glc_o2_df = pd.read_csv(f'{figure_path}/ecGEM_{x_id}_{z_id}_df.csv', sep=",", encoding="utf-8")
        # ecGEM_glc_o2_df = ecGEM_glc_o2_df.fillna(0)
        # ecGEM_glc_o2_df = ecGEM_glc_o2_df.drop('Unnamed: 0', axis=1)
        # ecGEM_glc_o2_df = ecGEM_glc_o2_df.drop(0, axis=0)
        # ecGEM_glc_o2_df.drop(0, axis=1, inplace=True)
        # ecGEM_glc_o2_df = ecGEM_glc_o2_df.drop('0', axis=1)
        fig = draw_3d_rbas2(GEM_glc_o2_df, substrate_bound, o2_bound, 2, 11, PhPP_output_fig_file,
                           x_label="glucose", z_label="O2")
        fig = draw_3d_rbas2(ecGEM_glc_o2_df, substrate_bound, o2_bound, 2, 11, ec_PhPP_output_fig_file,
                           x_label="glucose", z_label="O2")
        print("phpp图画完了")
        endtime = datetime.datetime.now()
        print(endtime - starttime)


def draw_overflow(figure_path, ecModel_output_file, obj):
    if is_auto:
        # obj = "BIOMASS_Ec_SynAuto_1"  ##如果模型初始没有指定优化目标，在这里一定要指定优化目标
        enz_model = get_enzyme_constraint_model(ecModel_output_file)
        norm_model = cobra.io.json.load_json_model(ecModel_output_file)
        enz_model.objective = obj
        norm_model.objective = obj
        enz_model.reactions.get_by_id("EX_co2_e_reverse").upper_bound = 16
        norm_model.reactions.get_by_id("EX_co2_e_reverse").upper_bound = 16
        enz_model.reactions.get_by_id("EX_photon_e_reverse").upper_bound = 300
        norm_model.reactions.get_by_id("EX_photon_e_reverse").upper_bound = 300
        overflow_result_figfile = f"{figure_path}/pfba_overflow_result.png"
        use_substrate = 'EX_co2_e'  # EX_photon_e  EX_co2_e
        glc_concentration_list = np.arange(1, 16, 1)
        columns = ['biomass', 'O_2', 'gly']
        correspond_rxn = ['BIOMASS_Ec_SynAuto_1', 'EX_o2_e', 'EX_gly_e']
        for reaction in enz_model.reactions:
            # 检查反应ID是否以"EX_"开头，这是交换反应的常见标识
            if not reaction.id.startswith("EX_"):
                # 将交换反应的ID添加到columns数组中
                columns.append(reaction.id)
                correspond_rxn.append(reaction.id)
        GEMyield_list = pd.DataFrame()
        ecGEMyield_list = pd.DataFrame()
        substrate_name = 'CO2'
        # Calculate yield for growth_model
        with enz_model as growth_model:
            for glc_concentration in glc_concentration_list:
                ecGEMyield_list = calculate_yield(growth_model, use_substrate, substrate_name, glc_concentration,
                                                  columns, correspond_rxn, ecGEMyield_list)

        # Calculate yield for norm_model
        with norm_model as growth_model:
            for glc_concentration in glc_concentration_list:
                GEMyield_list = calculate_yield(growth_model, use_substrate, substrate_name, glc_concentration, columns,
                                                correspond_rxn, GEMyield_list)
        substrate_bound = 15
        obj_bound = 1
        secrate_bound = 10
        column_list = ['biomass', 'O_2', 'gly']
        y_axis_loc_list = ['left', 'right', 'right', 'right']
        color_list = generate_random_colors(len(column_list) * 2)

        fig = draw_overflow_fig(GEMyield_list, ecGEMyield_list, column_list, y_axis_loc_list, color_list,
                                substrate_name, substrate_bound, obj_bound, secrate_bound, overflow_result_figfile)
        ecGEMyield_list.to_csv(f"{figure_path}/ecGEMyield.csv", index=False)
        GEMyield_list.to_csv(f"{figure_path}/GEMyield.csv", index=False)
        # 初始化一个空列表来存储有差异的列ID
        different_columns = []
        # 遍历所有列  MGDGE160  ANS GLYCLa GLYCLc GLXO3r UDPG4E THRS_num1 PPPGO_num1 POR_1 HISTDa_1_num1 SBP  CTPS2
        for column in ecGEMyield_list.columns:
            # 比较两个 DataFrame 在当前列的内容是否有差异
            if not ecGEMyield_list[column].equals(GEMyield_list[column]):
                different_columns.append(column)

        # 输出有差异的列ID
        print("有差异的列ID：", different_columns)
        #####################

        overflow_result_figfile = f"{figure_path}/pfba_overflow_result_photon.png"
        use_substrate = 'EX_photon_e'  # EX_photon_e  EX_co2_e
        glc_concentration_list = np.arange(0, 300, 30)
        columns = ['biomass', 'O_2', 'gly']
        correspond_rxn = ['BIOMASS_Ec_SynAuto_1', 'EX_o2_e', 'EX_gly_e']
        for reaction in enz_model.reactions:
            # 检查反应ID是否以"EX_"开头，这是交换反应的常见标识
            if not reaction.id.startswith("EX_"):
                # 将交换反应的ID添加到columns数组中
                columns.append(reaction.id)
                correspond_rxn.append(reaction.id)
        GEMyield_list = pd.DataFrame()
        ecGEMyield_list = pd.DataFrame()
        substrate_name = 'Photon'
        with enz_model as growth_model:
            for glc_concentration in glc_concentration_list:
                ecGEMyield_list = calculate_yield(growth_model, use_substrate, substrate_name, glc_concentration,
                                                  columns, correspond_rxn, ecGEMyield_list)

        # Calculate yield for norm_model
        with norm_model as growth_model:
            for glc_concentration in glc_concentration_list:
                GEMyield_list = calculate_yield(growth_model, use_substrate, substrate_name, glc_concentration, columns,
                                                correspond_rxn, GEMyield_list)
        substrate_bound = 16
        obj_bound = 1
        secrate_bound = 10
        column_list = ['biomass', 'O_2', 'gly']
        y_axis_loc_list = ['left', 'right', 'right', 'right']
        color_list = generate_random_colors(len(column_list) * 2)

        fig = draw_overflow_fig(GEMyield_list, ecGEMyield_list, column_list, y_axis_loc_list, color_list,
                                substrate_name, substrate_bound, obj_bound, secrate_bound, overflow_result_figfile)
        ecGEMyield_list.to_csv(f"{figure_path}/ecGEMyield_photon.csv", index=False)
        GEMyield_list.to_csv(f"{figure_path}/GEMyield_photon.csv", index=False)
        # 初始化一个空列表来存储有差异的列ID
        different_columns = []
        # 遍历所有列
        for column in ecGEMyield_list.columns:
            # 比较两个 DataFrame 在当前列的内容是否有差异
            if not ecGEMyield_list[column].equals(GEMyield_list[column]):
                different_columns.append(column)

        # 输出有差异的列ID
        print("有差异的列ID：", different_columns)
    else:
        # inputfiles
        # json_model_file = "./model/eciML1515.json"
        enz_model = get_enzyme_constraint_model(ecModel_output_file)
        norm_model = cobra.io.json.load_json_model(ecModel_output_file)
        # method = 'AutoPACMEN'  # DLKcat
        # reaction_kcat_MW_file = "./analysis/get_kcat_mw_by_%s/reaction_change_by_enzuse.csv" % method
        # reaction_kcat_MW = pd.read_csv(reaction_kcat_MW_file, index_col=0)
        # reaction_kcat_MW = round(reaction_kcat_MW, 3)
        
        # outputfiles
        overflow_result_figfile = f"{figure_path}/pfba_overflow_result.png"
        use_substrate = 'EX_glc__D_e'
        substrate_name = 'glucose'
        glc_concentration_list = np.arange(1, 16, 1)
        columns = ['biomass', 'O_2', 'CO_2', 'acetate']
        correspond_rxn = [obj, 'EX_o2_e_reverse', 'EX_co2_e', 'EX_ac_e']
        GEMyield_list = pd.DataFrame()
        ecGEMyield_list = pd.DataFrame()

        # Calculate yield for growth_model
        with enz_model as growth_model:
            for glc_concentration in glc_concentration_list:
                ecGEMyield_list = calculate_yield(growth_model, use_substrate, substrate_name, glc_concentration,
                                                  columns, correspond_rxn, ecGEMyield_list)

        # Calculate yield for norm_model
        with norm_model as growth_model:
            for glc_concentration in glc_concentration_list:
                GEMyield_list = calculate_yield(growth_model, use_substrate, substrate_name, glc_concentration, columns,
                                                correspond_rxn, GEMyield_list)
        substrate_name = 'glucose'
        substrate_bound = 16
        obj_bound = 5
        secrate_bound = 16
        column_list = ['biomass', 'CO_2', 'acetate']
        y_axis_loc_list = ['left', 'right', 'right', 'right']
        color_list = generate_random_colors(len(column_list) * 2)

        fig = draw_overflow_fig(GEMyield_list, ecGEMyield_list, column_list, y_axis_loc_list, color_list,
                                substrate_name, substrate_bound, obj_bound, secrate_bound, overflow_result_figfile)
        ecGEMyield_list.to_csv(f"{figure_path}/ecGEMyield.csv", index=False)
        GEMyield_list.to_csv(f"{figure_path}/GEMyield.csv", index=False)


def draw_efficiency(figure_path, ecModel_output_file, reaction_kcat_MW_file, obj):
    if is_auto:
        reaction_kcat_MW = pd.read_csv(reaction_kcat_MW_file, index_col=0)
        reaction_kcat_MW = round(reaction_kcat_MW, 3)
        enz_model = get_enzyme_constraint_model(ecModel_output_file)
        enz_model.objective = obj
        glc_concentration_list = np.arange(0, 10, 0.5)
        efficiency_file = f"{figure_path}/efficiency_pfba.csv"
        trade_off_enzyme_efficiency_figfile = f"{figure_path}/trade_off_enzyme_efficiency.png"
        use_substrate = 'EX_co2_e'

        yield_cost_efficiency_df = get_yield_cost_efficiency(enz_model, glc_concentration_list, use_substrate, obj,
                                                             reaction_kcat_MW, efficiency_file)
        fig = draw_trade_off(yield_cost_efficiency_df, trade_off_enzyme_efficiency_figfile)
    else:
        # inputfiles
        reaction_kcat_MW = pd.read_csv(reaction_kcat_MW_file, index_col=0)
        reaction_kcat_MW = round(reaction_kcat_MW, 3)
        enz_model = get_enzyme_constraint_model(ecModel_output_file)
        # norm_model = cobra.io.json.load_json_model(ecModel_output_file)
        glc_concentration_list = np.arange(1, 16, 1)
        efficiency_file = f"{figure_path}/efficiency_pfba.csv"
        trade_off_enzyme_efficiency_figfile = f"{figure_path}/trade_off_enzyme_efficiency.png"
        use_substrate = 'EX_glc__D_e'

        yield_cost_efficiency_df = get_yield_cost_efficiency(enz_model, glc_concentration_list, use_substrate, obj,
                                                             reaction_kcat_MW, efficiency_file)
        fig = draw_trade_off(yield_cost_efficiency_df, trade_off_enzyme_efficiency_figfile)


def main(args):



    method = "EkiLLm"  # DLkcat AutoPACMEN EkiLLm
    ecGEM_path = f"{args.path}/analysis/get_kcat_mw_by_{method}/ecGEM"
    figure_path = f"{ecGEM_path}/figure"
    create_file(ecGEM_path)
    create_file(figure_path)

    if is_auto:
        obj = 'BIOMASS_Ec_SynAuto_1'  # BIOMASS_Ec_SynAuto_1 BIOMASS_Ec_SynMixo_1  BIOMASS_Ec_SynHetero_1 obj是反应，一般指向带有BIOMASS的
    else:
        obj = 'BIOMASS_Ec_iJO1366_core_53p95M'
        # obj = None
    reaction_kcat_MW_file = f"{args.path}/analysis/get_kcat_mw_by_{method}/reaction_kcat_MW.csv"
    # ecModel_output_file = f"{ecGEM_path}/ecGEM_irr_enz_constraint_etc.json"
    ecModel_output_file = f"{ecGEM_path}/Bayesian/best_ecGEM_overflow.json"
    # ecModel_output_file = f"{ecGEM_path}/ecGEM_irr_enz_constraint.json"
    draw_overflow(figure_path, ecModel_output_file, obj)

    # draw_efficiency(figure_path, ecModel_output_file, reaction_kcat_MW_file, obj)
    # draw_phpp(figure_path, ecModel_output_file, obj)

    # if :
    #     print()
    # else:
    #     growth_exp_file = "./data/growth_exp.csv"
    #     json_model_file = "./model/eciML1515.json"
    #     enz_model = get_enzyme_constraint_model(json_model_file)
    #     growth_exp = pd.read_csv(growth_exp_file, index_col=0)
    #
    #     growth_rate_diff_substrate_file = "./analysis/enz_model_growth_pfba.csv"
    #     ECMpy_diff_substate_result_figfile = "./analysis/ECMpy_diff_substate_result.png"
    #     diff_model_diff_substate_result_figfile = "./analysis/diff_model_diff_substate_result.png"


if __name__ == '__main__':
    # predict/E.coli
    # predict/Synechocystis sp
    # predict/iECDH1ME8569_1439
    parser.add_argument("--path", type=str,
                        default="../predict/iECDH1ME8569_1439")
    # parser.add_argument("--path", type=str,
    #                     default="../predict/Synechocystis sp")
    # parser.add_argument("--is_auto", type=bool, default=True)
    args = parser.parse_args()

    if "Synechocystis sp" in args.path:
        is_auto = True
    else:
        is_auto = False

    main(args=args)

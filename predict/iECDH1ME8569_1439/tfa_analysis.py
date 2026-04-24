#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修改版pyTFA教程 - 使用自定义SBML代谢网络文件

基于pytfa/tutorials/tutorial_basics.py，修改为使用用户提供的SBML文件
"""

import pytfa

from optlang.exceptions import SolverError
from cobra.flux_analysis import flux_variability_analysis
from cobra.io import read_sbml_model

from pytfa.io import load_thermoDB, \
                            read_lexicon, annotate_from_lexicon, \
                            read_compartment_data, apply_compartment_data

# 求解器设置
CPLEX = 'optlang-cplex'
GUROBI = 'optlang-gurobi'
GLPK = 'optlang-glpk'
solver = GLPK  # 使用GLPK作为默认求解器

def apply_solver_settings(model, solver_name=solver):
    """应用求解器设置"""
    model.solver = solver_name
    model.solver.configuration.tolerances.feasibility = 1e-9
    if solver_name == 'optlang_gurobi':
        model.solver.problem.Params.NumericFocus = 3
    model.solver.configuration.presolve = True

def main():
    """主函数"""
    print("=== pyTFA 自定义SBML网络分析 ===")

    # 1. 加载热力学数据库
    print("正在加载热力学数据库...")
    try:
        thermo_data = load_thermoDB('/home/zhangyangyu/pytfa/data/thermo_data.thermodb')
        print("✓ 热力学数据库加载成功")
    except FileNotFoundError:
        print("✗ 热力学数据库文件未找到，请检查路径")
        return
    except Exception as e:
        print(f"✗ 热力学数据库加载失败: {e}")
        return

    # 2. 加载SBML模型
    print("正在加载SBML代谢网络...")
    sbml_file = "/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/network.xml"
    try:
        cobra_model = read_sbml_model(sbml_file)
        print(f"✓ SBML模型加载成功")
        print(f"  模型ID: {cobra_model.id}")
        print(f"  代谢物数量: {len(cobra_model.metabolites)}")
        print(f"  反应数量: {len(cobra_model.reactions)}")
    except FileNotFoundError:
        print("✗ SBML文件未找到，请检查路径")
        return
    except Exception as e:
        print(f"✗ SBML模型加载失败: {e}")
        return

    # 3. 检查和设置目标函数
    print("正在检查生物量反应...")

    # 查找生物量反应
    biomass_reactions = [rxn for rxn in cobra_model.reactions if 'biomass' in rxn.id.lower()]

    if biomass_reactions:
        print(f"找到 {len(biomass_reactions)} 个生物量反应:")
        for rxn in biomass_reactions:
            print(f"  - {rxn.id}: {rxn.name}")

        # 使用默认的目标函数或第一个生物量反应
        if cobra_model.objective:
            # 从目标函数表达式中提取反应ID
            objective_str = str(cobra_model.objective)
            print(f"原始目标函数: {objective_str}")

            # 查找生物量反应ID
            for rxn in biomass_reactions:
                if rxn.id in objective_str:
                    biomass_rxn = rxn.id
                    break
            else:
                biomass_rxn = biomass_reactions[0].id

            print(f"提取的生物量反应ID: {biomass_rxn}")
        else:
            biomass_rxn = biomass_reactions[0].id
            print(f"设置目标函数为: {biomass_rxn}")
    else:
        print("警告: 未找到生物量反应，将使用模型的默认目标函数")
        biomass_rxn = None

    # 4. 创建TFA模型
    print("正在创建TFA模型...")
    try:
        mytfa = pytfa.ThermoModel(thermo_data, cobra_model)
        mytfa.name = f'TFA_{cobra_model.id}'

        # 设置目标函数
        if biomass_rxn:
            mytfa.objective = biomass_rxn

        print("✓ TFA模型创建成功")
    except Exception as e:
        print(f"✗ TFA模型创建失败: {e}")
        return

    # 5. 应用求解器设置
    print("正在配置求解器...")
    apply_solver_settings(mytfa)
    print(f"✓ 使用求解器: {solver}")

    # 6. 运行FBA (原始模型)
    print("\n=== FBA分析 ===")
    try:
        apply_solver_settings(cobra_model)
        fba_solution = cobra_model.optimize()
        fba_value = fba_solution.objective_value
        print(f"FBA解: {fba_value:.6f}")

        if fba_value <= 1e-6:
            print("警告: FBA解接近零，可能存在问题")
    except Exception as e:
        print(f"FBA优化失败: {e}")
        fba_value = 0
        return

    # 7. TFA转换和优化
    print("\n=== TFA分析 ===")
    try:
        print("正在进行TFA预处理...")
        mytfa.prepare()

        print("正在转换为TFA模型...")
        mytfa.convert()

        print("✓ TFA转换完成")

        # 显示模型信息
        mytfa.print_info()

        print("正在运行TFA优化...")
        tfa_solution = mytfa.optimize()
        tfa_value = tfa_solution.objective_value
        print(f"TFA解: {tfa_value:.6f}")

    except Exception as e:
        print(f"TFA优化失败: {e}")
        print("尝试放松热力学约束...")

        # 8. 约束松弛 (如果需要)
        try:
            from pytfa.optim.relaxation import relax_dgo

            if biomass_rxn and biomass_rxn in [rxn.id for rxn in mytfa.reactions]:
                mytfa.reactions.get_by_id(biomass_rxn).lower_bound = 0.5 * fba_value
                relaxed_model, slack_model, relax_table = relax_dgo(mytfa)

                print("约束松弛信息:")
                print(relax_table)

                mytfa = relaxed_model
                tfa_solution = mytfa.optimize()
                tfa_value = tfa_solution.objective_value
                print(f"松弛后TFA解: {tfa_value:.6f}")
            else:
                print("无法进行约束松弛：生物量反应未找到")
                return

        except Exception as relax_e:
            print(f"约束松弛失败: {relax_e}")
            return

    # 9. 结果比较
    print(f"\n=== 结果摘要 ===")
    print(f"模型: {cobra_model.id}")
    print(f"代谢物数量: {len(cobra_model.metabolites)}")
    print(f"反应数量: {len(cobra_model.reactions)}")
    print(f"FBA目标值: {fba_value:.6f}")
    print(f"TFA目标值: {tfa_value:.6f}")
    print(f"相对差异: {abs(fba_value-tfa_value)/max(abs(fba_value),1e-10)*100:.2f}%")

    if abs(tfa_value - fba_value) > 0.01:
        print("⚠️  热力学约束显著影响了目标值")
    else:
        print("✓ 热力学约束对目标值影响较小")

    # 10. 保存结果 (可选)
    try:
        output_file = f"/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/tfa_results.json"
        import json
        results = {
            'model_id': cobra_model.id,
            'num_metabolites': len(cobra_model.metabolites),
            'num_reactions': len(cobra_model.reactions),
            'fba_objective': float(fba_value),
            'tfa_objective': float(tfa_value),
            'solver': solver,
            'biomass_reaction': biomass_rxn
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到: {output_file}")

    except Exception as save_e:
        print(f"保存结果失败: {save_e}")

    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()
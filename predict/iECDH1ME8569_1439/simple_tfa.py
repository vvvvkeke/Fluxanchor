#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版pyTFA分析 - 使用自定义SBML文件

直接修改原教程，最少改动来使用SBML文件
"""

import pytfa

from optlang.exceptions import SolverError
from cobra.flux_analysis import flux_variability_analysis
from cobra.io import read_sbml_model

from pytfa.io import load_thermoDB

# 求解器设置
CPLEX = 'optlang-cplex'
GUROBI = 'optlang-gurobi'
GLPK = 'optlang-glpk'
solver = GLPK

# 加载数据
print("Loading thermo data...")
thermo_data = load_thermoDB('/home/zhangyangyu/pytfa/data/thermo_data.thermodb')
print("Done!")

# 加载自定义SBML模型
print("Loading custom SBML model...")
cobra_model = read_sbml_model('/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/network.xml')
print(f"Model loaded: {cobra_model.id}")
print(f"Metabolites: {len(cobra_model.metabolites)}")
print(f"Reactions: {len(cobra_model.reactions)}")

# 查找生物量反应
biomass_reactions = [rxn for rxn in cobra_model.reactions if 'biomass' in rxn.id.lower()]
if biomass_reactions:
    biomass_rxn = biomass_reactions[0].id  # 使用第一个生物量反应
    print(f"Using biomass reaction: {biomass_rxn}")
else:
    print("No biomass reaction found, using default objective")
    biomass_rxn = None

# 创建TFA模型
mytfa = pytfa.ThermoModel(thermo_data, cobra_model)
mytfa.name = 'custom_tfa_model'
mytfa.solver = solver

if biomass_rxn:
    mytfa.objective = biomass_rxn

# 求解器设置
def apply_solver_settings(model, solver_name=solver):
    model.solver = solver_name
    model.solver.configuration.tolerances.feasibility = 1e-9
    model.solver.configuration.presolve = True

apply_solver_settings(mytfa)
apply_solver_settings(cobra_model)

## FBA
print("\n=== FBA Analysis ===")
fba_solution = cobra_model.optimize()
fba_value = fba_solution.objective_value
print(f"FBA Solution found: {fba_value:.6f}")

## TFA conversion
print("\n=== TFA Analysis ===")
try:
    print("TFA conversion starting...")
    mytfa.prepare()
    mytfa.convert()
    print("TFA conversion completed!")

    ## Model info
    mytfa.print_info()

    ## TFA Optimization
    print("TFA optimization starting...")
    tfa_solution = mytfa.optimize()
    tfa_value = tfa_solution.objective_value
    print(f"TFA Solution found: {tfa_value:.6f}")

    # 结果对比
    print(f"\n=== Results Comparison ===")
    print(f"FBA objective: {fba_value:.6f}")
    print(f"TFA objective: {tfa_value:.6f}")
    print(f"Difference: {abs(fba_value - tfa_value):.6f}")
    print(f"Relative difference: {abs(fba_value - tfa_value) / max(abs(fba_value), 1e-10) * 100:.2f}%")

except Exception as e:
    print(f"TFA conversion/optimization failed: {e}")
    print("This might be due to:")
    print("1. Missing thermodynamic data for some metabolites")
    print("2. Version compatibility issues")
    print("3. Model complexity")

    print("\nTrying relaxation approach...")
    try:
        from pytfa.optim.relaxation import relax_dgo

        if biomass_rxn:
            mytfa.reactions.get_by_id(biomass_rxn).lower_bound = 0.5 * fba_value
            relaxed_model, slack_model, relax_table = relax_dgo(mytfa)

            print("Relaxation successful!")
            print("Relaxation table:")
            print(relax_table)

            tfa_solution = relaxed_model.optimize()
            tfa_value = tfa_solution.objective_value
            print(f"Relaxed TFA solution: {tfa_value:.6f}")

        else:
            print("Cannot perform relaxation: no biomass reaction identified")

    except Exception as relax_e:
        print(f"Relaxation also failed: {relax_e}")
        print("\nThis suggests compatibility issues between:")
        print("- pyTFA version")
        print("- COBRApy version")
        print("- Other dependencies (sympy, optlang, etc.)")

print("\nAnalysis completed!")
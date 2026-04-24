import pandas as pd
import sys
sys.path.append(r'./script/')

from script.ECMpy_function import *

if __name__ == '__main__':
    raw_proteome = pd.read_csv('predict/iECDH1ME8569_1439/proteomics_with_uniprot.csv')
    ref_proteome_pgk = 0.000321192
    total = []
    pgk_abu = [0.003, 0.0031, 0.0032, 0.0036]  # 0.003 0.0031 0.0032
    # total = ref_proteome_pgk / gpk_abu

    raw_proteome['NQ381 (400)'] = raw_proteome['NQ381 (400)'].str.replace('%', '').astype(float) / 100
    raw_proteome['NQ381 (500)'] = raw_proteome['NQ381 (500)'].str.replace('%', '').astype(float) / 100
    raw_proteome['NQ381 (800)'] = raw_proteome['NQ381 (800)'].str.replace('%', '').astype(float) / 100
    raw_proteome['NCM3722'] = raw_proteome['NCM3722'].str.replace('%', '').astype(float) / 100

    for relative in pgk_abu:
        total.append(ref_proteome_pgk / relative)

    raw_proteome['NQ381 (400)_absolute'] = raw_proteome['NQ381 (400)'] * total[0]
    raw_proteome['NQ381 (500)_absolute'] = raw_proteome['NQ381 (500)'] * total[1]
    raw_proteome['NQ381 (800)_absolute'] = raw_proteome['NQ381 (800)'] * total[2]
    raw_proteome['NCM3722_absolute'] = raw_proteome['NCM3722'] * total[3]

    # raw_proteome.to_csv("test.csv", index=False)

    # print(raw_proteome)

    # 求解fba
    method = "EkiLLm"  # DLkcat AutoPACMEN EkiLLm

    ecGEM_path = f"predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_{method}/ecGEM"
    obj = 'BIOMASS_Ec_iJO1366_core_53p95M'
    ecModel_output_file = f"{ecGEM_path}/ecGEM_irr_enz_constraint.json"
    enz_model = get_enzyme_constraint_model(ecModel_output_file)
    solution = cobra.flux_analysis.pfba(enz_model)

    print(solution)

import cobra
import pandas as pd
import json

# ==========================================
# 1. 扩充后的调控规则库 (Regulatory Rules)
# ==========================================
# 包含大肠杆菌主要的代谢调控因子。
# 逻辑说明:
# - 'activated': TF 激活的基因。TF 敲除 -> 基因不再表达 -> 反应关闭 (Bounds=0)。
# - 'repressed': TF 抑制的基因。TF 敲除 -> 基因过表达。在 FBA 中通常不做处理(因为FBA是求最大值)，除非需要模拟毒性。
REGULATORY_RULES = {
    # --- 全局碳代谢调控 ---
    'Cra': { # 糖异生/糖酵解开关
        'activated': ['pck', 'fbp', 'pps', 'aceA', 'aceB', 'iclR', 'acnB'], 
        'repressed': ['pfkA', 'pfkB', 'pykF', 'pykA', 'edd', 'eda', 'gapA', 'eno', 'mglB', 'ptsH', 'ptsI', 'crr']
    },
    'Crp': { # cAMP受体蛋白，广泛激活
        'activated': ['ptsG', 'ptsH', 'ptsI', 'crr', 'cyoA', 'cyoB', 'sdhA', 'sdhB', 'sucA', 'sucB', 'mdh', 'fumA', 'gltA', 'acnB', 'icd', 'pck', 'acs', 'actP', 'manX', 'manY', 'manZ', 'malE', 'malF', 'malG', 'malK', 'lamB', 'araE', 'araF', 'araG', 'araH'],
        'repressed': ['mtlA', 'mtlD', 'rbsD', 'rbsA', 'rbsC', 'rbsB']
    },
    'Mlc': { # 抑制葡萄糖摄取
        'activated': [],
        'repressed': ['ptsG', 'manX', 'manY', 'manZ', 'malT']
    },
    
    # --- 呼吸与氧气响应 ---
    'ArcA': { # 厌氧主要调节因子，抑制好氧呼吸
        'activated': ['pflB', 'cydA', 'cydB', 'focA'],
        'repressed': ['gltA', 'acnB', 'icd', 'sucA', 'sucB', 'sucC', 'sucD', 'sdhA', 'sdhB', 'sdhC', 'sdhD', 'fumA', 'mdh', 'cyoA', 'cyoB', 'cyoC', 'cyoD', 'fadA', 'fadB', 'fadE', 'fadL', 'fadD', 'glcDEF']
    },
    'Fnr': { # 厌氧主开关
        'activated': ['narG', 'narH', 'narI', 'narJ', 'dmsA', 'dmsB', 'dmsC', 'frdA', 'frdB', 'frdC', 'frdD', 'aspA', 'menA', 'nirB', 'nirD'],
        'repressed': ['cyoA', 'cyoB', 'cyoC', 'cyoD', 'ndh', 'cydA', 'cydB', 'sodA']
    },
    'OxyR': { # 氧化应激
        'activated': ['katG', 'ahpC', 'ahpF', 'gor', 'grxA', 'trxC'],
        'repressed': []
    },
    'SoxS': { # 超氧化物应激
        'activated': ['zwf', 'fpr', 'nfo', 'sodA', 'acrA', 'acrB'],
        'repressed': []
    },

    # --- 氮、磷、硫代谢 ---
    'NtrC': { # 氮源
        'activated': ['glnA', 'glnK', 'amtB', 'nac'],
        'repressed': []
    },
    'PhoB': { # 磷酸盐饥饿
        'activated': ['pstS', 'pstA', 'pstB', 'pstC', 'phoA', 'phoE', 'ugpB', 'ugpA', 'ugpC', 'phnC', 'phnD', 'phnE'],
        'repressed': []
    },
    'CysB': { # 硫/半胱氨酸合成
        'activated': ['cysK', 'cysM', 'cysA', 'cysP', 'cysU', 'cysW', 'cysD', 'cysN', 'cysC', 'cysH', 'cysI', 'cysJ'],
        'repressed': []
    },
    'PurR': { # 嘌呤合成抑制
        'activated': [],
        'repressed': ['purF', 'purL', 'purM', 'purN', 'purH', 'purD', 'purE', 'purK', 'purT', 'guaA', 'guaB', 'glyA', 'glnB']
    },
    'ArgR': { # 精氨酸合成抑制
        'activated': [],
        'repressed': ['argA', 'argB', 'argC', 'argD', 'argE', 'argF', 'argG', 'argH', 'argI', 'carA', 'carB', 'artJ', 'artM', 'artP', 'artQ']
    },

    # --- 碳源特定调控 ---
    'FadR': { # 脂肪酸代谢
        'activated': ['fabA', 'fabB', 'icd'], 
        'repressed': ['fadD', 'fadL', 'fadE', 'fadB', 'fadA', 'fadH', 'fadI', 'fadJ']
    },
    'IclR': { # 乙醛酸循环抑制
        'activated': [],
        'repressed': ['aceA', 'aceB', 'aceK']
    },
    'GlpR': { # 甘油代谢
        'activated': [],
        'repressed': ['glpD', 'glpT', 'glpA', 'glpB', 'glpC', 'glpK', 'glpF', 'glpX']
    },
    'GcvA': { # 甘氨酸裂解
        'activated': ['gcvT', 'gcvH', 'gcvP'],
        'repressed': []
    },
    'NagC': { # NAG (N-乙酰葡糖胺)
        'activated': [],
        'repressed': ['nagE', 'nagB', 'nagA', 'nagD', 'nagC']
    },
    'PrpR': { # 丙酸代谢
        'activated': ['prpB', 'prpC', 'prpD', 'prpE'],
        'repressed': []
    },
    'FhlA': { # 甲酸氢解酶
        'activated': ['hycA', 'hycB', 'hycC', 'hycD', 'hycE', 'hycF', 'hycG', 'hycH', 'hycI'],
        'repressed': []
    },
    'LldR': { # 乳酸
        'activated': [],
        'repressed': ['lldP', 'lldR', 'lldD']
    },
    'DcuR': { # C4二羧酸
        'activated': ['dcuB', 'fumB'],
        'repressed': []
    },
    'BetI': { # 胆碱/甜菜碱
        'activated': [],
        'repressed': ['betA', 'betB', 'betT']
    },
    'CaiF': { # 肉碱
        'activated': ['caiT', 'caiA', 'caiB', 'caiC', 'caiD', 'caiE'],
        'repressed': []
    },
    
    # --- 硝酸盐/亚硝酸盐 ---
    'NarL': {
        'activated': ['narG', 'narH', 'narI', 'narJ', 'fdnG', 'fdnH', 'fdnI'],
        'repressed': ['frdA', 'frdB', 'frdC', 'frdD']
    },
    
    # --- 其它 ---
    'PdhR': { # 丙酮酸脱氢酶
        'activated': [],
        'repressed': ['aceE', 'aceF', 'lpd', 'ndh', 'cyoA', 'cyoB']
    },
    'CytR': { # 核苷转运
        'activated': [],
        'repressed': ['udp', 'ts', 'nupC', 'nupG']
    },
    'TyrR': { # 酪氨酸
        'activated': ['mtr', 'tyrP'],
        'repressed': ['aroF', 'tyrA', 'aroG']
    }
}

# 辅助函数: 建立 基因名 -> 模型ID 的映射
def create_gene_lookup(model):
    name_to_id = {}
    for gene in model.genes:
        # 1. 映射 Name
        if gene.name:
            name_to_id[gene.name.lower()] = gene.id
        # 2. 映射同义词 (Synonyms)
        if 'refseq_synonym' in gene.annotation:
            syns = gene.annotation['refseq_synonym']
            if isinstance(syns, str): syns = [syns]
            for s in syns:
                name_to_id[s.lower()] = gene.id
        # 3. 映射 ID 本身
        name_to_id[gene.id.lower()] = gene.id
    return name_to_id

def simulate_regulatory_knockouts(model_path, mutants_csv_path, output_csv):
    print(f"Loading model: {model_path} ...")
    model = cobra.io.load_json_model(model_path)
    model.solver = 'glpk' # 如果报错，可尝试 'cplex' 或 'gurobi'
    
    name_to_model_id = create_gene_lookup(model)
    print(f"Model loaded. Total genes: {len(model.genes)}")

    # 1. 计算野生型 (WT) 生长
    sol_wt = model.optimize()
    wt_growth = sol_wt.objective_value
    print(f"WT Growth Rate: {wt_growth:.4f}")

    # 2. 读取突变列表
    df_raw = pd.read_csv(mutants_csv_path, header=None)
    tf_mutants = []
    
    # 解析 CSV (提取 'Transcription factor mutants' 部分)
    current_cat = ""
    for idx, row in df_raw.iterrows():
        val = str(row[0]).strip()
        if "Transcription factor" in val: current_cat = "TF"; continue
        if "Sigma" in val: break 
        if current_cat == "TF" and len(val) < 15 and " " not in val and not val[0].isdigit():
            tf_mutants.append(val)
            
    print(f"Processing {len(tf_mutants)} TF mutants...")

    # 3. 模拟循环
    results = []
    for tf_name in tf_mutants:
        # 使用上下文管理器，保证每次模拟后恢复模型原状
        with model:
            rule = REGULATORY_RULES.get(tf_name, None)
            closed_genes_list = []
            
            if rule:
                # 获取该 TF 激活的基因列表
                targets = rule.get('activated', [])
                
                for t_name in targets:
                    gid = name_to_model_id.get(t_name.lower())
                    if gid:
                        # 找到模型中的基因对象
                        gene_obj = model.genes.get_by_id(gid)
                        
                        # 执行基因敲除 (Gene Knockout)
                        # cobra 会自动处理 GPR 关系 (例如: (A and B) -> 敲除A则反应关; (A or B) -> 敲除A则反应开)
                        gene_obj.knockout()
                        
                        closed_genes_list.append(t_name)
            
            # 运行 FBA
            sol = model.optimize()
            mutant_growth = sol.objective_value if sol.status == 'optimal' else 0.0
            
            # 记录数据
            results.append({
                'Mutant': tf_name,
                'WT_Growth': wt_growth,
                'Mutant_Growth': mutant_growth,
                'Relative_Growth': mutant_growth / wt_growth if wt_growth > 0 else 0,
                'Regulation_Implemented': 'Yes' if rule else 'No',
                'Targets_Knocked_Out': "; ".join(closed_genes_list) if closed_genes_list else "None (or Repressor only)"
            })

    # 4. 保存结果
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False)
    print(f"Done! Results saved to {output_csv}")
    print(df_res.head(10))

# 运行命令
if __name__ == "__main__":
    simulate_regulatory_knockouts('/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/get_kcat_mw_by_EkiLLm/ecGEM/ecGEM_irr_enz_constraint.json', 'msb20119-s2.csv', 'regulatory_simulation_results.csv')
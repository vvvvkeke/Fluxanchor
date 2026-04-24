import pandas as pd  
import os  

# 获取当前文件夹路径  
current_folder_path = os.path.dirname(os.path.abspath(__file__))   
file_path = os.path.join(current_folder_path, 'rubiscos_with_species.csv')  
output_file = os.path.join(current_folder_path, 'rubiscos_with_SMILE.csv')  

if __name__ == '__main__':  
    # 读取主数据文件  
    data = pd.read_csv(file_path, sep=",")  

    # 反应物以及生成物
    substrate = "Ribulose-1,5-bisphosphate"
    product = "3-phosphoglycerate"
    reaction = "CO2 + Ribulose-1,5-bisphosphate = 2 3-phosphoglycerate"

    #  SMILES 字符串  
    ribulose_1_5_bisphosphate = 'O=C(C(O)COP(=O)(O)O)[C@@H](O)COP(=O)(O)O'  
    phosphoglycerate_3 = 'C(C(C(=O)O)O)OP(=O)(O)O'  
    Reaction = 'O=C=O + O=P(O)(OCC(=O)[C@H](O)[C@H](O)COP(=O)(O)O)O -----> C(C(C(=O)O)O)OP(=O)(O)O'

    # 最佳温度/ph/类型
    ph = 8
    temperature = 30
    type = "mutant"

    # 添加
    data['substrate'] = substrate
    data['product'] = product
    data['reaction'] = reaction
    data['Ribulose 1,5-bisphosphate'] = ribulose_1_5_bisphosphate  
    data['3-phosphoglycerate'] = phosphoglycerate_3  
    data['Reaction'] = Reaction
    data['ph'] = ph
    data['temperature'] = temperature
    data['type'] = type
    data['uniprot_id'] = ['sequence_' + str(i + 1) for i in range(len(data))]  

    # 保存结果到新文件  
    data.to_csv(output_file, index=False)  
    print(f"Results saved to {output_file}")
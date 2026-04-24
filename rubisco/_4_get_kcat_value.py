import pandas as pd  
import os  

# 获取当前文件夹路径  
current_folder_path = os.path.dirname(os.path.abspath(__file__))   
file_path = os.path.join(current_folder_path, 'rubiscos_with_SMILE.csv')  
output_file = os.path.join(current_folder_path, 'rubiscos_with_all_info.csv')  

def calculate_kcat(activity, enzyme_concentration_mg_per_l, molecular_weight):
    """计算 kcat 值"""  
    try:  
        # 将浓度从 mg/L 转换为 mol/L  
        enzyme_concentration_mol_per_l = (enzyme_concentration_mg_per_l / 1000) / molecular_weight  
        kcat = activity / enzyme_concentration_mol_per_l  
        return kcat  
    except ZeroDivisionError:  
        return None  # 防止分母为 0 的情况  

if __name__ == '__main__':  
    # 读取主数据文件  
    data = pd.read_csv(file_path, sep=",")  

    # 固定酶浓度（mg/L） 
    enzyme_concentration_mg = 1000000

    # 计算 kcat 并存储到新列  
    data["kcat"] = data.apply(  
        lambda row: calculate_kcat(  
            row["Rate mean [s-1] "],   
            enzyme_concentration_mg,   
            row["Molecular Weight by protein"]  
        ),   
        axis=1  
    )  

    # 保存结果到新文件  
    data.to_csv(output_file, index=False)  
    print(f"Results saved to {output_file}")
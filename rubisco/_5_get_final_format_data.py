import pandas as pd  
import os  

# 获取当前文件夹路径  
current_folder_path = os.path.dirname(os.path.abspath(__file__))   
file_path = os.path.join(current_folder_path, 'rubiscos_with_all_info.csv')  
output_file = os.path.join(os.path.dirname(current_folder_path), 'predict/rubisco/dataset/input.tsv')  

if __name__ == '__main__':  
    # 读取主数据文件  
    data = pd.read_csv(file_path, sep=",")  

   # 选择需要的列并重命名  
    filtered_data = data[[  
        "Species Name",  # 对应 organisms  
        "Protein seq",   # 对应 sequence  
        "substrate",  
        "product",  
        "reaction",  
        "type",          # 对应 type  
        "ph",  
        "temperature",    #  对应 temperature
        "uniprot_id",    # 对应 uniprot_id  
        "Ribulose 1,5-bisphosphate",  # 对应 smiles_substrate  
        "3-phosphoglycerate",         # 对应 smiles_product  
        "Reaction",                    # 对应 smiles_reaction  
        "kcat"  # 对应kcat
    ]]  

    # 重命名列  
    filtered_data.columns = [  
        "organisms",  
        "sequence",  
        "substrate",  
        "product",  
        "reaction",  
        "type",  
        "ph",  
        "temperature",  
        "uniprot_id",  
        "smiles_substrate",  
        "smiles_product",  
        "smiles_reaction",
        "kcat"
    ]  

    # 保存结果到新文件  
    filtered_data.to_csv(output_file, index=False, sep="\t")  
    print(f"Filtered data saved to {output_file}")  
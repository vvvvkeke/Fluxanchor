import pandas as pd  
import os  

# 获取当前文件夹路径  
current_folder_path = os.path.dirname(os.path.abspath(__file__))   
file_path = os.path.join(current_folder_path, 'rubiscos_with_mw.csv')  # 主数据文件路径  
map_file_path = os.path.join(current_folder_path,"rubiscos.map.name")  # 物种名映射文件路径 

output_file = os.path.join(current_folder_path, 'rubiscos_with_species.csv')  # 输出文件路径  

if __name__ == '__main__':  
    # 读取主数据文件  
    data = pd.read_csv(file_path, sep=",")  
    
    # 读取物种名映射文件  
    species_map = pd.read_csv(map_file_path, sep="\t", header=None, names=["Identifier", "Species Name"])  

    # 合并数据，根据 Identifier 匹配  
    merged_data = pd.merge(data, species_map, on="Identifier", how="left")  

    merged_data['Species Name'] = merged_data['Species Name'].fillna('None')   
    merged_data['Species Name'] = merged_data['Species Name'].str.replace(r'\[|\]', '', regex=True) 

    # 保存结果到新文件  
    merged_data.to_csv(output_file, index=False)  
    print(f"Results saved to {output_file}")
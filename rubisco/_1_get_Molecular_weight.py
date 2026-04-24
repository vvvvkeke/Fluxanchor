import pandas as pd  
from Bio.SeqUtils import molecular_weight 
import os

current_folder_path = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(current_folder_path,'rubiscos.csv')
output_file = os.path.join(current_folder_path,'rubiscos_with_mw.csv')

# 计算分子量  
def calculate_molecular_weight_by_protein(protein_seq):  
    try:  
        return molecular_weight(protein_seq, seq_type="protein", monoisotopic=False)  
    except Exception as e:  
        print(f"Error calculating molecular weight for protein sequence: {protein_seq}\n{e}")  
        return None  

if __name__ == '__main__':
    data = pd.read_csv(file_path, sep=",")
    data["Molecular Weight by protein"] = data["Protein seq"].apply(calculate_molecular_weight_by_protein)
        
    # 删除全为空值的列  
    data = data.dropna(axis=1, how='all')  

    data.to_csv(output_file, index=False)  
    print(f"Results saved to {output_file}")  


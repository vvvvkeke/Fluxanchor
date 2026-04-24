import pandas as pd
import os
if __name__ == '__main__':
    organisms = "Rhodospirillum rubrum"
    ph = "8"
    temperature = "30"
    types = "mutant"
    substrate = "Ribulose-1,5-bisphosphate"
    product = "3-phosphoglycerate"
    reaction = "CO2 + Ribulose-1,5-bisphosphate = 2 3-phosphoglycerate"
    smiles_substrate = "O=C(C(O)COP(=O)(O)O)[C@@H](O)COP(=O)(O)O"
    smiles_product = "C(C(C(=O)O)O)OP(=O)(O)O"
    smiles_reaction = "O=C=O + O=P(O)(OCC(=O)[C@H](O)[C@H](O)COP(=O)(O)O)O -----> C(C(C(=O)O)O)OP(=O)(O)O"
    seq_df = pd.read_csv("/home/zhangyangyu/kcat_km_predict/predict/rubisco_km/changed_site.tsv")
    km_value_df = pd.read_csv("/home/zhangyangyu/kcat_km_predict/predict/rubisco_km/41586_2024_8455_MOESM4_ESM.csv")
    df_test = pd.DataFrame({
        'organisms': [],
        'ph': [],
        'temperature': [],
        'type': [],
        'substrate': [],
        'product': [],
        'reaction': [],
        'smiles_substrate': [],
        'smiles_product': [],
        'smiles_reaction': [],
        'uniprot_id': [],
        'sequence': [],
        'km': []
    })
    for index, row in seq_df.iterrows():
        sequence = row.loc["sequence"]
        km_value = km_value_df.iloc[index]["Km_median"]
        id = "mutant" + km_value_df.iloc[index]["mutant"]

        new_row = pd.DataFrame({
            'organisms': [organisms],
            'ph': [ph],
            'temperature': [temperature],
            'type': [types],
            'substrate': [substrate],
            'product': [product],
            'reaction': [reaction],
            'smiles_substrate': [smiles_substrate],
            'smiles_product': [smiles_product],
            'smiles_reaction': [smiles_reaction],
            'uniprot_id': [id],
            'sequence': [sequence],
            'km': [km_value]
        })
        df_test = pd.concat([df_test, new_row], ignore_index=True)

    df_test.to_csv("/home/zhangyangyu/kcat_km_predict/predict/rubisco_km/input.tsv", sep="\t", index=False)

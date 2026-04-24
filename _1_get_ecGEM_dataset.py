import argparse
import cobra
import datetime
import pandas as pd
import subprocess
import re
# from script.ECMpy_function import *
import sys

sys.path.append(r'./script/')

from script.ECMpy_function import *

import requests
import json
from bs4 import BeautifulSoup

from ecGEM_utils import update_dict, get_reaciton_bigg_id, convert_to_irreversible, isoenzyme_split
from bioservices import *

parser = argparse.ArgumentParser()


# def get_model_protein_sequence_and_mass(model, subbnumdf, prodf_file):
#     '''
#     This function is used to get the protein sequence and mass of the model.

#     Arguments
#     ----------
#     * model: cobra.Model ~ A Model object which will be modified in place.
#     * subbnumdf: pandas.DataFrame ~ DataFrame containing enzyme subunit information.
#     * prodf_file: str ~ File path to save the resulting DataFrame as a CSV file.

#     Returns
#     ----------
#     * prodf: pandas.DataFrame ~ DataFrame with protein information.
#     '''
#     genelist = []
#     prolist = []
#     enzyme_unit_number = subbnumdf
#     for gene in model.genes:
#         if 'uniprot' in gene.annotation:
#             pro = gene.annotation['uniprot']
#             genelist.append(gene.id)
#             prolist.append(str(pro))
#         # else:
#         #     get_uniprot_id(gene.annotation["refseq_old_locus_tag"])
#     prodf = pd.DataFrame(columns=['geneid', 'pro'])
#     prodf['geneid'] = genelist
#     prodf['pro'] = prolist

#     def fetch_protein_data(row):
#         try:
#             query = str(row)
#             uniprot_query_url = f"https://rest.uniprot.org/uniprotkb/search?query=accession:{query}&format=tsv&fields=accession,sequence"
#             uniprot_data = requests.get(uniprot_query_url).text.split("\n")[1].split("\t")[1]
#             return uniprot_data
#         except:
#             return None

#     prodf['aaseq'] = prodf['pro'].apply(fetch_protein_data)
#     prodf.dropna(subset=['aaseq'], inplace=True)
#     prodf['mass'] = prodf['aaseq'].apply(lambda seq: ProteinAnalysis(seq, monoisotopic=False).molecular_weight())
#     prodf['subunitmass'] = prodf.apply(
#         lambda row: row['mass'] * int(enzyme_unit_number.loc[row['pro'], 'subunitnumber'])
#         if row['pro'] in enzyme_unit_number.index else row['mass'], axis=1)

#     prodf.to_csv(prodf_file, index=False)
#     return prodf


#

def kcat_mw_calculation(ouputdf, metdf):
    '''
    Make the kcat_mw input file for ecGEM construction.

    Arguments:
    * DLouputdf: pandas.DataFrame - inputdf from DLKCAT.
    * metdf: pandas.DataFrame - DataFrame containing metabolite, reaction, gene, similes, prosequence, mass information.
    '''
    ouputdf.reset_index(drop=True, inplace=True)
    metdf.reset_index(drop=True, inplace=True)

    # Combine DLouputdf and metdf based on index
    outputdf_rex = pd.concat([metdf, ouputdf['pred_kcats_inverse_10']], axis=1)

    # Remove rows with 'None' in Kcat value
    outputdf_rex = outputdf_rex[outputdf_rex['pred_kcats_inverse_10'] != 'None']

    # Convert Kcat value to float
    outputdf_rex['pred_kcats_inverse_10'] = outputdf_rex['pred_kcats_inverse_10'].astype(float)

    # Sort by Kcat value and keep only the first occurrence of each reaction
    outputdf_rex = outputdf_rex.sort_values('pred_kcats_inverse_10', ascending=False).drop_duplicates(
        subset=['reactions'], keep='first')

    # Calculate kcat_mw
    outputdf_rex['kcat_mw'] = outputdf_rex['pred_kcats_inverse_10'] * 3600 * 1000 / outputdf_rex['totalmass']

    # Prepare reaction_kcat_mw DataFrame
    reaction_kcat_mw = pd.DataFrame()
    reaction_kcat_mw['reactions'] = outputdf_rex['reactions']
    reaction_kcat_mw['data_type'] = 'ours'
    reaction_kcat_mw['kcat'] = outputdf_rex['pred_kcats_inverse_10']
    reaction_kcat_mw['MW'] = outputdf_rex['mass']
    reaction_kcat_mw['kcat_MW'] = outputdf_rex['kcat_mw']
    reaction_kcat_mw.reset_index(drop=True, inplace=True)

    print('reaction_kcat_mw generated')
    return reaction_kcat_mw


def combine_reactions_simles_sequence(metdf, smilesdf, prodf, comdf_file):
    ''' Note: the incoming metdf is actually spdf.
    This function is used to combine the reaction--substrate--gene--protein_sequnce--mass.

    Arguments:
    * metdf: metabolites_reactions_gpr
    * similesdf: inchkeydf
    * prodf: prodf
    '''
    # Create a dictionary mapping met names to Canonical SMILES for later lookup.
    metname_to_smiles = dict(zip(smilesdf['metname'], smilesdf['Canonical_SMILES']))

    # Set the index of prodf to 'geneid'
    prodf.set_index('geneid', inplace=True)

    for index, row in metdf.iterrows():  # For each metabolite reaction, access metabolite names, genes, and GPR.
        metname = row['metabolitestotal']
        genes = row['genes']
        gpr = row['gpr']

        if metname in metname_to_smiles:  # If the metabolite exists in the dictionary, store its SMILES.
            metdf.loc[index, 'similes'] = metname_to_smiles[metname]

        if genes in prodf.index:  # If the gene is present, attach protein sequence and mass info.
            metdf.loc[index, 'prosequence'] = prodf.loc[genes, 'aaseq']
            metdf.loc[index, 'mass'] = prodf.loc[genes, 'mass']
            metdf.loc[index, 'uniprot_id'] = prodf.loc[genes, 'pro']

        totalmass = 0.0
        genelist = gpr.split(' and ')
        for gene in genelist:  # Split the GPR string and sum subunit masses for each gene.
            if gene in prodf.index:
                totalmass += prodf.loc[gene, 'subunitmass']

        metdf.loc[index, 'totalmass'] = totalmass

    # Reset the index of prodf
    prodf.reset_index(inplace=True)

    metdf.to_csv(comdf_file, index=False)
    return metdf


def compound_split(compounds: list, metdf_name: pd.DataFrame, metname_to_smiles: dict, name_to_smiles_dict: dict):
    met_dict = metdf_name.set_index('met')['name'].to_dict()
    normal_compounds = []
    smiles_compounds = []
    coffs = []
    for compound in compounds:
        compound = compound.strip()
        if str(compound).endswith("_c"):  # Remove cytosolic/extracellular/intermembrane suffixes
            compound = str(compound).split('_c')[0]
        elif str(compound).endswith("_e"):
            compound = str(compound).split('_e')[0]
        elif str(compound).endswith("_l"):
            compound = str(compound).split('_l')[0]
        elif str(compound).endswith("_p"):
            compound = str(compound).split('_p')[0]
        elif str(compound).endswith("_u"):
            compound = str(compound).split('_u')[0]
        elif str(compound).endswith("_x"):
            compound = str(compound).split('_x')[0]
        elif str(compound).endswith("_y"):
            compound = str(compound).split('_y')[0]
        else:
            compound = str(compound)
        if met_dict.__contains__(compound):  # Replace BiGG IDs with standard chemical names; some start with digits.
            normal_compounds.append(met_dict[compound])
            coffs.append("1")
            if metname_to_smiles.__contains__(compound) and metname_to_smiles[compound] != "noinchikey_inpubchem":  # If BiGG has a SMILES entry (via InChIKey), use it.
                smiles_compounds.append(metname_to_smiles[compound])
            else:  # Otherwise fallback to the name_to_smiles dictionary.
                if name_to_smiles_dict.__contains__(met_dict[compound]) and (name_to_smiles_dict[
                    met_dict[compound]] != "None"):
                    smiles_compounds.append(name_to_smiles_dict[met_dict[compound]])
                else:
                    # print(met_dict[compound])  Choline sulfate
                    smiles_compounds.append("None")
            continue

        # print(compound)
        comp = re.sub(r'^[\d.]+', '', compound).strip()  # Remove coefficients.
        if met_dict.__contains__(comp):  # Replace BiGG ID with formatted name after removing the coefficient.
            normal_compounds.append(met_dict[comp])
            if metname_to_smiles[comp] != "noinchikey_inpubchem":  # After removing the coefficient, check BiGG for SMILES again.
                smiles_compounds.append(metname_to_smiles[comp])
            else:  # Otherwise look up the dictionary.
                if name_to_smiles_dict.__contains__(met_dict[comp]) and name_to_smiles_dict[met_dict[comp]] != "None":
                    smiles_compounds.append(name_to_smiles_dict[met_dict[comp]])
                else:
                    # print(met_dict[comp])
                    smiles_compounds.append("None")
        # Remove leading numeric coefficients (including decimals).
        match = re.match(r'^[\d.]+', compound)  # Match the leading number/decimal.
        if match is None:
            # normal_compounds.append(met_dict[comp])
            # smiles_compounds.append(metname_to_smiles[comp])
            coffs.append("1")
        else:
            coff = re.match(r'^[\d.]+', compound)[0]
            # normal_compounds.append(coff + " " + comp)
            coffs.append(coff)

    # return " + ".join(normal_compounds), " + ".join(smiles_compounds)
    return normal_compounds, smiles_compounds, coffs


def get_main_product(bigg_product, normal_product, smiles_product,
                     currency_metabolites, name_without_smiles, name_to_smiles_dict):
    for index, s in enumerate(smiles_product):  # First pass: skip energy-currency metabolites.
        compound = re.sub(r'^[\d.]+', '', bigg_product[index].strip()).strip()
        if compound not in currency_metabolites and compound != "h_u":  # If not a currency metabolite, treat as candidate.
            if smiles_product[index] != "None" and smiles_product[index] != "noinchikey" and smiles_product[index] != "nosmiles":  # Non-empty SMILES => important product.
                return normal_product[index], smiles_product[index]
            # elif normal_product[index] not in name_without_smiles:
            #     print(normal_product[index])  # 05

    for index, s in enumerate(smiles_product):
        if smiles_product[index] != "None":  # Fallback: first non-empty SMILES wins.
            return normal_product[index], smiles_product[index]
        elif normal_product[index] not in name_without_smiles:
            print(normal_product[index])

    return "None", "None"


def chemical_name_to_smiles(chemical_name):
    try:
        time.sleep(2.0)
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chemical_name}/property/CanonicalSMILES/TXT'
        response = requests.get(url)
        if response.status_code != 200:
            return "None"
        else:
            smiles = response.content.splitlines()[0].decode()
            return smiles
    except:
        return "None"


def smilesdf_from_model_annotations(model):
    def _strip_compartment(met_id):
        for suffix in ("_c", "_e", "_l", "_p", "_u", "_x", "_y"):
            if met_id.endswith(suffix):
                return met_id[:-len(suffix)]
        return met_id

    rows = {}
    for met in model.metabolites:
        ann = met.annotation or {}
        smiles = ann.get('smiles') or ann.get('SMILES')
        if isinstance(smiles, list) and smiles:
            smiles = smiles[0]
        if isinstance(smiles, str) and smiles.startswith('SMILES:'):
            smiles = smiles.split('SMILES:', 1)[1]
        if not smiles:
            smiles = 'noinchikey'
        rows[_strip_compartment(str(met.id))] = smiles

    return pd.DataFrame({'metname': list(rows.keys()), 'Canonical_SMILES': list(rows.values())})


def reaction_split(reactions: list, metdf_name: pd.DataFrame, metname_to_smiles,
                   smiles_main_substrates, name_smiles_file, substrates, currency_metabolites):
    normal_reactions = []
    main_products = []
    smiels_reactions = []
    smiles_main_products = []
    smiles_substrates = []
    name_to_smiles = pd.read_csv(name_smiles_file)
    name_to_smiles_dict = name_to_smiles.set_index('name')['smiles'].to_dict()
    update_dict(name_to_smiles_dict)

    not_main_product = ["Reduced ferredoxin", "Oxidized ferredoxin"]
    from ecGEM_utils import name_without_smiles
    N = len(reactions)
    for index, reaction in enumerate(reactions):
        bar(index, N)
        if reaction is np.nan:  # Some entries are NaN.
            normal_reactions.append(np.nan)
            smiels_reactions.append(np.nan)
            main_products.append(np.nan)
            smiles_main_products.append(np.nan)
            smiles_substrates.append(np.nan)
            continue
        react = reaction.split("-->")
        substrate, product = react[0], react[1]

        normal_substrate, smiles_substrate, coffs = compound_split(substrate.split(" + "), metdf_name,
                                                                   metname_to_smiles, name_to_smiles_dict)
        if any(item != "1" for item in coffs):  # If any compound has a coefficient.
            result = []
            # Interleave coefficients with corresponding compounds.
            for i in range(max(len(coffs), len(normal_substrate))):
                number = coffs[i] if i < len(coffs) else None
                char = normal_substrate[i] if i < len(normal_substrate) else None
                if number == '1':
                    result.append(char)  # Keep the compound only when coefficient is 1.
                elif number is not None:
                    # Otherwise combine coefficient and compound text.
                    result.append(f"{number} {char}" if char else number)
            normal_substrate = result
        normal_product, smiles_product, coffs = compound_split(product.split(" + "), metdf_name, metname_to_smiles,
                                                               name_to_smiles_dict)
        # print(smiles_product)
        main_product, smiles_main_product = get_main_product(product.split(" + "), normal_product, smiles_product,
                                                             currency_metabolites, name_without_smiles,
                                                             name_to_smiles_dict)
        # print(smiles_main_product)
        main_products.append(main_product)
        smiles_main_products.append(smiles_main_product)
        if any(item != "1" for item in coffs):  # Repeat the formatting for products.
            result = []
            # Interleave coefficients and product names.
            for i in range(max(len(coffs), len(normal_product))):
                number = coffs[i] if i < len(coffs) else None
                char = normal_product[i] if i < len(normal_product) else None
                if number == '1':
                    result.append(char)
                elif number is not None:
                    # Combine coefficient with product name.
                    result.append(f"{number} {char}" if char else number)
            normal_product = result

        if smiles_main_substrates[index] == "None":  # If the main substrate lacks a SMILES.
            if name_to_smiles_dict.__contains__(substrates[index]):
                smiles_main_substrates[index] = name_to_smiles_dict[substrates[index]]
            elif substrates[index] not in name_without_smiles:  # Skip querying if in the blacklist.
                temp_smiles = chemical_name_to_smiles(substrates[index])  # Query PubChem (run once).
                print(f"--------{substrates[index]}--------------{temp_smiles}")
                # If the NCBI database cannot be found, you can consider manually searching and adding the
                # contains_dict method or strict_update method in the update_dict of the utils
                # if temp_smiles != "None":
                #     name_to_smiles_dict[substrates[index]] = temp_smiles
                name_to_smiles_dict[substrates[index]] = temp_smiles
                smiles_main_substrates[index] = temp_smiles
            # Report substrates missing SMILES unless explicitly excluded.
            if smiles_main_substrates[index] == "None" and substrates[index] not in name_without_smiles:
                print(f"The substrate do not have smiles:{substrates[index]}")
        smiles_substrate_clean = [str(s) for s in smiles_substrate if str(s) not in {'None', 'nan', 'noinchikey'}]
        smiles_product_clean = [str(s) for s in smiles_product if str(s) not in {'None', 'nan', 'noinchikey'}]
        smiels_reactions.append(" + ".join(smiles_substrate_clean) + " -----> " + " + ".join(smiles_product_clean))
        normal_reactions.append(" + ".join(normal_substrate) + " = " + " + ".join(normal_product))
        # smiels_reactions.append(" + ".join(smiles_substrate) + " -----> " + " + ".join(smiles_product))
        smiles_substrates.append(smiles_main_substrates)

    pd.DataFrame(list(name_to_smiles_dict.items()),
                 columns=['name', 'smiles']).to_csv(name_smiles_file, sep=",", index=False)  # Overwrite archive.
    print("save the new name_to_smiles.tsv")
    return normal_reactions, smiels_reactions, \
        main_products, smiles_main_products, \
        substrates, smiles_main_substrates, name_to_smiles_dict


def generate_input(metdf, metdf_name, readf, ph, temperature, enzyme_type, organisms,
                       metdf_outfile, smilesdf, currency_metabolites,
                       inputdf_file, name_smiles_file):
    '''
    This function is used to generate the input file.

    Arguments:
    * metdf: pandas.DataFrame - DataFrame containing reaction, substrate, gene, protein sequence, and mass information.
    * metdf_name: pandas.DataFrame - DataFrame containing metabolite names.
    * metdf_outfile: str - File path to save the modified metdf DataFrame as a CSV file.
    * DLinputdf_file: str - File path to save the input DataFrame as a CSV file.

    Returns:
    * DLinputdf: pandas.DataFrame - DataFrame containing DLKCAT input data.
    '''
    # Generate the input file
    metdf_name.index = metdf_name['met']
    metdf['metname'] = metdf_name.loc[metdf['metabolitestotal'], 'name'].values
    # Drop rows with missing prosequence
    metdf.dropna(subset=['prosequence'], inplace=True)

    # Replace specific values with None
    metdf['similes'].replace(['noinchikey_inpubchem', 'noinchikey', 'nosmiles'], 'None', inplace=True)

    # Remove trailing decimals from similes
    metdf['similes'] = metdf['similes'].astype(str).str.split('.').str[0]

    metdf.rename(columns={'similes': 'smiles_substrate',
                          'reactions': 'reac_id',
                          'prosequence': 'sequence',
                          'metname': 'substrate',
                          }, inplace=True)

    # Rename readf columns ('reac' -> 'reac_id') to facilitate merging.
    readf.rename(columns={'reac': 'reac_id', 'ec_number': 'EC_number'}, inplace=True)
    # Merge reaction info into metdf.
    merged_df = pd.merge(metdf, readf, on='reac_id', how='left')
    metdf.reset_index(drop=True, inplace=True)
    metdf.to_csv(metdf_outfile, index=False)
    metname_to_smiles = dict(zip(smilesdf['metname'], smilesdf['Canonical_SMILES']))
    normal_reactions, smiels_reactions, main_products, smiles_main_products, \
        substrates, smiles_substrates, name_to_smiles_dict = reaction_split(
        merged_df['reaction'].to_list(),
        metdf_name, metname_to_smiles,
        metdf['smiles_substrate'].to_list(), name_smiles_file,
        merged_df['substrate'].to_list(), currency_metabolites)
    merged_df['ph'] = ph
    merged_df['temperature'] = temperature
    merged_df['organisms'] = organisms
    merged_df['type'] = enzyme_type
    merged_df['product'] = main_products
    merged_df['reaction'] = normal_reactions
    merged_df['smiles_product'] = smiles_main_products
    merged_df['smiles_reaction'] = smiels_reactions
    merged_df['substrate'] = substrates
    merged_df['smiles_substrate'] = smiles_substrates

    inputdf = merged_df[['reac_id', 'organisms', 'sequence', 'substrate', 'product', 'reaction',
                            'type', 'ph', 'temperature', 'uniprot_id', 'smiles_substrate', 'smiles_product',
                            'smiles_reaction']].copy()  #
    inputdf = inputdf[inputdf['smiles_substrate'] != 'None']
    # Drop any rows containing missing values outright.
    inputdf = inputdf[~inputdf.isin(['noinchikey']).any(axis=1) & ~inputdf.isnull().any(axis=1)]
    #
    inputdf['smiles_reaction'] = inputdf['smiles_reaction'].apply(
        lambda x: x.replace('noinchikey', 'None') if 'noinchikey' in x else x)
    inputdf = inputdf[inputdf['smiles_product'] != 'None']
    inputdf.to_csv(inputdf_file, sep='\t', index=False)
    print('input file generated')
    return inputdf


def main(args):
    dataset_path = f"{args.path}/dataset/"
    create_file(dataset_path)
    sbml_path = f"{args.path}/network.xml"
    name_smiles_file = f"predict/data_file/name_smiles.tsv"
    gene_subnum_path = f"{dataset_path}gene_subnum.csv"
    sub_description_path = f'{dataset_path}get_gene_subunitDescription.csv'
    inchikey_list_file = f'{dataset_path}inchikey_list.csv'
    inchikey_list_smilesfile = f'{dataset_path}inchikey_list_smiles.csv'
    comdf_file = f'{dataset_path}comdf.csv'
    metdf_outfile = f'{dataset_path}metabolites_reactions_gpr_similes_prosequence_mass_dropna.csv'
    metabolites_reactions_gpr_file = f'{dataset_path}metabolites_reactions_gpr.csv'
    prodf_file = f'{dataset_path}prodf.csv'
    reaction_file = f'{dataset_path}reaction_ec_number.csv'
    input_file = f'{dataset_path}input.tsv'

    # Step 0: read GEM
    if re.search('\.xml', sbml_path):
        model = cobra.io.read_sbml_model(sbml_path)
    elif re.search('\.json', sbml_path):
        model = cobra.io.json.load_json_model(sbml_path)

    starttime = datetime.datetime.now()
    # Step 1: subunit number of each reaction
    print("Starting to fetch subunit number of each enzyme")
    # get_gene_subunitDescription(sub_description_path, model)  # 000 Download from the UniProt API, run it once.
    subbnumdf = get_subunit_number(sub_description_path, gene_subnum_path)
    print("Calculation done!")
    
    print()
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 2: convert metbolites bigg id to smiles
    print("Starting to convert metbolites bigg id to smiles...")
    metdf_name = get_met_bigg_id(model)
    smilesdf = smilesdf_from_model_annotations(model)
    if (smilesdf['Canonical_SMILES'] != 'noinchikey').any():
        smilesdf.to_csv(inchikey_list_smilesfile, index=False)
    else:
        # inchkeydf = convert_bigg_met_to_inchikey(metdf_name['met'], inchikey_list_file)  #111 from BIGG
        inchkeydf = pd.read_csv(inchikey_list_file)
        smilesdf = convert_inchikey_to_smiles(inchkeydf, inchikey_list_smilesfile)  # From PubChem; saved to inchikey_list_smiles.csv
    print("Converting done!")
    print()
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 3: get protein sequence and mass in model
    print("Starting to get protein sequence and mass in model...")
    subbnumdf = pd.read_csv(gene_subnum_path)
    # prodf = get_model_protein_sequence_and_mass(model, subbnumdf, prodf_file)  # 333
    print("Getting done!")
    print()
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 4: split the substrate of reactions to match the gene
    print("Starting to split the substrate of reactions to match the gene...")

    spdf = split_substrate_to_match_gene(model, metabolites_reactions_gpr_file)  ## Split reactions by substrate
    currency_metabolites = get_currency_metabolites()
    print("Splitting done!")
    print()
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 5: combine the reaction--substrate--gene--protein_sequnce--mass and formate Kcat input file
    print("Starting to get reaction and ec number data...")
    readf = get_reaciton_bigg_id(model, reaction_file)
    print("Combinning done!")
    print()
    endtime = datetime.datetime.now()
    print(endtime - starttime)

    print("Starting to combine data...")
    metdf_name = get_met_bigg_id(model)
    readf = get_reaciton_bigg_id(model, reaction_file)
    smilesdf = pd.read_csv(inchikey_list_smilesfile)
    spdf = pd.read_csv(metabolites_reactions_gpr_file)  # spdf
    prodf = pd.read_csv(prodf_file)
    comdf = combine_reactions_simles_sequence(spdf, smilesdf, prodf, comdf_file)  # t  metdf
    readf = pd.read_csv(reaction_file)
    ph = float(args.ph)
    organisms = args.organisms
    temperature = float(args.temperature)
    enzyme_type = args.enzyme_type
    inputdf = generate_input(comdf, metdf_name, readf, ph, temperature, enzyme_type, organisms, metdf_outfile,
                                    smilesdf, currency_metabolites, input_file,
                                    name_smiles_file)  # enzyme kinetic network
    print("Combinning done!")
    print()
    


if __name__ == '__main__':
    # predict/km_predict
    # predict/Synechocystis sp
    # predict/rubisco
    # predict/E.coli
    # predict/iECDH1ME8569_1439
    data = {
        'iECDH1ME8569_1439': {
            'ph': 7.4,
            'organisms': 'Escherichia coli',
            'temperature': 37.0,
            'is_auto': False,
            'enzyme_type': 'wildtype'
        },
        'yeast9': {
            'ph': 7.4,
            'organisms': 'yeast',
            'temperature': 30.0,
            'is_auto': False,
            'enzyme_type': 'wildtype'
        },
        'iM1515': {
            'ph': 7.4,
            'organisms': 'Escherichia coli',
            'temperature': 37.0,
            'is_auto': False,
            'enzyme_type': 'wildtype'
        },
        'Synechocystis sp':{
            'ph': 8.0,
            'organisms': 'Synechocystis sp. PCC 6803',
            'temperature': 30.0,
            'is_auto': True,
            'enzyme_type': 'wildtype'
        },
        'ENGRO2':{
            'ph': 7.2,
            'organisms': 'Homo sapiens (Human)',
            'temperature': 37.0,
            'is_auto': False,
            'enzyme_type': 'wildtype'
        },
        'Bacillus subtilis':{
            'ph': 7.2,
            'organisms': 'Bacillus subtilis',
            'temperature': 37.0,
            'is_auto': False,
            'enzyme_type': 'wildtype'
        }
    }
    input = "yeast9"

    parser.add_argument("--path", type=str,
                        default=f"predict/{input}")  # predict/km_predict  #predict/Synechocystis sp
    parser.add_argument("--ph", type=str, default=f"{data[input]['ph']}")  # 8.0
    parser.add_argument("--organisms", type=str, default=f"{data[input]['organisms']}")  # Synechocystis sp. PCC 6803
    parser.add_argument("--temperature", type=str, default=f"{data[input]['temperature']}")  # 30.0
    parser.add_argument("--is_auto", type=bool, default=f"{data[input]['is_auto']}")
    parser.add_argument("--enzyme_type", type=str, default=f"{data[input]['enzyme_type']}")  # wildtype

    args = parser.parse_args()
    main(args=args)

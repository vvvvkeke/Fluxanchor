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
# from ecmpy.script.AutoPACMEN_function import *
from script.AutoPACMEN_function import parse_bigg_metabolites_file, \
    parse_brenda_textfile, parse_brenda_json_for_model, \
    create_combined_kcat_database, get_reactions_kcat_mapping, get_ec_number_kcats_wildcard_search
from script.ECMpy_function import get_reaction_mw

parser = argparse.ArgumentParser()


def parse_sabio_rk_for_model(model: cobra.Model, json_output_path: str, bigg_id_name_mapping_path: str) -> None:
    ec_numbers_list: List[str] = []
    for reaction in model.reactions:
        if "ec-code" not in reaction.annotation.keys():
            continue
        ec_codes = reaction.annotation["ec-code"]
        if type(ec_codes) is str:
            ec_codes = [ec_codes]
        ec_numbers_list += ec_codes
    ec_numbers_list = list(set(ec_numbers_list))

    ec_numbers_list = [item for item in ec_numbers_list if item.count(".") == 3]
    # GET KCATS FOR EC NUMBERS
    ec_number_kcat_mapping = get_ec_number_kcats_wildcard_search(
        ec_numbers_list, bigg_id_name_mapping_path)

    json_write(json_output_path, ec_number_kcat_mapping)


def main(args):
    autopacmen_folder = f"{args.path}/analysis/get_kcat_mw_by_AutoPACMEN/"
    kcat_gap_fill = 'mean'  # 'mean'#'median'
    reaction_gap_fill = 'mean'
    sbml_path = f"{args.path}/network.xml"
    organism = args.organisms
    project_name = kcat_gap_fill
    create_file(autopacmen_folder)
    protein_kcat_database_path = "none"
    f"predict/data_file/name_smiles.tsv"
    bigg_metabolites_file = "predict/data_file/bigg_models_metabolites.txt"
    brenda_textfile_path = "predict/data_file/brenda_2024_1.txt"
    uniprot_data_file = "predict/data_file/uniprot_data_accession_key.json"

    # output files
    brenda_json_path = "%skcat_database_brenda.json" % autopacmen_folder
    brenda_json_path2 = "%ssa_database_brenda.json" % autopacmen_folder
    sabio_rk_json_path = "%skcat_database_sabio_rk.json" % autopacmen_folder
    bigg_id_name_mapping_path = "%sbigg_id_name_mapping.json" % autopacmen_folder
    brenda_output_json_path = "%skcat_database_brenda_for_model.json" % autopacmen_folder
    combined_output_path = "%skcat_database_combined.json" % autopacmen_folder
    sub_description_path = '%sget_gene_subunitDescription.csv' % autopacmen_folder
    gene_subnum_path = "%sgene_subnum.csv" % autopacmen_folder
    reaction_mw_path = "%sreaction_mw.json" % autopacmen_folder
    reaction_kcat_mw_path = '%sreaction_kcat_MW.csv' % autopacmen_folder

    starttime = datetime.datetime.now()
    # Step 1: get bigg metbolite
    print("Starting to deal BIGG metabolites text file...")
    if not os.path.exists(bigg_metabolites_file):
        url = "http://bigg.ucsd.edu/static/namespace/bigg_models_metabolites.txt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(bigg_metabolites_file, "wb") as file:
                file.write(response.content)
            print("download bigg metabolites file success!")
        else:
            print(f"download failed :{response.status_code}")
    else:
        print("bigg metabolites file already exists!")
    parse_bigg_metabolites_file(bigg_metabolites_file, autopacmen_folder)
    print("BIGG metabolites text file done!")
    print()

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 2: BRENDA kcat
    print("Starting to deal BRENDA textfile...")
    parse_brenda_textfile(brenda_textfile_path, autopacmen_folder, brenda_json_path, brenda_json_path2)
    print("BRENDA textfile done!")
    print()

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 3: Select Brenda kcat for model
    print("Starting to deal brenda json for model...")
    parse_brenda_json_for_model(sbml_path, brenda_json_path, brenda_output_json_path)
    print("BRENDA json for model done!")
    print()

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 4: SABIO-RK kcat for model
    print("Starting EC numbers kcat search in SABIO-RK...")
    parse_sabio_rk_for_model_with_sbml(sbml_path, sabio_rk_json_path, bigg_id_name_mapping_path)  # 000 run it once
    print("SABIO-RK done!")
    print()

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 5: Brenda and SABIO-RK kcat combined
    print("Combining kcat database...")
    create_combined_kcat_database(sabio_rk_json_path, brenda_output_json_path, combined_output_path)
    print("Combining kcat database done!")
    print()

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 6: subunit number of each reaction
    print("Starting to fetch subunit number of each enzyme")
    if re.search('\.xml', sbml_path):
        model = cobra.io.read_sbml_model(sbml_path)
    elif re.search('\.json', sbml_path):
        model = cobra.io.json.load_json_model(sbml_path)
    get_gene_subunitDescription(sub_description_path, model)  # 111 
    subbnumdf = get_subunit_number(sub_description_path, gene_subnum_path)
    print("Calculation done!")
    print()

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 7: get mw for model gene (must be uniprot ID)
    print("Starting UniProt ID<->Protein mass search using UniProt...")
    if not os.path.exists(uniprot_data_file):
        url = "https://zenodo.org/records/8119567/files/uniprot_data_accession_key.json"
        response = requests.get(url)
        if response.status_code == 200:
            with open(uniprot_data_file, "wb") as file:
                file.write(response.content)
            print("download uniprot_data_file success!")
        else:
            print(f"download failed :{response.status_code}")
    else:
        print("uniprot_data_file already exists!")
    get_protein_mass_mapping_from_local(sbml_path, autopacmen_folder, project_name, uniprot_data_file)
    get_reaction_mw(sbml_path, autopacmen_folder, project_name, reaction_mw_path, gene_subnum_path)
    print("Protein ID<->Mass mapping done!")
    print()

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 8: kcat assignment for model(include sa)
    print("Starting to assign kcat for model...")
    get_reactions_kcat_mapping(sbml_path, autopacmen_folder, project_name, organism, combined_output_path,
                               brenda_json_path2, reaction_mw_path, protein_kcat_database_path, kcat_gap_fill)

    print("kcat assignment done!")
    print()

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # Step 9: get_reaction_kcat_mw for model
    print("Starting to get reaction kcat_mw for model...")
    if re.search('\.xml', sbml_path):
        model = cobra.io.read_sbml_model(sbml_path)
    elif re.search('\.json', sbml_path):
        model = cobra.io.json.load_json_model(sbml_path)
    get_reaction_kcat_mw(model, autopacmen_folder, project_name, reaction_gap_fill, gene_subnum_path,
                         reaction_kcat_mw_path)
    print("Reaction kcat_mw done!")

    endtime = datetime.datetime.now()
    print(endtime - starttime)

    starttime = datetime.datetime.now()
    # get ecGEM
    print("Starting to get reaction kcat_mw for model......")
    method = "AutoPACMEN"
    ecGEM_path = f"{args.path}/analysis/get_kcat_mw_by_{method}/ecGEM"
    create_file(ecGEM_path)

    if args.is_auto:
        obj = 'BIOMASS_Ec_SynAuto_1' 
    reaction_kcat_MW_file = f"{args.path}/analysis/get_kcat_mw_by_{method}/reaction_kcat_MW.csv"
    gene_abundance_colname = 'abundance'
    gene_abundance_file = './predict/data_file/gene_abundance.csv'  # downolad from https://pax-db.org/download
    ecModel_output_file = f"{ecGEM_path}/ecGEM_irr_enz_constraint.json"
    f = 0.4  # The enzyme mass fraction 
    ptot = 0.56  # The total protein fraction in cell. 
    sigma = 0.5  # The approximated saturation of enzyme.e.g.,0.5/1.
    lowerbound = 0  # Lowerbound  of enzyme concentration constraint.
    if organism == "iECDH1ME8569_1439":
        upperbound = 0.35 
    else:
        upperbound = 0.5  # round(ptot * f * sigma, 3)  # total enzyme
    normal_model = cobra.io.read_sbml_model(sbml_path)
    if args.is_auto:
        normal_model = set_auto_model(normal_model)
    
    # normal_model.objective = obj
    trans_model2enz_json_model_split_isoenzyme(ecGEM_path, sbml_path, reaction_kcat_MW_file, f, ptot, sigma, lowerbound,
                                               upperbound,
                                               ecModel_output_file, synauto=args.is_auto)
    enz_model = get_enzyme_constraint_model(ecModel_output_file)
    print(f"ecGEM saved in {ecModel_output_file}")
    endtime = datetime.datetime.now()
    print(endtime - starttime)


def parse_sabio_rk_for_model_with_sbml(sbml_path: str, json_output_path: str, bigg_id_name_mapping_path: str) -> None:
    """See this module's parse_sabio_rk_for_model() documentation. This function uses an SBML path.

    Arguments
    ----------
    * sbml_path: str ~ The model's SBML path.
    * json_output_path: str ~ The path of the JSON that shall be created
    """
    # LOAD SBML MODEL
    # model: cobra.Model = cobra.io.read_sbml_model(sbml_path)
    if re.search('\.xml', sbml_path):
        model = cobra.io.read_sbml_model(sbml_path)
    elif re.search('\.json', sbml_path):
        model = cobra.io.json.load_json_model(sbml_path)

    parse_sabio_rk_for_model(model, json_output_path,
                             bigg_id_name_mapping_path)


if __name__ == '__main__':
    # predict/km_predict
    # predict/Synechocystis sp
    # predict/rubisco
    # predict/E.coli
    # predict/iECDH1ME8569_1439
    # predict/Bacillus subtilis
    parser.add_argument("--path", type=str,
                        default="predict/Bacillus subtilis")  # predict/km_predict  #predict/Synechocystis sp
    # Synechocystis sp.
    # Escherichia coli
    # Bacillus subtilis
    parser.add_argument("--organisms", type=str, default='Bacillus subtilis')
    parser.add_argument("--is_auto", type=bool, default=False)
    # 30.0
    # 37.0
    args = parser.parse_args()
    main(args=args)

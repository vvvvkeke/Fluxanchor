import json
import time

import numpy as np
import pandas as pd
import os

import requests
from cobra import Reaction
from cobra.util import set_objective
from pyprobar import probar
import json
import os
import cobra
import copy
import csv
import io
import requests
import time
import math
import random
import pickle
import sys
import statistics
from typing import Any, Dict, List
from Bio import Entrez
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re  # Mr.Mao

from script.AutoPACMEN_function import _get_searched_metabolites, _get_kcat_from_protein_kcat_database, json_load, \
    standardize_folder, json_write, _get_kcat

name_without_smiles = ["Apoprotein", "Cis-octadec-9-enoyl-[acyl-carrier protein] (n-C18 1)",
                           "1,2-dihexadec-9-enoyl-sn-glycerol 3-phosphate", "Reduced thioredoxin",
                           "1-octadec-6-9-12-15-tetraenoyl-sn-glycerol 3-phosphate",
                           "1,2-dioctadec-11-enoyl-sn-glycerol 3-phosphate",
                           "1,2-Diacyl-sn-glycerol (dioctadec-6-9-12-15-tetraenoyl, n-C18 4)",
                           "(9Z)-Hexadecanoyl-[acp]", "1,2-dioctadec-6-9-12-trienoyl-sn-glycerol 3-phosphate",
                           "1,2-Diacyl-sn-glycerol (dioctadec-9-12-15-trienoyl, n-C18 3)",
                           "Stearidonoyl-[acp]Stearidonoyl-[acyl-carrier protein](6Z,9Z,12Z,Z15)-Octadecatrienoyl-ACP(6Z,9Z,12Z,Z15)-Octadecatrienoyl-[acyl-carrier protein]",
                           "CDP-1,2-dioctadec-9-12-15-trienoylglycerol",
                           "1,2-dioctadec-9-12-15-trienoyl-sn-glycerol 3-phosphate", "Enoylpimeloyl-[acp] methyl ester",
                           "Plastosemiquinone located at the luminal side of the Cytochrome-b6/f complex",
                           "Reduced rieske_B6Reduced Rieske complex of the Cytochrome-b6/f complex",
                           "Oxidized rieske_B6Oxidized Rieske complex of the Cytochrome-b6/f complex",
                           "Heme B located in the Cytochrome-b6/f complex, twice protonated",
                           "PQ_B6_SPlastoquinone located at the stromal side of the Cytochrome-b6/f complex",
                           "Plastosemiquinone located at the luminal side of the Cytochrome-b6/f complex",
                           "Heme B located in the Cytochrome-b6/f complex, triply protonated",
                           "1,2-Diacyl-sn-glycerol (dioctadec-6-9-12-trienoyl, n-C18 3)",
                           "1,2-dioctadec-6-9-12-trienoyl-sn-glycerol 3-phosphate",
                           "1,2-Diacyl-sn-glycerol (dioctadec-9-12-15-trienoyl, n-C18 3)",
                           "1,2-dioctadec-9-12-15-trienoyl-sn-glycerol 3-phosphate",
                           "1,2-Diacyl-sn-glycerol (dioctadec-6-9-12-15-tetraenoyl, n-C18 4)",
                           "1,2-dioctadec-6-9-12-15-tetraenoyl-sn-glycerol 3-phosphate",
                           "Phosphatidylglycerophosphate (dioctadec-9-12-15-trienoyl, n-C18 3)",
                           "Plastosemiquinone located at the stromal side of the Cytochrome-b6/f complex",
                           "Photons with 700nm wavelength",
                           "Uncharged reaction centre of the Photosystem I (light with 700 nm is sufficient for the excitation of the one reaction center)",
                           "Internal bound semiquinone radical of the Photosystem II",
                           "Positive charged reaction centre of the Photosystem I (light with 700 nm is sufficient for the excitation of the one reaction center)",
                           "Photons with 680nm wavelength",
                           "Neutral reaction centre of the Photosystem II (light with 680 nm is sufficient for the excitation of the one reaction center)",
                           "Internal bound plastoquinone of the Photosystem II",
                           "Semiplastoquinone radical loosely bound to the Photosystem II (can except one further electron)",
                           "Positive charged reaction centre of the Photosystem II (light with 680 nm is sufficient for the excitation of the one reaction center)",
                           "Starting state of a cluster of probably four manganese atoms",
                           "First oxidation state of the cluster of manganese atoms in the Photosystem II (needed for water splitting)",
                           "Second oxidation state of the cluster of manganese atoms in the Photosystem II (needed for water splitting)",
                           "Third oxidation state of the cluster of manganese atoms in the Photosystem II (needed for water splitting)",
                           "Fourth oxidation state of the cluster of manganese atoms in the Photosystem II (needed for water splitting)",
                           "1-octadec-9-12-15-trienoyl-sn-glycerol 3-phosphate",
                           "CDP-1,2-dioctadec-6-9-12-15-tetraenoylglycerol",
                           "3-oxo-cis-myristol-7-eoyl-[acyl-carrier protein]",
                           "Trans-3-cis-9-palmitoleoyl-[acyl-carrier protein]",
                           "Stearidonoyl-[acp]Stearidonoyl-[acyl-carrier protein](6Z,9Z,12Z,Z15)-Octadecatrienoyl-ACP(6Z,9Z,12Z,Z15)-Octadecatrienoyl-[acyl-carrier protein]",
                           "1-octadec-9-enoyl-sn-glycerol 3-phosphate",
                           "1-octadec-9-12-dienoyl-sn-glycerol 3-phosphate",
                           "1-octadec-6-9-12-15-tetraenoyl-sn-glycerol 3-phosphate",
                           "1-octadec-6-9-12trienoyl-sn-glycerol 3-phosphate",
                           "1,2-dioctadec-9-enoyl-sn-glycerol 3-phosphate",
                           "1,2-Diacyl-sn-glycerol (dioctadec-9-enoyl, n-C18 1)",
                           "1,2-Diacyl-sn-glycerol (dioctadec-9-12-dienoyl, n-C18 2)",
                           "CDP-1,2-dioctadec-6-9-12-trienoylglycerol",
                           "1,2-dioctadec-9-12-dienoyl-sn-glycerol 3-phosphate",
                           "CDP-1,2-dihexadecanoylglycerol", "CDP-1,2-dioctadec-9-enoylglycerol",
                           "CDP-1,2-dioctadec-9-enoylglycerol ",
                           "CDP-1,2-dioctadec-9-12-dienoylglycerol",
                           "Sulfoquinovosyldiacylglycerol (n-C18::1:9)",
                           "Oxidized rieske B6",
                           "Heme X located in the Cytochrome-b6/f complex",
                           "Reduced rieske B6", "Oxidized ferredoxin",
                           "Ferricytochrome c6",
                           "Heme X located in the Cytochrome-b6/f complex, protonated",
                           "Heme O C49H56FeN4O5",
                           "Protoheme C34H30FeN4O4", "Acyl carrier protein",
                           "Positive charged reaction centre of the Photosystem II",
                           "PSII reaction center P680", "PSI reaction center P700",
                           "Positive charged reaction centre of the Photosystem I",
                           "OctadecatetraenoilACP",
                           "Third oxidation state of the cluster of manganese atoms in the Photosystem II",
                           "H2O H2O", "O2 O2", "H2O H2O", "CO2 CO2",
                           "3-Oxohexanoyl-[acyl-carrier protein]",
                           "(R)-3-Hydroxybutanoyl-[acyl-carrier protein]",
                           "Hexanoyl-ACP (n-C6:0ACP)",
                           "Butyryl-ACP (n-C4:0ACP)", "Acyl carrier protein", "Decanoyl-ACP (n-C10:0ACP)",
                           "Cysteine enzyme", "Heme A C49H55FeN4O6", "Protoheme C34H30FeN4O4",
                           "Plastocyanin(Cu+)", "Dihydrolipolprotein",
                            "2 Amino 4 hydroxy 6 hydroxymethyl 7 8 dihydropteridine diphosphate C7H8N5O8P2",
                           "3-Oxooctanoyl-[acyl-carrier protein]",
                           "Trans-Dodec-2-enoyl-[acyl-carrier protein]", "Beta-Aminopropion aldehyde",
                                                                         "Plastocyanin(Cu2+)",
                           "(R)-3-Hydroxytetradecanoyl-[acyl-carrier protein]",
                           "Trans-Dec-2-enoyl-[acyl-carrier protein]", "3-Oxohexadecanoyl-[acyl-carrier protein]",
                                                                       "Trans-Oct-2-enoyl-[acyl-carrier protein]",
                           "TRNA containing uridine at position 54",
                           "Trans-Hex-2-enoyl-[acyl-carrier protein]", "Biliverdin cytosol",
                                                                       "Trans-octadec-2-enoyl-[acyl-carrier protein]",
                           "Glutaredoxin", "FMN C17H19N4O9P", "Flavin adenine dinucleotide oxidized",
                           "Trans-Tetradec-2-enoyl-[acyl-carrier protein]", "Ferrocytochrome c6",
                                                                            "Oxidized thioredoxin",
                           "Trans-Hexadec-2-enoyl-[acyl-carrier protein]", "Adenosyl-cobyric acid",
                           "2 5 Diamino 6 hydroxy 4  5  phosphoribosylamino  pyrimidine C9H14N5O8P",
                           "Glycogen C6H10O50", "Tetrahydrofolyl Glu 2  C24H27N8O9", "Glycogen C6H10O5",
                           "Heme X located in the Cytochrome-b6/f complex,", "Linoleoyl-ACP (n-C18 2ACP)",
                           "Succinate semialdehyde-thiamin diphosphate anion", "Branching glycogen",
                           "Semiplastoquinone radical loosely bound to the Photosystem II",
                           "Second oxidation state of the cluster of manganese atoms in the Photosystem II",
                           "Fourth oxidation state of the cluster of manganese atoms in the Photosystem II",
                           "Cyanophycin (multi-Larginyl-poli [L--aspartic acid]) polimer (n+2)",
                           "Cyanophycin (multi-Larginyl-poli [L--aspartic acid]) polimer (n)",
                           "3-Oxodecanoyl-[acyl-carrier protein]", "3-Oxooctadecanoyl-[acyl-carrier protein]",
                           "3-Oxododecanoyl-[acyl-carrier protein]", "3-Oxotetradecanoyl-[acyl-carrier protein]",
                           "First oxidation state of the cluster of manganese atoms in the Photosystem II",
                           "13(1)-Hydroxy-magnesium-protoporphyrin IX 13-monomethyl ester", "Palmitoyl-ACP (n-C16:0ACP)",
                           "TRNA Asn  C10H17O10PR2", "G-linolenoilACP", "Acetoacetyl-ACP",
                           "(R)-3-Hydroxyoctanoyl-[acyl-carrier protein]", "P1,P4-Bis(5''-adenosyl)tetraphosphate",
                           "R-3-hydroxypalmitoyl-[acyl-carrier protein]", "Arsenite", "Acetyl-ACP",
                           "(R)-3-Hydroxydecanoyl-[acyl-carrier protein]", "Glutaredoxin disulfide",
                           "A-lineloyl-ACP (n-C18 3ACP)", "7-Dehydrocholesterol; Provitamin D3",
                           "Octadecanoyl-ACP (n-C18:0ACP)", "TRNA containing ribothymidine at position 54",
                           "Myristoyl-ACP (n-C14:0ACP)", "R-3-hydroxypalmitoyl-[acyl-carrier protein]",
                           "Dodecanoyl-ACP (n-C12:0ACP)", "Octanoyl-ACP (n-C8:0ACP)",
                           "1,2-Diacyl-3-beta-D-galactosyl-sn-glycerol (n-C16)",
                           "1,2-Diacyl-3-beta-D-galactosyl-sn-glycerol (n-C16 1)",
                           "1,2-Diacyl-3-beta-D-galactosyl-sn-glycerol (n-C18 0)",
                           "1,2-Diacyl-3-beta-D-galactosyl-sn-glycerol (n-C18 2)",
                           "1,2-Diacyl-3-beta-D-galactosyl-sn-glycerol (n-C18 3)",
                           "1,2-Diacyl-3-beta-D-galactosyl-sn-glycerol (n-C18 4)",
                           "1,2-Diacyl-3-beta-D-galactosyl-sn-glycerol (n-C18 1)"
                           ]


def isoenzyme_split(model):
    """Split isoenzyme reaction to mutiple reaction
    isoenzyme_split 函数用于将具有多个基因反应规则的同工酶反应拆分为多个独立的反应。这一过程有助于在代谢模型中更清晰地表达同工酶的功能。
    Arguments
    ----------
    * model: cobra.Model.

    :return: new cobra.Model.
    """
    for r in model.reactions:  # 遍历模型中的每个反应。
        if re.search(" or ", r.gene_reaction_rule):  # 如果反应的基因反应规则中包含 " or "，说明该反应可以由多个基因催化。
            rea = r.copy()  # 复制当前反应，以便创建新的反应实例。
            gene = r.gene_reaction_rule.split(" or ")  # 将基因反应规则按 " or " 拆分，得到一个基因列表。
            # 对于拆分后的每个基因：
            # 对于第一个基因，修改原反应的 ID 和基因反应规则。

            for index, value in enumerate(gene):
                if index == 0:
                    r.id = r.id + "_num1"  # 改名成_num1
                    r.gene_reaction_rule = value
                else:
                    r_add = rea.copy()
                    r_add.id = rea.id + "_num" + str(index + 1)  # 改名成_numi
                    r_add.gene_reaction_rule = value
                    model.add_reaction(r_add)  # 对于后续的基因，复制原始反应，修改其 ID 和基因反应规则，并将其添加到模型中。
    for r in model.reactions:  # 遍历模型中的所有反应，去除基因反应规则两端的空格和括号。
        r.gene_reaction_rule = r.gene_reaction_rule.strip("( )")
    return model

def convert_to_irreversible(model):
    """Split reversible reactions into two irreversible reactions

    These two reactions will proceed in opposite directions. This
    guarentees that all reactions in the model will only allow
    positive flux values, which is useful for some modeling problems.

    Arguments
    ----------
    * model: cobra.Model ~ A Model object which will be modified in place.

    """
    # warn("deprecated, not applicable for optlang solvers", DeprecationWarning)
    reactions_to_add = []  # 用于存储新添加的不可逆反应。
    coefficients = {}  # 用于存储反应的目标系数。
    for reaction in model.reactions:  # 遍历模型中的每个反应。
        # if reaction.id[:2] == "BM":
        #     print(reaction.id)
        #     print(reaction.reaction)
        # if reaction.id == "EX_glc__D_e":
        #     reaction.lower_bound = -10

        # 在这种情况下，反应的代谢物系数会被调整，反应 ID 也会被修改为 _reverse，并更新其上下界。
        if reaction.lower_bound < 0 and reaction.upper_bound == 0:  # 如果反应的下界小于 0 且上界等于 0，表示该反应是反向反应。
            for metabolite in reaction.metabolites:  # 如果反应的下界小于 0 且上界大于 0，表示该反应是可逆的。
                # 创建一个新的反应 reverse_reaction，表示反向反应，其上下界根据原反应的上下界进行设置。
                original_coefficient = reaction.get_coefficient(metabolite)
                reaction.add_metabolites({metabolite: -2 * original_coefficient})
            reaction.id += "_reverse"  # 改名成_reverse
            reaction.upper_bound = -reaction.lower_bound
            reaction.lower_bound = 0
        # If a reaction is reverse only, the forward reaction (which
        # will be constrained to 0) will be left in the model.
        if reaction.lower_bound < 0 and reaction.upper_bound > 0:
            reverse_reaction = Reaction(reaction.id + "_reverse")
            reverse_reaction.lower_bound = max(0, -reaction.upper_bound)
            reverse_reaction.upper_bound = -reaction.lower_bound
            coefficients[
                reverse_reaction] = reaction.objective_coefficient * -1  # 反向反应的目标系数与原反应的系数相反。
            reaction.lower_bound = max(0, reaction.lower_bound)
            reaction.upper_bound = max(0, reaction.upper_bound)  # 将反应的上下界调整为正值，确保模型的有效性。
            # Make the directions aware of each other
            reaction.notes["reflection"] = reverse_reaction.id  # 将反向反应与原反应进行关联，以便后续使用。
            reverse_reaction.notes["reflection"] = reaction.id
            reaction_dict = {k: v * -1  # 复制原反应的代谢物信息并反向，添加到新反应中
                             for k, v in reaction._metabolites.items()}
            reverse_reaction.add_metabolites(reaction_dict)
            reverse_reaction._model = reaction._model  # 确保新反应保留与原反应相同的模型和基因信息。
            reverse_reaction._genes = reaction._genes
            for gene in reaction._genes:
                gene._reaction.add(reverse_reaction)
            reverse_reaction.subsystem = reaction.subsystem  # 设置子系统和基因反应规则
            reverse_reaction.gene_reaction_rule = reaction.gene_reaction_rule
            reactions_to_add.append(reverse_reaction)
    model.add_reactions(reactions_to_add)  # 将所有新创建的不可逆反应添加到模型中。总共有294个可逆反应，920+294=1214
    set_objective(model, coefficients, additive=True)  # 根据存储的系数设置模型的目标函数。

def get_reaciton_bigg_id(model, reaction_file):
    reaclist = []
    reaction = []
    name = []
    ec_number = []
    # Loop through reactions in the model
    for reac in model.reactions:
        if '_c' in str(reac):
            reaclist.append(str(reac.id).split('_c')[0])
            reaction.append(str(reac.reaction))
            # name.append(str(reac))
        elif '_e' in str(reac):
            reaclist.append(str(reac.id).split('_e')[0])
            reaction.append(str(reac.reaction))
        elif '_p' in str(reac):
            reaclist.append(str(reac.id).split('_p')[0])
            reaction.append(str(reac.reaction))
        else:
            reaclist.append(str(reac.id))
            reaction.append(str(reac.reaction))
    # Create a DataFrame for reactions
    reacdf = pd.DataFrame()
    # reacdf_unique = pd.DataFrame()

    reacdf['reac'] = reaclist
    reacdf['reaction'] = reaction

    if not os.path.exists(reaction_file):
        for reac in probar(reaclist):
            time.sleep(2.0)
            try:
                reaction_bigg_id = reac
                if "_num" in reac:
                    reaction_bigg_id = reaction_bigg_id.split("_num")[0]
                elif "_reverse" in reac:
                    reaction_bigg_id = reaction_bigg_id.split("_reverse")[0]
                url = f'http://bigg.ucsd.edu/api/v2/universal/reactions/{reaction_bigg_id}'
                response = requests.get(url, headers={"Accept": "application/json"})
                jsonData = json.loads(response.text)
                ec_number.append(jsonData['database_links']['EC Number'][0]['id'])

            except:
                ec_number.append('NA')

        reacdf = pd.DataFrame({'reac': reaclist, 'reaction': reaction, 'ec_number': ec_number})
        reacdf.to_csv(reaction_file, index=False)

    return reacdf


def contains_dict(name_to_smiles_dict, name, value):
    if not name_to_smiles_dict.__contains__(name) or name_to_smiles_dict[name] == "None":
        name_to_smiles_dict[name] = value

def strict_update(name_to_smiles_dict, name, value):
    name_to_smiles_dict[name] = value
def update_dict(name_to_smiles_dict):
    strict_update(name_to_smiles_dict, "Reduced ferredoxin",
                  "[*][*][Fe+2]1([*][*])[S-2][Fe+3]([*][*])([*][*])[S-2]1")
    strict_update(name_to_smiles_dict, "Oxidized ferredoxin",
                  "[*][*][Fe+3]1([*][*])[S-2][Fe+3]([*][*])([*][*])[S-2]1")
    strict_update(name_to_smiles_dict, "Divinylprotochlorophyllide",
                  "[H+].CC1=C(C2=C3C(C(=C4C3=NC(=C4C)C=C5C(=C(C(=N5)C=C6C(=C(C(=N6)C=C1[N-]2)C)C=C)C)C=C)[O-])C(=O)OC)CCC(=O)[O-].[Mg+2]")
    strict_update(name_to_smiles_dict, "DIVINYL-PROTOCHLOROPHYLLIDE-A",
                  "C=CC2(\C3(/C=c5(c(c6(C(C(C7(\C1(C(/CCC(O)=O)=C(C)/C(/N=1)=C/c4([nH](c(\C=C(C(/C)=2)/N=3)c(c4C)C=C)[Mg][nH]5c6=7))))C(OC)=O)=O))C)))")
    strict_update(name_to_smiles_dict, "Thioredoxin", "[*]N[C@@H](CS)C(=O)N[C@@H]([*])C(=O)N[C@@H]([*])C(=O)N[C@@H](CS)C([*])=O")
    strict_update(name_to_smiles_dict, "3-Oxotetradecanoyl-[acp]",
                  "CCCCCCCCCCCC(=O)CC(=O)S[*]")
    strict_update(name_to_smiles_dict, "(3R)-3-Hydroxytetradecanoyl-[acyl-carrier protein]",
                  "CCCCCCCCCCC[C@@H](O)CC(=O)S[*]")
    strict_update(name_to_smiles_dict, "Reduced plastocyanin",  # [*][*][Cu+]([*][*])([*][*])[*][*]
                  "[Cu+](C1=C(NC(=C1)C(C(=O)O)N)C(C(=O)O)N)(CC(C(=O)O)NCCS)(C(C(=O)O)NCC(S)C(=O)O)(C1=C(NC(=C1)C(C(=O)O)N))")
    strict_update(name_to_smiles_dict, "Oxidized plastocyanin",
                  "[Cu+2](C1=C(NC(=C1)C(C(=O)O)N)C(C(=O)O)N)(CC(C(=O)O)NCCS)(CC(C(=O)O)NCC(S)C(=O)O)(C1=C(NC(=C1)C(C(=O)O)N))")
    strict_update(name_to_smiles_dict, "Trans-3-cis-7-myristoleoyl-[acyl-carrier protein]",
                  r"CCCCCCC=CCCC=CCC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)([O-])O[*]")
    strict_update(name_to_smiles_dict, "Plastoquinol",
                  "CC1=C(C=C(C(=C1C)O)CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C)O")
    ###################################
    contains_dict(name_to_smiles_dict, "Cis-Aconitate", "C(C(=CC(=O)O)C(=O)O)C(=O)O")
    contains_dict(name_to_smiles_dict, "Lipoylprotein", "C1CSSC1CCCCC(=O)N")
    contains_dict(name_to_smiles_dict, "S-Aminomethyldihydrolipoylprotein", "C(CCC(=O)N)CC(CCSCN)S")
    contains_dict(name_to_smiles_dict, "O-Phospho-L-serine", "C(C(C(=O)O)N)OP(=O)(O)O")
    contains_dict(name_to_smiles_dict, "DTDP-4-dehydro-beta-L-rhamnose",
                  "CC1C(=O)C(C(C(O1)OP(=O)(O)OP(=O)(O)OCC2C(CC(O2)N3C=C(C(=O)NC3=O)C)O)O)O")
    contains_dict(name_to_smiles_dict, "Plastoquinone",
                  "CC1=C(C(=O)C(=CC1=O)CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C)C")
    contains_dict(name_to_smiles_dict, "All-trans-Phytoene",
                  "CC(=CCCC(=CCCC(=CCCC(=CC=CC=C(C)CCC=C(C)CCC=C(C)CCC=C(C)C)C)C)C)C")
    contains_dict(name_to_smiles_dict, "All-trans-Phytofluene",
                  "CC(=CCCC(=CCCC(=CCCC(=CC=CC=C(C)C=CC=C(C)CCC=C(C)CCC=C(C)C)C)C)C)C")
    contains_dict(name_to_smiles_dict, "Chlorophyllide a",
                  "[H+].CCC1=C(C2=NC1=CC3=C(C4=C(C(C(=C4[N-]3)C5=NC(=CC6=NC(=C2)C(=C6C)C=C)C(C5CCC(=O)[O-])C)C(=O)OC)[O-])C)C.[Mg+2]")
    contains_dict(name_to_smiles_dict, "D-erythro-3-Methylmalate", "CC(C(C(=O)O)O)C(=O)O")
    contains_dict(name_to_smiles_dict, "2-Methylmaleate", "CC(=CC(=O)O)C(=O)O")
    contains_dict(name_to_smiles_dict, "1D-myo-Inositol 1-phosphate", "C1(C(C(C(C(C1O)O)OP(=O)(O)O)O)O)O")
    contains_dict(name_to_smiles_dict, "D-Ribulose 1,5-bisphosphate", "C(C(C(C(=O)COP(=O)(O)O)O)O)OP(=O)(O)O")
    contains_dict(name_to_smiles_dict, "2 Amino 4 hydroxy 6 hydroxymethyl 7 8 dihydropteridine C7H9N5O2",
                  "N=C1N=C(O)C2=C(NCC(CO)=N2)N1")
    contains_dict(name_to_smiles_dict, "13(1)-Hydroxy-magnesium-protoporphyrin IX 13-monomethyl ester",
                  "CC1=C(C2=CC3=NC(=CC4=C(C(=C([N-]4)C=C5C(=C(C(=N5)C=C1[N-]2)C)CCC(=O)O)C(CC(=O)OC)O)C)C(=C3C)C=C)C=C.[Mg+2]")
    contains_dict(name_to_smiles_dict, "N1-(5-Phospho-alpha-D-ribosyl)-5,6-dimethylbenzimidazole",
                  "CC1=CC2=C(C=C1C)N(C=N2)C3C(C(C(O3)COP(=O)(O)O)O)O")
    contains_dict(name_to_smiles_dict, "2-(alpha-Hydroxyethyl)thiamine diphosphate",
                  "CC1=C(SC(=[N+]1CC2=CN=C(N=C2N)C)C(C)O)CCOP(=O)(O)OP(=O)(O)O")

    contains_dict(name_to_smiles_dict, "Apo-[carboxylase]", "NCCCC[C@H](NC([*])=O)C(=O)N[*]")
    contains_dict(name_to_smiles_dict, "Biotinyl-5'-AMP",
                  "[H][C@]12CS[C@@H](CCCCC(=O)OP(O)(=O)OC[C@H]3O[C@H]([C@H](O)[C@@H]3O)n3cnc4c(N)ncnc34)[C@@]1([H])NC(=O)N2")
    contains_dict(name_to_smiles_dict, "Pimeloyl-[acyl-carrier protein] methyl ester", "OC(=O)C1=CN=C(O)C=C1")
    contains_dict(name_to_smiles_dict, "L-Glutamyl-tRNA(Glu)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(OC(=O)C(N)CCC(O)=O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "tRNA(Glu)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "Protoporphyrin",
                  r"CC\1=C(/C/2=C/C3=N/C(=C\C4=C(C(=C(N4)/C=C\5/C(=C(C(=N5)/C=C1\N2)C=C)C)C=C)C)/C(=C3CCC(=O)O)C)CCC(=O)O")
    contains_dict(name_to_smiles_dict, "Magnesium protoporphyrin",
                  "CC1=C(C2=CC3=NC(=CC4=C(C(=C([N-]4)C=C5C(=C(C(=N5)C=C1[N-]2)C=C)C)C=C)C)C(=C3CCC(=O)[O-])C)CCC(=O)[O-].[Mg+2]")
    contains_dict(name_to_smiles_dict, "Magnesium protoporphyrin monomethyl ester",
                  "CC1=C(C2=CC3=NC(=CC4=C(C(=C([N-]4)C=C5C(=C(C(=N5)C=C1[N-]2)C=C)C)C=C)C)C(=C3CCC(=O)O)C)CCC(=O)OC.[Mg+2]")
    contains_dict(name_to_smiles_dict, "Protochlorophyllide",
                  "CCC1=C(C2=NC1=CC3=C(C4=C([N-]3)C(=C5C(=C(C(=N5)C=C6C(=C(C(=C2)[N-]6)C=C)C)C)CCC(=O)O)[C@H](C4=O)C(=O)OC)C)C.[Mg+2]")
    contains_dict(name_to_smiles_dict, "Chlorophyllide",
                  "CCC1=C(C2=NC1=CC3=C(C4=C([N-]3)C(=C5[C@H]([C@@H](C(=N5)C=C6C(=C(C(=C2)[N-]6)C=C)C)C)CCC(=O)O)[C@H](C4=O)C(=O)OC)C)C.[Mg+2]")
    contains_dict(name_to_smiles_dict, "Adenosine 3',5'-bisphosphate",
                  "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)O)OP(=O)(O)O)O)N")
    contains_dict(name_to_smiles_dict, "DNA cytosine", "O=C1Nccc(N)n1")
    contains_dict(name_to_smiles_dict, "L-Cysteine", "[H]OC(=O)[C@@]([H])(N([H])[H])C([H])([H])S[H]")
    contains_dict(name_to_smiles_dict, "Plastoquinone-9",
                  r"CC1=C(C(=O)C(=CC1=O)C/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CC/C=C(\C)/CCC=C(C)C)C")
    contains_dict(name_to_smiles_dict, "Plastoquinol-9",
                  "CC(C)=CCC\C(C)=C\CC\C(C)=C\CC\C(C)=C\CC\C(C)=C\CC\C(C)=C\CC\C(C)=C\CC\C(C)=C\CC\C(C)=C\Cc1cc(O)c(C)c(C)c1O")
    contains_dict(name_to_smiles_dict, "Heme",
                  "OC(=O)CC/c6c(\C)c3n7c6cc2c(/CCC(O)=O)c(/C)c1cc5n8c(cc4n([Fe]78n12)c(c=3)c(C=C)c4c)c(\C=C)c5\C")
    contains_dict(name_to_smiles_dict, "Biliverdin",
                  r"CC\1=C(/C(=C/C2=C(C(=C(N2)/C=C\3/C(=C(C(=O)N3)C)C=C)C)CCC(=O)O)/N/C1=C\C4=NC(=O)C(=C4C)C=C)CCC(=O)O")
    contains_dict(name_to_smiles_dict, "(3Z)-Phycocyanobilin",
                  r"C/C=C1([C@@H](C)C(NC\1=C/C4(/NC(/C=C3(C(/CCC([O-])=O)=C(C)/C(\C=C2(C(/C)=C(CC)/C(=O)N2))=N\3))=C(CCC([O-])=O)/C(\C)=4))=O)")
    contains_dict(name_to_smiles_dict, "Acyl-carrier protein",
                  "CCC(C)C(C(=O)NC(CC(=O)O)C(=O)NC(CC1=CC=C(C=C1)O)C(=O)NC(C(C)CC)C(=O)NC(CC(=O)N)C(=O)NCC(=O)O)NC(=O)C(C)NC(=O)C(C)NC(=O)C(CCC(=O)N)NC(=O)C(C(C)C)N")
    contains_dict(name_to_smiles_dict, "3-Oxohexanoyl-[acp]", "[*]SC(=O)CC(=O)CCC")
    contains_dict(name_to_smiles_dict, "Butyryl-[acp]", "[*]SC(=O)CCC")
    contains_dict(name_to_smiles_dict, "Hexanoyl-[acp]", "CCCCCC(=O)S[*]")
    contains_dict(name_to_smiles_dict, "Tetradecanoyl-[acp]", "[*]SC(=O)CCCCCCCCCCCCC")
    contains_dict(name_to_smiles_dict, "Decanoyl-[acp]",
                  "CCCCCCCCCC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")


    contains_dict(name_to_smiles_dict, "3-Oxohexadecanoyl-[acp]", "[*]SC(=O)CC(=O)CCCCCCCCCCCCC")
    contains_dict(name_to_smiles_dict, "Acetoacetyl-[acp]", "CC(=O)CC(=O)S[*]")
    contains_dict(name_to_smiles_dict, "(3R)-3-Hydroxybutanoyl-[acyl-carrier protein]",
                  "CC(O)CC(SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O)=O")
    contains_dict(name_to_smiles_dict, "3-Oxododecanoyl-[acp]", "CCCCCCCCCC(=O)CC(=O)S[*]")
    contains_dict(name_to_smiles_dict, "D-Glucose", "C(C1C(C(C(C(O1)O)O)O)O)O")
    contains_dict(name_to_smiles_dict, "Arsenate ion", "[O-][As](=O)([O-])[O-]")
    contains_dict(name_to_smiles_dict, "N6-(L-1,3-Dicarboxypropyl)-L-lysine",
                  "C(CCNC(CCC(=O)O)C(=O)O)CC(C(=O)O)N")
    contains_dict(name_to_smiles_dict, "Cyanide ion",
                  "[C-]#N")
    contains_dict(name_to_smiles_dict, "Thiosulfate",
                  "[O-]S(=O)(=S)[O-]")
    contains_dict(name_to_smiles_dict, "Hydrogen cyanide",
                  "C#N")
    contains_dict(name_to_smiles_dict, "Hexadecanoic acid",
                  "CCCCCCCCCCCCCCCC(=O)O")
    contains_dict(name_to_smiles_dict, "Cholesterol",
                  "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C")
    contains_dict(name_to_smiles_dict, "Ethanolamine",
                  "C(CO)N")
    contains_dict(name_to_smiles_dict, "7,8-Dihydrobiopterin",
                  "CC(C(C1=NC2=C(NC1)N=C(NC2=O)N)O)O")
    contains_dict(name_to_smiles_dict, "3-Indoleacetonitrile",
                  "C1=CC=C2C(=C1)C(=CN2)CC#N")
    contains_dict(name_to_smiles_dict, "Indole",
                  "C1=CC=C2C(=C1)C=CN2")
    contains_dict(name_to_smiles_dict, "3-Oxostearoyl-[acp]",
                  "CCCCCCCCCCCCCCCC(=O)CC(SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O)=O")
    contains_dict(name_to_smiles_dict, "(S)-1-Pyrroline-5-carboxylate", "C1CC(N=C1)C(=O)O")
    contains_dict(name_to_smiles_dict, "Nitrate", "[N+](=O)([O-])[O-]")  # ?
    contains_dict(name_to_smiles_dict, "L-Proline", "C1C[C@H](NC1)C(=O)O")
    contains_dict(name_to_smiles_dict, "Malonyl-[acyl-carrier protein]",
                  "CC(C)(COP([O-])(=O)OC[C@H](N-*)C(-*)=O)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CC([O-])=O")
    contains_dict(name_to_smiles_dict, "Malonyl-[acp] methyl ester",
                  "COC(CC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O)=O")
    contains_dict(name_to_smiles_dict, "3-Ketoglutaryl-[acp] methyl ester", "*SC(=O)CC(=O)CC(=O)OC")
    contains_dict(name_to_smiles_dict, "Glutaryl-[acp] methyl ester",
                  "COC(=O)CCCC(=O)SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "Enoylglutaryl-[acp] methyl ester",
                  "COC(=O)CC=CC(=O)SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "L-Alanine", "C[C@@H](C(=O)O)N")
    # 特殊蛋白
    contains_dict(name_to_smiles_dict, "3-Hydroxyglutaryl-[acp] methyl ester",
                  "COC(=O)C[C@H](CC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O)O")
    contains_dict(name_to_smiles_dict, "Holo-[carboxylase]", "[*]NC(=O)C(CCCCNC(=O)CCCCC1C2NC(=O)NC2(CS1))NC([*])=O")
    contains_dict(name_to_smiles_dict, "3-Hydroxyechinenone",
                  r"CC1=C(C(C[C@@H](C1)O)(C)C)/C=C/C(=C/C=C/C(=C/C=C/C=C(\C)/C=C/C=C(\C)/C=C/C2=C(C(=O)CCC2(C)C)C)/C)/C")
    contains_dict(name_to_smiles_dict, "DNA 5-methylcytosine",
                  r"CC1(\C(\N)=N/C(N(\C=1)[C@@H]2(O[C@H](COP(=O)([O-])O[*])[C@@H](OP(=O)(O[*])[O-])C2))=O)")
    contains_dict(name_to_smiles_dict, "[Enzyme]-cysteine", "C(=O)([*])[C@@H](N[*])CS")
    contains_dict(name_to_smiles_dict, "[Enzyme]-S-sulfanylcysteine", "C([*])([C@@H](N[*])CSS)=O")
    contains_dict(name_to_smiles_dict, "Octanoyl-[acp]",
                  "CCCCCCCC(SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O)=O")
    contains_dict(name_to_smiles_dict, "Protein N6-(lipoyl)lysine", "C(NC(CCCC[C@@H]1(SSCC1))=O)CCC[C@H](N[*])C(=O)[*]")
    contains_dict(name_to_smiles_dict, "Protein N6-(octanoyl)lysine", "CCCCCCCC(=O)[NH2+]CCCC[C@H](N[*])C([*])=O")
    contains_dict(name_to_smiles_dict, "trans-Hexadec-2-enoyl-[acp]",
                  "CCCCCCCCCCCCC/C=C/C(SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O)=O")
    contains_dict(name_to_smiles_dict, "(2E)-Octadecenoyl-[acp]", "CCCCCCCCCCCCCCC\C=C\C(=O)S[*]")
    contains_dict(name_to_smiles_dict, "Octadecanoyl-[acyl-carrier protein]",
                  "CCCCCCCCCCCCCCCCCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP([O-])(=O)OC[C@H](N-*)C(-*)=O")
    contains_dict(name_to_smiles_dict, "Peptidoglycan",
                  "C[C@@H](C(=O)N[C@H](CON)C(=O)N[C@@H](CCCCN)C(=O)N[C@H](C)C(=O)O)NC(=O)[C@@H](C)O[C@@H]1[C@H]([C@@H](O[C@@H]([C@H]1O[C@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CO)O)O)NC(=O)C)CO)O)NC(=O)C")
    contains_dict(name_to_smiles_dict, "DialurateDialuric acid", "C1(C(=O)NC(=O)NC1=O)O")
    contains_dict(name_to_smiles_dict, "3-Hydroxypimeloyl-[acp] methyl ester",
                  "COC(=O)CCC[C@@H](O)CC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "3-Ketopimeloyl-[acp] methyl ester",
                  "COC(=O)CCCC(=O)CC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "(3R)-3-Hydroxyoctanoyl-[acyl-carrier protein]",
                  "CCCCC[C@@H](O)CC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "3-Oxooctanoyl-[acp]",
                  "CCCCCC(=O)CC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "Chlorophyll a",
                  r"CCC1=C(C2=NC1=CC3=C(C4=C([N-]3)C(=C5[C@H]([C@@H](C(=N5)C=C6C(=C(C(=C2)[N-]6)C=C)C)C)CCC(=O)OC/C=C(\C)/CCCC(C)CCCC(C)CCCC(C)C)[C@H](C4=O)C(=O)OC)C)C.[Mg+2]")
    contains_dict(name_to_smiles_dict, "Pimeloyl-[acyl-carrier protein]",
                  "C(CCCCCC([O-])=O)(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "Thioredoxin disulfide", "C([C@@H](N*)CSSC[C@@H](C(=O)*)N*)(=O)*")
    contains_dict(name_to_smiles_dict, "Ophthalmate", "CC[C@@H](C(=O)NCC(=O)[O-])NC(=O)CC[C@@H](C(=O)[O-])[NH3+]")
    contains_dict(name_to_smiles_dict, "CDP-glucose",
                  "C1=CN(C(=O)N=C1N)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)OP(=O)(O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)O)O)O)O")
    contains_dict(name_to_smiles_dict, "(R)-2-Methylmalate", "CC(CC(=O)[O-])(C(=O)[O-])O")
    contains_dict(name_to_smiles_dict, "myo-Inositol", "[2H]C1(C(C(C(C(C1([2H])O)([2H])O)([2H])O)([2H])O)([2H])O)O")
    contains_dict(name_to_smiles_dict, "Myo-Inositol", "[2H]C1(C(C(C(C(C1([2H])O)([2H])O)([2H])O)([2H])O)([2H])O)O")
    contains_dict(name_to_smiles_dict, "trans-Tetradec-2-enoyl-[acp]", r"CCCCCCCCCCC\C=C\C(=O)S[*]")
    contains_dict(name_to_smiles_dict, "trans-Dodec-2-enoyl-[acp]", r"CCCCCCCCC\C=C\C(=O)S[*]")
    contains_dict(name_to_smiles_dict, "Linoleoyl-[acyl-carrier protein]", r"CCCCC\C=C/C\C=C/CCCCCCCC(=O)S[*]")
    contains_dict(name_to_smiles_dict, "alpha-Linolenoyl-ACP", r"CC\C=C/C\C=C/C\C=C/CCCCCCCC(=O)S[*]")
    contains_dict(name_to_smiles_dict, "trans-Hex-2-enoyl-[acp]", r"CCC\C=C\C(=O)S[*]")
    contains_dict(name_to_smiles_dict, "trans-Oct-2-enoyl-[acp]",
                  "CCCCC/C=C/C(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "3-Hydroxyoctadecanoyl-[acp]", "CCCCCCCCCCCCCCCC(O)CC(=O)S[*]")
    contains_dict(name_to_smiles_dict, "(3R)-3-Hydroxypalmitoyl-[acyl-carrier protein]",
                  "CCCCCCCCCCCCCC(O)CC(SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O)=O")
    contains_dict(name_to_smiles_dict, "(3R)-3-Hydroxydecanoyl-[acyl-carrier protein]",
                  "CC(C)(COP([O-])(=O)OC[C@H](N-*)C(-*)=O)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)C[C@H](O)[*]")
    contains_dict(name_to_smiles_dict, "3-Oxodecanoyl-[acp]",
                  "CCCCCCCC(=O)CC(SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O)=O")
    contains_dict(name_to_smiles_dict, "Glycogen granule",
                  "OC[C@H]1O[C@H](OC[C@H]2O[C@H](O[C@H]3[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]3CO)[C@H](O)[C@@H](O)[C@@H]2O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O")
    contains_dict(name_to_smiles_dict, "Enoylpimeloyl-[acp] methyl ester",
                  "COC(=O)CCC/C=C/C(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict,
                  "gamma-Linolenoyl-ACPgamma-Linolenoyl-[acyl-carrier protein](6Z,9Z,12Z)-Octadecatrienoyl-ACP(6Z,9Z,12Z)-Octadecatrienoyl-[acyl-carrier protein]",
                  r"CC\C=C/C\C=C/C\C=C/CCCCCCCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(O)(=O)OP(O)(=O)OC[C@H]1O[C@H]([C@H](O)[C@@H]1OP(O)(O)=O)n1cnc2c(N)ncnc12")
    contains_dict(name_to_smiles_dict, "Hexadecenoyl-[acyl-carrier protein]", r"CCCCCC\C=C/CCCCCCCC(=O)S[*]")
    contains_dict(name_to_smiles_dict, "(9Z)-Hexadecanoyl-[acp]", "CCCCCCCCCCCCCCCC(=O)S[*]")
    contains_dict(name_to_smiles_dict, "3-oxo-cis-palm-9-eoyl-[acyl-carrier protein]",
                  "CCCCCCC=CCCCCCC(=O)CC(=O)SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "Cyanophycin polymer",
                  "C(NC(N)=[NH2+])CC[C@@H](C([O-])=O)NC(C[C@@H](C([O-])=O)[NH3+])=O")
    contains_dict(name_to_smiles_dict, "2-Amino-3-oxo-4-phosphonooxybutyrate", "C(=O)([O-])C(N)C(=O)COP(=O)([O-])[O-]")
    contains_dict(name_to_smiles_dict, "Cis-dec-3-enoyl-[acyl-carrier protein] (n-C10:1)",
                  "CCCCCCC=CCC(=O)SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "3-oxo-cis-dodec-5-enoyl-[acyl-carrier protein]",
                  "CCCCCCC=CCC(=O)CC(=O)SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "Cobamide coenzyme",
                  "[H][C@]12[C@H](CC(N)=O)[C@@]3(C)CCC(=O)NC[C@@H](C)OP([O-])(=O)O[C@@H]4[C@@H](CO)O[C@@H]([C@@H]4O)n4c[n+](c5cc(C)c(C)cc45)[Co-3]456(C[C@H]7O[C@H]([C@H](O)[C@@H]7O)n7cnc8c(N)ncnc78)N1C3=C(C)C1=[N+]4C(=CC3=[N+]5C(=C(C)C4=[N+]6[C@]2(C)[C@@](C)(CC(N)=O)[C@@H]4CCC(N)=O)[C@@](C)(CC(N)=O)[C@@H]3CCC(N)=O)C(C)(C)[C@@H]1CCC(N)=O")
    contains_dict(name_to_smiles_dict, "(9Z)-Hexadecanoyl-[acp]", "CCCCCCCCCCCCCCCC(=O)S[*]")
    contains_dict(name_to_smiles_dict, "1-hexadecanoyl-sn-glycerol 3-phosphate",
                  "[C@@H](COC(=O)CCCCCCCCCCCCCCC)(COP(OCC[N+](C)(C)C)(=O)[O-])O")
    contains_dict(name_to_smiles_dict, "Cis-octadec-11-enoyl-[acyl-carrier protein] (n-C18:1)",
                  "*[NH2+][C@@H](COP(=O)([O-])OCC(C)(C)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCCCCCCCC/C=C\CCCCCC)C(*)=O")
    contains_dict(name_to_smiles_dict, "3-oxo-cis-vacc-11-enoyl-[acyl-carrier protein]",
                  "CCCCCC\C=C/CCCCCCCC(=O)CC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "Cis-hexadec-9-enoyl-[acyl-carrier protein] (n-C16:1)",
                  "CCCCCCC=CCCCCCCCC(=O)SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "Iron chelate", "C1=CC=C(C(=C1)C=NC2=CC=CC=C2N=CC3=CC=CC=C3O)O.[Fe]")

    contains_dict(name_to_smiles_dict, "Cis-tetradec-7-enoyl-[acyl-carrier protein] (n-C14:1)",
                  r"CCCCCC\C=C/CCCCCC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "(R)-3-hydroxy-cis-myristol-7-eoyl-[acyl-carrier protein]",
                  r"CCCCCCC=CCCCC(O)CC(=O)SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "1-octadec-11-enoyl-sn-glycerol 3-phosphate",
                  r"[H]\C(CCCCCC)=C(\[H])CCCCCCCCCC(=O)OCC(O)COP([O-])([O-])=O")
    contains_dict(name_to_smiles_dict, "1-hexadec-9-enoyl-sn-glycerol 3-phosphate",
                  "CCCCCC/C=C/CCCCCCCC(=O)OC[C@@H](O)COP(=O)([O-])[O-]")
    contains_dict(name_to_smiles_dict, "1,2-dihexadec-9-enoyl-sn-glycerol 3-phosphate",
                  "CCCCCCC=CCCCCCCCC(=O)OCC(COP(=O)([O-])O)OC(=O)CCCCCCCC=CCCCCCC")
    contains_dict(name_to_smiles_dict, "Poly-beta-hydroxybutyrate", "CCC(CC(=O)O)O.CC(CC(=O)O)O")
    contains_dict(name_to_smiles_dict, "1,2-Diacyl-sn-glycerol (dioctadec-11-enoyl, n-C18:1)",
                  r"[H][C@](CO)(COC(=O)CCCCCCCCC\C=C/CCCCCC)OC(=O)CCCCCCCCC\C=C/CCCCCC")
    contains_dict(name_to_smiles_dict, "trans-Dec-2-enoyl-[acp]", "[*]SC(=O)C=CCCCCCCC")
    contains_dict(name_to_smiles_dict, "Acetyl-[acyl-carrier protein]",
                  "CC(SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O)=O")
    contains_dict(name_to_smiles_dict, "1,2-Diacyl-sn-glycerol (dioctadecanoyl, n-C18:0)",
                  "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCCCCCCCCCCCC")
    contains_dict(name_to_smiles_dict, "1,2-Diacyl-sn-glycerol (dihexadec-9-enoyl, n-C16:1)",
                  r"CCCCCC/C=C\CCCCCCCC(=O)OC[C@H](CO[Si](C)(C)C(C)(C)C)OC(=O)CCCCCCC/C=C\CCCCCC")
    contains_dict(name_to_smiles_dict, "1,2-Diacyl-sn-glycerol (dihexadecanoyl, n-C16:0)",
                  r"CCCCCCCCCCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCCCCCCCCCC")
    contains_dict(name_to_smiles_dict, "Trans-3-cis-5-dodecenoyl-[acyl-carrier protein]",
                  r"CCCCCC\C=C/C=C/CC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "Cis-dodec-5-enoyl-[acyl-carrier protein] (n-C12:1)",
                  "CCCCCCC=CCCCC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)([O-])O[*]")
    contains_dict(name_to_smiles_dict, "Trans-3-cis-11-vacceoyl-[acyl-carrier protein]",
                  r"CCCCCC\C=C/CCCCCC/C=C/CC(=O)SCCNC(=O)CCNC([C@H](O)C(C)(C)COP(=O)([O-])OC[C@H](N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "(R)-3-hydroxy-cis-dodec-5-enoyl-[acyl-carrier protein]",
                  "CCCCCCC=CCC(O)CC(=O)[*]")
    contains_dict(name_to_smiles_dict, "(R)-3-hydroxy-cis-palm-9-eoyl-[acyl-carrier protein]",
                  "CCCCCCC=CCCCCCC(O)CC(=O)SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "(R)-3-hydroxy-cis-vacc-11-enoyl-[acyl-carrier protein]",
                  "CCCCCCC=CCCCCCCCC(O)CC(=O)SCCNC(=O)CCNC(C(O)C(C)(C)COP(=O)([O-])OCC(N[*])C(=O)[*])=O")
    contains_dict(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC([*])=O")
    strict_update(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol (n-C16::0)",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC(=O)CCCCCCCCCCCCCCCC")
    strict_update(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol (n-C16::1)",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC(=O)CCCCCCCCCCCCCCCC(C=C)")
    strict_update(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol (n-C18::0)",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC(=O)CCCCCCCCCCCCCCCCCC")
    strict_update(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol (n-C18::1)",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC(=O)CCCCCCCCCCCCCCCC(C=C)")
    strict_update(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol (n-C18::1:9)",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC(=O)CCCCCCCCCCCCCCCC(C=C)")
    strict_update(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol (n-C18::2)",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC(=O)CCCCCCCCCCCCCCCC(C=C)C(C=C)")
    strict_update(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol (n-C18::3 (6,9,12))",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC(=O)CCCCCCCCCCCCCC(C=C)C(C=C)C(C=C)")
    strict_update(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol (n-C18::3 (9,12,15))",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC(=O)CCCCCCCCCCCCCC(C=C)C(C=C)C(C=C)")
    strict_update(name_to_smiles_dict, "3-D-glucosyl-1,2-Diacylglycerol (n-C18::4)",
                  "[*]C(=O)OCC(COC1OC(CO)C(O)C(O)C1(O))OC(=O)CCCCCCCCCCCCCC(C=C)C(C=C)C(C=C)C(C=C)")

    strict_update(name_to_smiles_dict, "Sulfoquinovosyldiacylglycerol",
                  r"C(OC(=O)[R1])[C@@H](OC(=O)[R2])CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Sulfoquinovosyldiacylglycerol (n-C16::0)",
                  r"C(OC(=O)CCCCCCCCCCCCCC)[C@@H](OC(=O)CCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Sulfoquinovosyldiacylglycerol (n-C16::1)",
                  r"C(OC(=O)CCCCCCCCCCCCCC(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Sulfoquinovosyldiacylglycerol (n-C18::0)",
                  r"C(OC(=O)CCCCCCCCCCCCCCCC)[C@@H](OC(=O)CCCCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Sulfoquinovosyldiacylglycerol (n-C18::1)",
                  r"C(OC(=O)CCCCCCCCCCCCCCC(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Sulfoquinovosyldiacylglycerol (n-C18::2)",
                  r"C(OC(=O)CCCCCCCCCCCCCC(C=C)C(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Sulfoquinovosyldiacylglycerol (n-C18::3 (6,9,12))",
                  r"C(OC(=O)CCCCCCCCCCCCCC(C=C)C(C=C)C(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Sulfoquinovosyldiacylglycerol (n-C18::3 (9,12,15))",
                  r"C(OC(=O)CCCCCCCCCCCCCC(C=C)C(C=C)C(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Sulfoquinovosyldiacylglycerol (n-C18::4)",
                  r"C(OC(=O)CCCCCCCCCCCCCCC(C=C)(C=C)C(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")

    strict_update(name_to_smiles_dict, "Phosphatidylglycerophosphate",
                  r"C(OP([O-])(=O)[O-])[C@@H](COP([O-])(=O)OC[C@H](OC(=O)[R2])COC(=O)[R1])O")
    strict_update(name_to_smiles_dict, "Phosphatidylglycerophosphate (dioctadec-11-enoyl, n-C18:1)",
                  r"C(OC(=O)CCCCCCCCCCCCCCC(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Phosphatidylglycerophosphate (dioctadec-9-enoyl, n-C18 1)",
                  r"C(OC(=O)CCCCCCCCCCCCCCC(C=C)[C@H])(OC(=O)CCCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Phosphatidylglycerophosphate (dioctadec-9-12-dienoyl, n-C18 2)",
                  r"C(OC(=O)CCCCCCCCCCCCCC(C=C)C(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Phosphatidylglycerophosphate (dioctadec-6-9-12-15-tetraenoyl, n-C18 4)",
                  r"C(OC(=O)CCCCCCCCCCCCCC(C=C)(C=C)C(C=C)C(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Phosphatidylglycerophosphate (dioctadec-6-9-12-trienoyl, n-C18 3)",
                  r"C(OC(=O)CCCCCCCCCCCCCC(C=C)(C=C)C(C=C))[C@@H](OC(=O)CCCCCCCCCCCCCC)CO[C@@H]1([C@H](O)[C@H]([C@H](O)[C@@H](CS(=O)(=O)[O-])O1)O)")
    strict_update(name_to_smiles_dict, "Reduced thioredoxin",
                  "None")
    strict_update(name_to_smiles_dict, "",
                  "None")
    strict_update(name_to_smiles_dict, "Acetyl-ACP",
                  "None")
    contains_dict(name_to_smiles_dict, "Adenosyl cobinamide",
                  "[H][C@@]12[C@H](CC(N)=O)[C@@](C)(CCC(=O)NC[C@@H](C)O)C3=C(C)C4=[N+]5C(=CC6=[N+]7C(=C(C)C8=[N+](["
                  "C@]1(C)[C@@](C)(CC(N)=O)[C@@H]8CCC(N)=O)[Co--]57(C[C@H]1O[C@H]([C@H](O)[C@@H]1O)n1cnc5c("
                  "N)ncnc15)N23)[C@@](C)(CC(N)=O)[C@@H]6CCC(N)=O)C(C)(C)[C@@H]4CCC(N)=O")
    contains_dict(name_to_smiles_dict, "Tetrahydrobiopterin",
                  "CC(C(C1CNC2=C(N1)C(=O)NC(=N2)N)O)O")
    contains_dict(name_to_smiles_dict, "Diphosphate",
                  "[O-]P(=O)([O-])OP(=O)([O-])[O-]")
    contains_dict(name_to_smiles_dict, "Oxygen", "O=O")

    contains_dict(name_to_smiles_dict, "N-Acetyl-D-glucosamine 1-phosphate", "CC(=O)NC1C(C(C(OC1OP(=O)(O)O)CO)O)O")
    contains_dict(name_to_smiles_dict, "All-trans-Nonaprenyl diphosphate",
                  "CC(=CCCC(=CCCC(=CCCC(=CCCC(=CCCC(=CCCC(=CCCC(=CCCC(=CCOP(=O)(O)OP(=O)(O)O)C)C)C)C)C)C)C)C)C")
    contains_dict(name_to_smiles_dict, "Adenosyl cobyrinate diamide",
                  "CC1=C2C(C(C([N-]2)C3(C(C(C(=N3)C(=C4C(C(C(=N4)C=C5C(C(C1=N5)CCC(=O)O)(C)C)CCC(=O)O)(C)CC("
                  "=O)N)C)CCC(=O)O)(C)CC(=O)N)C)CC(=O)O)(C)CCC(=O)O.[CH2-]C1C(C(C(O1)N2C=NC3=C(N=CN=C32)N)O)O.[Co+3]")
    contains_dict(name_to_smiles_dict, "Dimethylbenzimidazole", "CC1=CC2=C(C=C1C)N=CN2")
    contains_dict(name_to_smiles_dict, "TRNA (Glu)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Thr)",
                  "[*]C3OC(COP(O)(=O)OC2C(O)C([*])OC2(COP(O)(=O)OC1C(O)C([*])OC1(CO)))C(O)C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Ala)",
                  "[*]C3OC(COP(O)(=O)OC2C(O)C([*])OC2(COP(O)(=O)OC1C(O)C([*])OC1(CO)))C(O)C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Phe)",
                  "[*]C3OC(COP(O)(=O)OC2C(O)C([*])OC2(COP(O)(=O)OC1C(O)C([*])OC1(CO)))C(O)C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Arg)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Leu)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Ile)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Pro)",
                  "[*]C3OC(COP(O)(=O)OC2C(O)C([*])OC2(COP(O)(=O)OC1C(O)C([*])OC1(CO)))C(O)C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(His)",
                  "[*]C3OC(COP(O)(=O)OC2C(O)C([*])OC2(COP(O)(=O)OC1C(O)C([*])OC1(CO)))C(O)C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Gly)",
                  "[*]C3OC(COP(O)(=O)OC2C(O)C([*])OC2(COP(O)(=O)OC1C(O)C([*])OC1(CO)))C(O)C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Val)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Met)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Tyr)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Lys)",
                  "[*]C3OC(COP(O)(=O)OC2C(O)C([*])OC2(COP(O)(=O)OC1C(O)C([*])OC1(CO)))C(O)C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Ser)",
                  "[*]C3OC(COP(O)(=O)OC2C(O)C([*])OC2(COP(O)(=O)OC1C(O)C([*])OC1(CO)))C(O)C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Asp)",
                  "[*]C3OC(COP(O)(=O)OC2C(O)C([*])OC2(COP(O)(=O)OC1C(O)C([*])OC1(CO)))C(O)C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Trp)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "L-Aspartyl-tRNA(Asn)",
                  "[*]C3OC(CO)C(OP(O)(=O)OCC2OC([*])C(O)C2(OP(O)(=O)OCC1OC([*])C(O)C1(OC(=O)C(N)CC(O)=O)))C3(O)")
    contains_dict(name_to_smiles_dict, "TRNA(Cys)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "L-Glutamyl-tRNA(Gln)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(OC(=O)C(N)CCC(O)=O)C1(O))n3cnc2c(N)ncnc23))C5(O)")
    contains_dict(name_to_smiles_dict, "L-Methionyl-tRNA (Met)",
                  "[*]C5OC(CO)C(OP(O)(=O)OCC4OC([*])C(O)C4(OP(O)(=O)OCC1OC(C(OC(=O)C(N)CCSC)C1(O))n3cnc2c(N)ncnc23))C5(O)")

    contains_dict(name_to_smiles_dict, "Adenosylcobalamin",
                  r"CC1=CC2=C(C=C1C)N(C=N2)[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)OP(=O)([O-])O[C@H](C)CNC(=O)CC[C@@]\4(["
                  r"C@H]([C@@H]5[C@]6([C@@]([C@@H](C(=N6)/C(=C\7/[C@@]([C@@H](C(=N7)/C=C\8/C([C@@H](C(=N8)/C(=C4\["
                  r"N-]5)/C)CCC(=O)N)(C)C)CCC(=O)N)(C)CC(=O)N)/C)CCC(=O)N)(C)CC(=O)N)C)CC(=O)N)C)O.[CH2-][C@@H]1["
                  r"C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)O.[Co]")
    contains_dict(name_to_smiles_dict, "Deoxyadenosine",
                  "C1C(C(OC1N2C=NC3=C(N=CN=C32)N)CO)O")
    contains_dict(name_to_smiles_dict, "5-Carboxyamino-1-(5-phospho-D-ribosyl)imidazole",
                  "C1=C(N(C=N1)C2C(C(C(O2)COP(=O)(O)O)O)O)NC(=O)O")
    contains_dict(name_to_smiles_dict, "D-Glucono-1,5-lactone",
                  "C(C1C(C(C(C(=O)O1)O)O)O)O")
    contains_dict(name_to_smiles_dict, "Thiocyanate",
                  "C(#N)[S-]")
    contains_dict(name_to_smiles_dict, "Dihydrobiopterin",
                  "CC(C(C1=NC2=C(NC1)N=C(NC2=O)N)O)O")
    contains_dict(name_to_smiles_dict, "(R)-3-Hydroxydodecanoyl-[acyl-carrier protein]", "CCCCCCCCCC(CC(=O)S)O")
    contains_dict(name_to_smiles_dict, " 13(1)-Hydroxy-magnesium-protoporphyrin IX 13-monomethyl ester",
                  "CC1=C(C2=CC3=NC(=CC4=C(C(=C([N-]4)C=C5C(=C(C(=N5)C=C1[N-]2)C)CCC(=O)O)C(CC(=O)OC)O)C)C(=C3C)C=C)C=C.[Mg+2]")
    contains_dict(name_to_smiles_dict, "(R)-3-Hydroxyhexanoyl-[acyl-carrier protein]",
                  "CCCC(CC(=O)S)O")
    contains_dict(name_to_smiles_dict, "Nicotinate",
                  "C1=CC(=CN=C1)C(=O)[O-]")
    contains_dict(name_to_smiles_dict, "L-Phenylalanyl-tRNA(Phe)",
                  "[*]C4OC(CO)C(OP(O)(=O)OCC3OC([*])C(O)C3(OP(O)(=O)OCC2OC([*])C(O)C2(OC(=O)C(N)Cc1ccccc1)))C4(O)")
    # contains_dict()
    # contains_dict(name_to_smiles_dict, "1,2-Diacyl-sn-glycerol",
    #               r"OC[C@@H](COC([*])=O)OC([*])=O")


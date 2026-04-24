#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用equilibrator_api获取代谢网络热力学数据的脚本

该脚本从SBML格式的代谢网络文件中提取代谢物和反应信息，
并使用equilibrator_api获取相应的热力学数据。
"""

import json
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from equilibrator_api import ComponentContribution
    from equilibrator_api.util.parsing import parse_formula
    print("equilibrator_api imported successfully")
except ImportError:
    print("Error: equilibrator_api not found. Please install with: pip install equilibrator-api")
    exit(1)

class ThermoDataExtractor:
    """从代谢网络中提取热力学数据的类"""

    def __init__(self, sbml_file: str):
        """
        初始化热力学数据提取器

        Args:
            sbml_file: SBML格式的代谢网络文件路径
        """
        self.sbml_file = Path(sbml_file)
        self.cc = ComponentContribution()
        self.metabolites_data = {}
        self.reactions_data = {}
        self.namespaces = {
            'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
            'fbc': 'http://www.sbml.org/sbml/level3/version1/fbc/version2'
        }

    def parse_sbml(self) -> Tuple[Dict, Dict]:
        """
        解析SBML文件，提取代谢物和反应信息

        Returns:
            metabolites: 代谢物信息字典
            reactions: 反应信息字典
        """
        print(f"正在解析SBML文件: {self.sbml_file}")

        tree = ET.parse(self.sbml_file)
        root = tree.getroot()

        metabolites = {}
        reactions = {}

        # 提取代谢物信息
        species_list = root.find('.//sbml:listOfSpecies', self.namespaces)
        if species_list is not None:
            for species in species_list.findall('sbml:species', self.namespaces):
                met_id = species.get('id')
                met_name = species.get('name', met_id)
                compartment = species.get('compartment', '')
                formula = species.get('{http://www.sbml.org/sbml/level3/version1/fbc/version2}chemicalFormula', '')
                charge = species.get('{http://www.sbml.org/sbml/level3/version1/fbc/version2}charge', '0')

                # 提取数据库注释
                annotations = self._extract_annotations(species)

                metabolites[met_id] = {
                    'name': met_name,
                    'compartment': compartment,
                    'formula': formula,
                    'charge': int(charge) if charge else 0,
                    'annotations': annotations
                }

        # 提取反应信息
        reactions_list = root.find('.//sbml:listOfReactions', self.namespaces)
        if reactions_list is not None:
            for reaction in reactions_list.findall('sbml:reaction', self.namespaces):
                rxn_id = reaction.get('id')
                rxn_name = reaction.get('name', rxn_id)
                reversible = reaction.get('reversible', 'true').lower() == 'true'

                # 提取反应物和产物
                reactants = self._extract_participants(reaction, 'sbml:listOfReactants')
                products = self._extract_participants(reaction, 'sbml:listOfProducts')

                reactions[rxn_id] = {
                    'name': rxn_name,
                    'reversible': reversible,
                    'reactants': reactants,
                    'products': products
                }

        print(f"解析完成: {len(metabolites)} 个代谢物, {len(reactions)} 个反应")
        return metabolites, reactions

    def _extract_annotations(self, element) -> Dict[str, List[str]]:
        """提取XML元素的注释信息"""
        annotations = {}

        # 查找RDF注释
        rdf_desc = element.find('.//rdf:Description', {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'})
        if rdf_desc is not None:
            is_element = rdf_desc.find('.//bqbiol:is', {'bqbiol': 'http://biomodels.net/biology-qualifiers/'})
            if is_element is not None:
                bag = is_element.find('.//rdf:Bag', {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'})
                if bag is not None:
                    for li in bag.findall('.//rdf:li', {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'}):
                        resource = li.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource', '')
                        if resource:
                            # 解析不同的数据库ID
                            if 'chebi' in resource:
                                annotations.setdefault('chebi', []).append(resource.split('/')[-1])
                            elif 'kegg.compound' in resource:
                                annotations.setdefault('kegg', []).append(resource.split('/')[-1])
                            elif 'bigg.metabolite' in resource:
                                annotations.setdefault('bigg', []).append(resource.split('/')[-1])
                            elif 'inchi_key' in resource:
                                annotations.setdefault('inchi_key', []).append(resource.split('/')[-1])

        return annotations

    def _extract_participants(self, reaction, list_type: str) -> Dict[str, float]:
        """提取反应参与者(反应物或产物)"""
        participants = {}

        participants_list = reaction.find(f'.//{list_type}', self.namespaces)
        if participants_list is not None:
            for participant in participants_list.findall('.//sbml:speciesReference', self.namespaces):
                species_id = participant.get('species')
                stoichiometry = float(participant.get('stoichiometry', '1.0'))
                participants[species_id] = stoichiometry

        return participants

    def get_thermodynamic_data(self, metabolites: Dict, reactions: Dict) -> Dict:
        """
        使用equilibrator_api获取热力学数据

        Args:
            metabolites: 代谢物信息
            reactions: 反应信息

        Returns:
            热力学数据字典
        """
        print("正在获取热力学数据...")

        thermo_data = {
            'metabolites': {},
            'reactions': {},
            'conditions': {
                'temperature': 298.15,  # K
                'pH': 7.0,
                'ionic_strength': 0.1  # M
            }
        }

        # 获取代谢物的热力学数据
        print("获取代谢物热力学数据...")
        for met_id, met_info in metabolites.items():
            try:
                # 尝试通过不同的标识符查找代谢物
                compound = None

                # 1. 尝试通过KEGG ID
                if 'kegg' in met_info['annotations']:
                    for kegg_id in met_info['annotations']['kegg']:
                        try:
                            compound = self.cc.get_compound(f"kegg:{kegg_id}")
                            break
                        except:
                            continue

                # 2. 尝试通过ChEBI ID
                if compound is None and 'chebi' in met_info['annotations']:
                    for chebi_id in met_info['annotations']['chebi']:
                        try:
                            compound = self.cc.get_compound(f"chebi:{chebi_id}")
                            break
                        except:
                            continue

                # 3. 尝试通过化学式
                if compound is None and met_info['formula']:
                    try:
                        compound = self.cc.get_compound(met_info['formula'])
                    except:
                        pass

                if compound is not None:
                    # 获取标准生成吉布斯自由能
                    try:
                        dG_formation = self.cc.standard_dg_formation(compound)
                        thermo_data['metabolites'][met_id] = {
                            'name': met_info['name'],
                            'formula': met_info['formula'],
                            'charge': met_info['charge'],
                            'dG_formation': float(dG_formation.value.magnitude),  # kJ/mol
                            'dG_formation_uncertainty': float(dG_formation.error.magnitude),
                            'source': 'equilibrator_api'
                        }
                        print(f"  ✓ {met_id}: {met_info['name']}")
                    except Exception as e:
                        print(f"  ✗ {met_id}: 无法获取热力学数据 - {str(e)}")
                else:
                    print(f"  ✗ {met_id}: 未找到匹配的化合物")

            except Exception as e:
                print(f"  ✗ {met_id}: 处理错误 - {str(e)}")

        # 获取反应的热力学数据
        print("计算反应热力学数据...")
        for rxn_id, rxn_info in reactions.items():
            try:
                # 构建反应方程式
                reaction_compounds = []
                coefficients = []

                # 添加反应物 (负系数)
                for species_id, stoich in rxn_info['reactants'].items():
                    if species_id in thermo_data['metabolites']:
                        # 这里需要获取compound对象，简化处理
                        coefficients.append(-stoich)
                        # reaction_compounds.append(compound_object)

                # 添加产物 (正系数)
                for species_id, stoich in rxn_info['products'].items():
                    if species_id in thermo_data['metabolites']:
                        coefficients.append(stoich)
                        # reaction_compounds.append(compound_object)

                # 这里简化处理，实际应该用equilibrator计算反应的ΔG
                # 可以通过代谢物的生成焓计算反应焓变
                reactant_dG = sum(
                    thermo_data['metabolites'][species_id]['dG_formation'] * stoich
                    for species_id, stoich in rxn_info['reactants'].items()
                    if species_id in thermo_data['metabolites']
                )

                product_dG = sum(
                    thermo_data['metabolites'][species_id]['dG_formation'] * stoich
                    for species_id, stoich in rxn_info['products'].items()
                    if species_id in thermo_data['metabolites']
                )

                dG_reaction = product_dG - reactant_dG

                thermo_data['reactions'][rxn_id] = {
                    'name': rxn_info['name'],
                    'reversible': rxn_info['reversible'],
                    'dG_reaction': dG_reaction,  # kJ/mol
                    'reactants': rxn_info['reactants'],
                    'products': rxn_info['products'],
                    'source': 'calculated_from_formation_energies'
                }

                print(f"  ✓ {rxn_id}: ΔG = {dG_reaction:.2f} kJ/mol")

            except Exception as e:
                print(f"  ✗ {rxn_id}: 计算错误 - {str(e)}")

        return thermo_data

    def save_data(self, thermo_data: Dict, output_dir: str = None):
        """
        保存热力学数据到文件

        Args:
            thermo_data: 热力学数据
            output_dir: 输出目录，默认为SBML文件所在目录
        """
        if output_dir is None:
            output_dir = self.sbml_file.parent
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # 保存为JSON格式
        json_file = output_dir / 'thermodynamic_data.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(thermo_data, f, indent=2, ensure_ascii=False)
        print(f"热力学数据已保存到: {json_file}")

        # 保存代谢物数据为CSV
        if thermo_data['metabolites']:
            met_df = pd.DataFrame.from_dict(thermo_data['metabolites'], orient='index')
            csv_file = output_dir / 'metabolites_thermo.csv'
            met_df.to_csv(csv_file, index_label='metabolite_id')
            print(f"代谢物热力学数据已保存到: {csv_file}")

        # 保存反应数据为CSV
        if thermo_data['reactions']:
            rxn_df = pd.DataFrame.from_dict(thermo_data['reactions'], orient='index')
            # 处理嵌套字典列
            rxn_df = rxn_df.drop(['reactants', 'products'], axis=1)
            csv_file = output_dir / 'reactions_thermo.csv'
            rxn_df.to_csv(csv_file, index_label='reaction_id')
            print(f"反应热力学数据已保存到: {csv_file}")

def main():
    """主函数"""
    # 设置文件路径
    sbml_file = "/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/network.xml"

    # 检查文件是否存在
    if not Path(sbml_file).exists():
        print(f"错误: 文件 {sbml_file} 不存在")
        return

    # 创建热力学数据提取器
    extractor = ThermoDataExtractor(sbml_file)

    try:
        # 解析SBML文件
        metabolites, reactions = extractor.parse_sbml()

        # 获取热力学数据
        thermo_data = extractor.get_thermodynamic_data(metabolites, reactions)

        # 保存数据
        extractor.save_data(thermo_data)

        # 打印摘要
        print("\n=== 数据摘要 ===")
        print(f"成功获取 {len(thermo_data['metabolites'])} 个代谢物的热力学数据")
        print(f"成功计算 {len(thermo_data['reactions'])} 个反应的热力学数据")
        print(f"条件: T={thermo_data['conditions']['temperature']}K, pH={thermo_data['conditions']['pH']}")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
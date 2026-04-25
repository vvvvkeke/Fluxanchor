# 该py用于检查路径中的文件夹是否存在(若不存在则创建)、检查路径中的文件是否存在(若存在则删除后重建，若不存在直接重建)

import os
import json


def create_directory_if_not_exists(file_path):
    """
    Summary:该函数用于检查输入的文件路径中的目录是否存在, 若存在则不做任何操作, 若不存在则创建
    """

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} has been created.")
    else:
        print(f"Directory {directory} already exists.")


def delete_file_if_exists_create_if_not(file_path):
    """
    Summary:该函数用于检查输入的路径中的文件是否存在,若存在则删除后创建一个新的空文件, 若不存在则直接创建一个新的空文件
    """

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} has been deleted.")

    if file_path.endswith(".json"):
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump([], file)
        print(f"Empty JSON file {file_path} has been created.")
    else:
        with open(file_path, "w", encoding="utf-8") as file:
            pass
        print(f"Empty File {file_path} has been created.")
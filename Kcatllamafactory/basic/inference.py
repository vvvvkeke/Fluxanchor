# 该py用于为LLM加载lora checkpoint(若无lora checkpoint，则不加载），然后进行推理

import re
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
import basic
import numpy as np
    

def get_response(system_content, information, model, tokenizer, gpu):
    """
    Summary:该函数用于向LLM传入query, 返回LLM的response, 作一般的inference使用
    """

    messages = [
        # {
        #     "role": "system",
        #     "content": system_content
        # },
        {   
            "role": "user", 
            "content": system_content + information
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(gpu)
    
    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=4096,
            temperature=0.6,
            top_p=0.9
        )
        
    response_generated_ids = []
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids):
        generated_tokens = output_ids[len(input_ids):]
        response_generated_ids.append(generated_tokens)
    response = tokenizer.batch_decode(response_generated_ids, skip_special_tokens=True)[0]
    torch.cuda.empty_cache()
    return response


def extract_kcat_km(system_content, user_content, model, tokenizer, gpu):
    """
    使用正则表达式从模型输出中提取kcat和km的值。
    """
    # pattern = r'\[(\d+\.\d+),\s*(\d+\.\d+)\]'
    pattern = r'^\[\d+\.\d+,\s*\d+\.\d+\]$'
    j = 0
    while j < 10: 
        response = get_response(system_content, user_content, model, tokenizer, gpu)
        match = re.search(pattern, response)
        if match:  
            return response
        j += 1
        print(f"Retry times: {j} | Response: {response}")
    return str([0, 0])

def inference(input_file, output_file, model_path, peft_model, gpu,output_vector,logging):
    """
    Summary:该函数用于对proofnet的每行内容作操作: 传入LLM并得到相应的response, 并将其写入到新的txt中
    """

    basic.file_directory.create_directory_if_not_exists(output_file)
    basic.file_directory.delete_file_if_exists_create_if_not(output_file)

    basic.file_directory.create_directory_if_not_exists(output_vector)
    basic.file_directory.delete_file_if_exists_create_if_not(output_vector)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=gpu, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if peft_model != "None":
        model = PeftModel.from_pretrained(model, peft_model)
        print(f"加载Lora checkpoint: {peft_model}")
    else:
        print("无Lora checkpoint加载")
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    with open(input_file, "r", encoding="utf-8") as input_file, open(output_file, "w", encoding="utf-8") as output_file:
        data = json.load(input_file)        
        for i in range(len(data)):
            system_content = data[i]["instruction"]
            information = data[i]["input"]
            user_content = information
            response = extract_kcat_km(system_content, user_content, model, tokenizer, gpu)
            print(f"Process: {i+1}/{len(data)} | Response: {response}")
            if response == str([0, 0]):
                logging.info((f"处理第{i}条数据,uniprot_id为{data[i]['uniprot_id']}, Response: {response}"))
            response_data = {}
            response_data["information"] = information
            response_data["label"] = data[i]["output"]
            response_data["predict"] = response
            response_data["index"] = data[i]["index"]
            json_line = json.dumps(response_data, ensure_ascii=False)
            output_file.write(json_line+"\n")

# 该py用于手动对LLM进行Lora SFT

# Part1: 导入包
import torch
import pandas as pd
from datasets import Dataset
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

# Part2: 设置显卡 --> 下载模型 --> 加载模型
snapshot_download("./LLM-Research/Meta-Llama-3.1-8B-Instruct", cache_dir="./model/")
model = AutoModelForCausalLM.from_pretrained("./model/LLM-Research", device_map="cuda", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("./model/LLM-Research")
tokenizer.pad_token = tokenizer.eos_token
model.enable_input_require_grads()

# Part3: 对数据作预处理
def process_func(data):
    input = [
        {
            "role": "system",
            "content": data["instruction"]
        },
        {   
            "role": "user", 
            "content": data["input"]
        }
    ]
    output = [
        {
            "role": "assistant",
            "content": data["reoutput"]
        }
    ]
    
    input = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
    input = tokenizer(input, add_special_tokens=False)
    output = tokenizer.apply_chat_template(output, tokenize=False, add_generation_prompt=False)
    output = output.replace("<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n", "")
    output = tokenizer(output, add_special_tokens=False)
    
    input_ids = input.input_ids + output.input_ids
    attention_mask = input.attention_mask + output.attention_mask
    labels = [-100] * len(input.input_ids) + output.input_ids
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

df = pd.read_json("./ceshi.json")
ds = Dataset.from_pandas(df)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# Part4: 配置Lora参数 --> 将Lora加载到basemodel中以创建等待微调的模型
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# Part5: 设置训练参数和训练器 --> 开启训练
training_args = TrainingArguments(
    output_dir="./output/llama3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=1,
    save_steps=20000,
    save_on_each_node=True,
    gradient_checkpointing=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
trainer.train()

# Part6: 保留训练后得到的lora权重
lora_path = "./llama3_lora"
trainer.model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
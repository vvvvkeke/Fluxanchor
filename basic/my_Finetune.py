import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM,TrainingArguments,Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig,TaskType, get_peft_model
from datasets import Dataset

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDApip i
        with torch.cuda.device('cuda:1'):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

#数据预处理
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
            "content": data["output"]
        }
    ]

    input = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
    input_tokenized = tokenizer(input, add_special_tokens=False)
    output = tokenizer.apply_chat_template(output, tokenize=False, add_generation_prompt=False)
    output = output.replace("<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n", "")
    output_tokenized = tokenizer(output, add_special_tokens=False)

    input_ids = input_tokenized.input_ids + output_tokenized.input_ids
    attention_mask = input_tokenized.attention_mask + output_tokenized.attention_mask
    labels = [-100] * len(input_tokenized.input_ids) + output_tokenized.input_ids
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
            }


# 设置显卡 --> 加载模型
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
pretrained_model_path = os.path.join(os.path.dirname(__file__), 'model/LLaMA3/Meta-Llama-3-8B-Instruct')
pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, device_map="auto",torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
pretrained_model.enable_input_require_grads()

data_path =  './dataset/merged.json'
df = pd.read_json(data_path) 
ds = Dataset.from_pandas(df)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# 配置Lora参数 --> 将Lora加载到basemodel中以创建等待微调的模型
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #需要训练的模型层的名字，主要就是attention部分的层，不同的模型对应的层的名字不同，可以传入数组，也可以字符串，也可以正则表达式。
    inference_mode=False, 
    r=8, # Lora 秩
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(pretrained_model, lora_config)
model.print_trainable_parameters()

# 自定义 TrainingArguments 参数
args = TrainingArguments(
    output_dir= os.path.join(os.path.dirname(__file__),"Checkpoint/model/LLaMA3"), 
    overwrite_output_dir = True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=25, 
    # logging_steps=5, #多少步，输出一次log。这里的步指的是进行这样的迭代（前向传播、反向传播、参数更新）
    num_train_epochs=5, #epoch
    save_steps=2, #这里每2步保存一次模型,包括最后一步
    learning_rate=1e-4,
    save_on_each_node=True, #在使用多节点（即多台机器）训练时，此选项确保每个节点都会保存模型。这在分布式训练环境中非常有用。
    gradient_checkpointing=True, #梯度检查，这个一旦开启，模型就必须执行model.enable_input_require_grads()
)

#使用 Trainer 训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()

# 保存 lora 权重
save_lorapath = os.path.join(os.path.dirname(__file__), 'lora_model/LLaMA3')
trainer.model.save_pretrained(save_lorapath)
tokenizer.save_pretrained(save_lorapath)

torch_gc()  # 执行GPU内存清理
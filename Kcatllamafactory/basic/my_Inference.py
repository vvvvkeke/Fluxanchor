#加载 lora 权重推理
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig,TaskType,AutoPeftModelForCausalLM
import torch,os,time
from peft import PeftModel, PeftConfig

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device('cuda:0'):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

def get_response(question, model, tokenizer, gpu):
    """_summary_该函数用于向LLM传入query, 返回LLM的response, 作一般的inference使用

    Args:
        question (str): 传入的query内容
        model (str): 使用的LLM
        tokenizer (transformers.PreTrainedTokenizer): 使用的分词器
        gpu (_type_): 使用的gpu设备序号

    Returns:
        str: LLM的回答
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert in biochemistry and metabolic pathways. You will be given a metabolic pathway task, and you must provide the steps and enzymes required to achieve the given biochemical transformation. Please annotate each compound in Chinese."
        },
        {   
            "role": "user", 
            "content": question
        }
                ]
    input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_tokenized = tokenizer(input, add_special_tokens=False, return_tensors="pt").to(gpu)

    with torch.no_grad():
        generated_ids = model.generate(
            input_tokenized.input_ids,
            attention_mask=input_tokenized.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            max_new_tokens=4096,
            temperature=0.6,
            top_p=0.9
        )
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(input_tokenized.input_ids, generated_ids)
]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

start_time = time.time()
lora_path = os.path.join(os.path.dirname(__file__), 'llama3_lora_model')
lora_model = AutoPeftModelForCausalLM.from_pretrained(lora_path, device_map="cuda:0",torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(lora_path)
lora_model.eval()

inference_data_path =  os.path.join(os.path.dirname(__file__), 'dataset/Inference.txt')
output_file_path = os.path.join(os.path.dirname(__file__), 'Inference_Result/llama3_lora.txt')
output_dir = os.path.dirname(output_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

gpu = 'cuda:0'
with open(inference_data_path, "r", encoding="utf-8") as file:
        i = 0
        for line in file:
            question = str(line[:-1])
            response = get_response(question, lora_model, tokenizer, gpu)
            i = i + 1
            response = f"## Answer {i}: " + response
            print(response)
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(response)
                output_file.write("\n\n")
print("Inference work has done.")
end_time = time.time()
print(end_time-start_time)

torch_gc()  # 执行GPU内存清理


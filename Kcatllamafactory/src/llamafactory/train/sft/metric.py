# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available
from scipy.stats import pearsonr
import os,re

#-- inference内容
def find_largest_checkpoint(directory):
    """
    Summary: 该函数用于查找最新的checkpoint
    """

    pattern = re.compile(r'^checkpoint-(\d+)$')
    checkpoints = []

    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path):
            match = pattern.match(item)
            if match:
                number = int(match.group(1))
                checkpoints.append((number, item))

    if checkpoints:
        largest_checkpoint = max(checkpoints, key=lambda x: x[0])
        return largest_checkpoint[0]
    else:
        return 0
    
if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge

#logits模型的原始输出，它是一个未经过 softmax 函数处理的张量。(batch_size, seq_len, vocab_size),表示每个样本的每个位置的所有类别的预测得分
def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)): #模型可能输出多个信息：一个是主要的预测结果，另一个可能是辅助损失（例如在 moe 模型中)
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0] #如果 logits[0] 的维度是 3，这表示 logits[0] 是我们需要的主要输出，包含了模型预测的 logits。此时，代码会将 logits 赋值为 logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3: #确保 logits 的维度是 (batch_size, seq_len, vocab_size)，即每个 token 对应一个得分向量。如果维度不符合要求，抛出异常。
        raise ValueError("Cannot process the logits.")
    
    return torch.argmax(logits, dim=-1)#每个 token 找到预测得分最大的类别。


#将该类自动转换为一个数据类，提供了 __init__、__repr__ 等功能。
@dataclass
class ComputeAccuracy: #用于计算 准确度，并且支持批量评估
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]: #在计算准确度后，会将准确度存储在 score_dict 中，并返回该字典的平均值
        result = None
        if hasattr(self, "score_dict"): #它检查 self（即 ComputeAccuracy 实例）是否有 score_dict 属性
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self): #它会在 __init__ 方法完成后被自动调用.在这里目的是初始化 score_dict 并确保它在对象创建时被清空。
        self._dump() 

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]: 
        # __call__ 使得 ComputeAccuracy 类的实例可以像函数一样被调用。它计算给定预测值与标签之间的准确度。
        # eval_preds 是一个包含模型预测和真实标签的对象。通常是 EvalPrediction 类型。其中包含 predictions（模型的预测）和 label_ids（真实标签）
        # eval_preds.predictions模型的输出预测值，通常是一个二维数组（batch size x sequence length）。
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:] #将预测和标签切割成合适的部分，排除掉最后一个预测值和第一个标签值。
            label_mask = label != IGNORE_INDEX #它会检查每个标签是否等于 IGNORE_INDEX，如果标签不等于 IGNORE_INDEX，则对应的掩码值为 True，否则为 False。
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity: #这部分代码的主要作用是计算文本生成任务中的 文本相似度分数，具体来说，它计算了 ROUGE 和 BLEU 等常用的自然语言处理评估指标。
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer" #对预测文本和参考文本进行解码。

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id) #np.where(condition, x, y) 函数，condition 为 True，则返回 x
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id) #如果某个位置的值不等于 IGNORE_INDEX，就保持该位置原有的值。
#如果某个位置的值等于 IGNORE_INDEX，就将其替换为填充符号 pad_token_id。

        #将预测和标签的 ID 转换为字符串（即将标记还原为单词或句子）。
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4)) #v["f"]: 这是从 rouge.get_scores() 返回的分数字典中获取的 F1 分数。ROUGE 分数通常会返回一个字典，包含多个分数（如 f（F1 分数）、p（精确度）、r（召回率）等），这里使用的是 F1 分数。

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()
        





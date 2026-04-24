# Copyright 2024 the LlamaFactory team.
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

import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.logging import get_logger
from ..hparams import get_infer_args, get_train_args
from ..model import load_model, load_tokenizer
from .callbacks import LogCallback
from .dpo import run_dpo
from .kto import run_kto
from .ppo import run_ppo
from .pt import run_pt
from .rm import run_rm
from .sft import run_sft


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    if finetuning_args.stage == "pt": #运行预训练阶段 模型在大量数据上进行训练，以学习通用的语言或领域知识。通常涉及的大规模训练数据和较长的训练时间。
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "sft": #运行有监督的微调 当你希望让模型在特定任务上（如文本分类、命名实体识别、翻译等）进行优化时，使用有监督微调。
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "rm": #运行奖励模型微调 通常在 RL 或其他需要基于反馈进行优化的任务中使用 RM 微调方法。 与传统监督学习不同，RM 更侧重于基于奖励的反馈，而非直接的标签数据。奖励信号通常来自环境（例如，通过与用户的互动或模拟环境的反馈）。任务不直接给出标注数据，而是通过与环境互动获得反馈的任务。典型的任务包括对话系统、游戏智能体、自动驾驶等。
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "ppo": #运行 Proximal Policy Optimization  PPO 使用了一个近端策略优化算法，重点是平衡探索和利用，通过限制策略变化来避免过大的更新。通常用来训练智能体在强化学习环境中最大化奖励。
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo": #运行 DPO 微调 适用于需要模型基于偏好做出决策的任务，尤其是在需要排序或决策任务的场景（例如推荐系统、决策支持系统）。
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    elif finetuning_args.stage == "kto": #运行 Knowledge Transfer Optimization 微调 当你需要从一个任务或领域中迁移知识到另一个任务或领域时，KTO 是一个非常合适的方法。KTO 是通过在源任务中训练得到的知识来指导目标任务的训练，通常可以显著提高目标任务的训练效率，尤其是在数据稀缺的情况下。KTO 主要用于迁移学习。
        run_kto(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError("Unknown task: {}.".format(finetuning_args.stage))


def export_model(args: Optional[Dict[str, Any]] = None) -> None: #保存并导出微调后的模型
    model_args, data_args, finetuning_args, _ = get_infer_args(args)

    if model_args.export_dir is None:
        raise ValueError("Please specify `export_dir` to save model.")

    if model_args.adapter_name_or_path is not None and model_args.export_quantization_bit is not None:
        raise ValueError("Please merge adapters before quantizing the model.")

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args)  # must after fixing tokenizer to resize vocab

    if getattr(model, "quantization_method", None) is not None and model_args.adapter_name_or_path is not None:
        raise ValueError("Cannot merge adapters to a quantized model.")

    if not isinstance(model, PreTrainedModel):
        raise ValueError("The model is not a `PreTrainedModel`, export aborted.")

    if getattr(model, "quantization_method", None) is not None:  # quantized model adopts float16 type
        setattr(model.config, "torch_dtype", torch.float16)
    else:
        if model_args.infer_dtype == "auto":
            output_dtype = getattr(model.config, "torch_dtype", torch.float16)
        else:
            output_dtype = getattr(torch, model_args.infer_dtype)

        setattr(model.config, "torch_dtype", output_dtype)
        model = model.to(output_dtype)
        logger.info("Convert model dtype to: {}.".format(output_dtype))

    model.save_pretrained(
        save_directory=model_args.export_dir,
        max_shard_size="{}GB".format(model_args.export_size),
        safe_serialization=(not model_args.export_legacy_format),
    )
    if model_args.export_hub_model_id is not None:
        model.push_to_hub(
            model_args.export_hub_model_id,
            token=model_args.hf_hub_token,
            max_shard_size="{}GB".format(model_args.export_size),
            safe_serialization=(not model_args.export_legacy_format),
        )

    if finetuning_args.stage == "rm":
        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        if os.path.exists(os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_SAFE_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_SAFE_WEIGHTS_NAME),
            )
            logger.info("Copied valuehead to {}.".format(model_args.export_dir))
        elif os.path.exists(os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME)):
            shutil.copy(
                os.path.join(vhead_path, V_HEAD_WEIGHTS_NAME),
                os.path.join(model_args.export_dir, V_HEAD_WEIGHTS_NAME),
            )
            logger.info("Copied valuehead to {}.".format(model_args.export_dir))

    try:
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"
        tokenizer.save_pretrained(model_args.export_dir)
        if model_args.export_hub_model_id is not None:
            tokenizer.push_to_hub(model_args.export_hub_model_id, token=model_args.hf_hub_token)

        if processor is not None:
            getattr(processor, "image_processor").save_pretrained(model_args.export_dir)
            if model_args.export_hub_model_id is not None:
                getattr(processor, "image_processor").push_to_hub(
                    model_args.export_hub_model_id, token=model_args.hf_hub_token
                )

    except Exception as e:
        logger.warning("Cannot save tokenizer, please copy the files manually: {}.".format(e))

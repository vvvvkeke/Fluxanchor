import torch
from torch import nn


class LLMExtractor(nn.Module):
    def __init__(self, input_dim=3584, model_name="llm", num_experts_shared=3, num_experts_task=2, num_tasks=2):
        super(LLMExtractor, self).__init__()

        self.llm_extractor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.num_experts_shared = num_experts_shared  # 共享专家数量
        self.num_experts_task = num_experts_task  # 每个任务的独有专家数量
        self.num_tasks = num_tasks  # 任务数量
        self.model_name = model_name

        # 共享专家模块
        self.shared_experts = nn.ModuleList([self.llm_extractor for _ in range(num_experts_shared)])

        # 每个任务的独有专家模块
        self.task_experts = nn.ModuleList([
            nn.ModuleList([self.llm_extractor for _ in range(num_experts_task)]) for _ in range(num_tasks)
        ])

        # 每个任务的门控网络（分别对共享和独有专家进行权重分配）
        self.shared_gating_networks = nn.ModuleList(
            [nn.Linear(input_dim, num_experts_shared) for _ in range(num_tasks)])
        self.task_gating_networks = nn.ModuleList(
            [nn.Linear(input_dim, num_experts_task) for _ in range(num_tasks)])
        self.shared_gate_networks = nn.ModuleList([nn.Linear(128, 1) for _ in range(num_tasks)])

        # 每个任务不同输出头(2)
        self.task_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(self.num_tasks)])
        self.model_name = model_name

    def forward(self, x):
        # 共享专家输出
        shared_expert_outputs_substrate = torch.stack([
            expert(x)
            for expert in self.shared_experts
        ], dim=1)  # (batch, num_experts_shared, 64)

        # 每个任务的独有专家输出
        task_expert_outputs_substrate = [
            torch.stack([
                expert(x)
                for expert in self.task_experts[task_idx]
            ], dim=1)  # (batch_size, num_experts_task, 64)
            for task_idx in range(self.num_tasks)
        ]  # (num_experts_task,batch_size, num_experts_task, 64)

        task_outputs = []

        for task_idx in range(self.num_tasks):
            # 门控网络对共享专家输出的加权
            shared_gates_substrate = torch.softmax(self.shared_gating_networks[task_idx](x),
                                                   dim=-1)  # (batch_size,  num_experts_shared)

            # 门控网络对独有专家输出的加权
            task_gates_substrate = torch.softmax(self.task_gating_networks[task_idx](x),
                                                 dim=-1)  # (batch_size,  num_experts_task)

            # 加权组合共享专家输出
            shared_task_output_substrate = torch.einsum('be,bed->bd', shared_gates_substrate,
                                                        shared_expert_outputs_substrate)  # (batch_size,64)

            # 加权组合独有专家输出
            task_specific_output_substrate = torch.einsum('be,bed->bd', task_gates_substrate,
                                                          task_expert_outputs_substrate[task_idx])  # (batch_size,64)

            # 门控网络生成权重
            gate_weight_shared = torch.sigmoid(self.shared_gate_networks[task_idx](task_specific_output_substrate))
            gate_weight_task = 1 - gate_weight_shared  # 确保两者和为1

            # 加权组合
            combined_output_substrate = gate_weight_shared * shared_task_output_substrate + gate_weight_task * task_specific_output_substrate  # (batch_size,64)

            task_output = self.task_heads[task_idx](combined_output_substrate)
            task_outputs.append(task_output)
        return task_outputs[0].squeeze(1), task_outputs[1].squeeze(1)

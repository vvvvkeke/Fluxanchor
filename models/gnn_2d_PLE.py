from torch import nn
from torch_geometric.nn import GCNConv, DenseGCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import MLP
from torch_geometric.nn import GraphNorm
from torch.nn import ReLU
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import BatchNorm1d
from torch.nn import Sequential
from torch_geometric.nn import global_mean_pool


class SmilesExtractor(torch.nn.Module):
    def __init__(self, input_features, model_name, num_experts_shared=2, num_experts_task=2,num_layers=3,num_tasks=2):
        super(SmilesExtractor, self).__init__()
        # self.substrate_extractor = GCN_model(input_features)
        # self.product_extractor = GCN_model(input_features)
        self.cross_attention = SmilesCrossAttention(model_name=model_name)

        self.num_experts_shared = num_experts_shared  # 共享专家数量
        self.num_experts_task = num_experts_task  # 每个任务的独有专家数量
        self.num_tasks = num_tasks  # 任务数量
        self.num_layers = num_layers #CGC层
        self.model_name = model_name

        # 共享专家模块（多层）
        self.layers_shared_experts = nn.ModuleList([
            nn.ModuleList([GCN_model(input_features) for _ in range(num_experts_shared)])
            for _ in range(num_layers)
        ])
        
        # 每个任务的独有专家模块（多层）
        self.layers_task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([GCN_model(input_features) for _ in range(num_experts_task)])
                for _ in range(num_tasks)
            ])
            for _ in range(num_layers)
        ])

        # 每个任务的门控网络（分别对共享和独有专家进行权重分配）（多层）
        self.layers_shared_gates = nn.ModuleList([
            nn.ModuleList([nn.Linear(input_features, num_experts_shared) for _ in range(num_tasks)])
            for _ in range(num_layers)
        ])

        self.layers_task_gates = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_features, num_experts_task) for _ in range(num_tasks)
            ])
            for _ in range(num_layers)
        ])

        # 每个任务不同输出头(2)
        self.task_heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(self.num_tasks)])
        

    def forward(self, substrate, product, substrate_embedding, product_embedding):

        # 初始化共享信息和任务特定信息
        shared_output_substrate = substrate.X #(N1，512)
        shared_output_product = product.X #(N2，512)
        task_outputs_substrate = [substrate.X for _ in range(self.num_tasks)] #(num_tasks,N1，512)
        task_outputs_product = [product.X for _ in range(self.num_tasks)]#(num_tasks,N2，512)

        # 逐层提取
        for layer_idx in range(self.num_layers):

            # 当前层的共享专家输出 每一层都会更新这个共享的
            shared_expert_outputs_substrate = torch.stack([
                expert(shared_output_substrate, substrate.edge_index, substrate.edge_attr, substrate.batch)
                for expert in self.layers_shared_experts[layer_idx]
            ], dim=1)  # (Batch, num_experts_shared, 512)

            shared_expert_outputs_product = torch.stack([
                expert(shared_output_product, product.edge_index, product.edge_attr, product.batch)
                for expert in self.layers_shared_experts[layer_idx]
            ], dim=1)  # (Batch, num_experts_shared, 512)

            # 当前层的任务特定专家输出 每一层都会更新这个特定任务的专家的信息
            task_expert_outputs_substrate = [
                torch.stack([
                    expert(task_outputs_substrate[task_idx], substrate.edge_index, substrate.edge_attr, substrate.batch) for expert in self.layers_task_experts[layer_idx][task_idx]
                ], dim=1)   # (batch_size, num_experts_task, 64)
                for task_idx in range(self.num_tasks)
            ]# (num_experts_task,batch_size, num_experts_task, 64)

            task_expert_outputs_product = [
                torch.stack([
                    expert(task_outputs_substrate[task_idx], substrate.edge_index, substrate.edge_attr, substrate.batch) for expert in self.layers_task_experts[layer_idx][task_idx]
                ], dim=1)  # (batch_size, num_experts_task, 64)
                for task_idx in range(self.num_tasks)
            ]# (num_experts_task,batch_size, num_experts_task, 64)

             # 任务输出
            new_task_outputs_substrate = []
            new_task_outputs_product = []

            for task_idx in range(self.num_tasks):

                if layer_idx == 0:
                    # 对第一层的节点级别特征进行全局池化，转化为 (batch_size, feature_dim)
                    global_embedding_substrate = global_mean_pool(task_outputs_substrate[task_idx], substrate.batch)
                    global_embedding_product = global_mean_pool(task_outputs_product[task_idx], product.batch)
                else:
                    # 后续层直接使用任务输出 (batch_size, feature_dim)
                    global_embedding_substrate = task_outputs_substrate[task_idx]
                    global_embedding_product = task_outputs_product[task_idx]

                # 门控网络对共享专家输出的加权
                shared_gates_substrate = torch.softmax(self.layers_shared_gates[layer_idx][task_idx](task_outputs_substrate[task_idx]), dim=-1)  # 第一层是（N1，num_experts_shared），然后是(batch_size, num_experts_shared)
                shared_gates_product = torch.softmax(self.layers_shared_gates[layer_idx][task_idx](task_outputs_product[task_idx]), dim=-1)  # 第一层是（N2，num_experts_shared），然后是(batch_size, num_experts_shared)

                # 门控网络对独有专家输出的加权
                task_gates_substrate = torch.softmax(self.layers_task_gates[layer_idx][task_idx](task_outputs_substrate[task_idx]), dim=-1)  # 第一层是（N1，num_experts_shared），然后是(batch_size, num_experts_shared)
                task_gates_product = torch.softmax(self.layers_task_gates[layer_idx][task_idx](task_outputs_product[task_idx]), dim=-1)  # 第一层是（N2，num_experts_shared），然后是(batch_size, num_experts_shared)

                # 共享和任务特定专家输出的加权组合（底物和产物分别处理）
                shared_task_output_substrate = torch.einsum('be,bed->bd', shared_gates_substrate, shared_expert_outputs_substrate)  # (batch_size, 64)
                shared_task_output_product = torch.einsum('be,bed->bd', shared_gates_product, shared_expert_outputs_product)  # (batch_size, 64)

                task_specific_output_substrate = torch.einsum('be,bed->bd', task_gates_substrate, task_expert_outputs_substrate[task_idx])  # (batch_size, 64)
                task_specific_output_product = torch.einsum('be,bed->bd', task_gates_product, task_expert_outputs_product[task_idx])  # (batch_size, 64)

                # 更新任务输出
                combined_output_substrate = shared_task_output_substrate + task_specific_output_substrate
                combined_output_product = shared_task_output_product + task_specific_output_product

                new_task_outputs_substrate.append(combined_output_substrate)
                new_task_outputs_product.append(combined_output_product)
            
            task_outputs_substrate = new_task_outputs_substrate
            task_outputs_product = new_task_outputs_product

            # 更新共享信息（基于共享专家直接生成）
            shared_output_substrate = torch.mean(shared_expert_outputs_substrate, dim=1)
            shared_output_product = torch.mean(shared_expert_outputs_product, dim=1)

        # 最终任务输出
        task_outputs = []
        for task_idx in range(self.num_tasks):
            # 将共享输出与任务特定输出结合
            combined_output_substrate = torch.cat([shared_output_substrate, task_outputs_substrate[task_idx]], dim=-1)
            combined_output_product = torch.cat([shared_output_product, task_outputs_product[task_idx]], dim=-1)
            # 是否使用交叉注意力
            CrossOver = 0
            if CrossOver == 0:
                combined_output = torch.cat([combined_output_substrate, combined_output_product], dim=-1)  # (batch_size, 256)
            else:
                shared_attention = self.cross_attention(combined_output_substrate, combined_output_product)  # (batch_size, 128)
                combined_output = shared_attention
            # 通过任务的输出头生成结果
            task_output = self.task_heads[task_idx](combined_output)  # (batch_size, 1)
            task_outputs.append(task_output)

        if self.model_name == "2dgnn":
            return task_outputs[0].squeeze(1), task_outputs[1].squeeze(1)
        else:
            return task_outputs

        # substrate_output = self.substrate_extractor(substrate.X, substrate.edge_index, substrate.edge_attr,
        #                                             substrate.batch,
        #                                             # substrate.morgan_fp, substrate.rdk_fp
        #                                             )
        # product_output = self.product_extractor(product.X, product.edge_index, product.edge_attr, product.batch,
        #                                         # product.morgan_fp,product.rdk_fp
        #                                         )

        # if self.model_name == "2dgnn":
        #     kcat, km = self.cross_attention(substrate_output, product_output)
        #     return kcat, km
        # else:
        #     output = self.cross_attention(substrate_output, product_output)
        #     return output


class GCN_model(torch.nn.Module):
    def poly_regression(self):
        # Write the equation here
        pass

    # Too large output_features: Make it 2 or 4.
    def __init__(self, input_features):
        super(GCN_model, self).__init__()

        self.edge_lin = Linear(3, 1)

        self.conv1 = GCNConv(in_channels=input_features, out_channels=256)
        self.conv2 = GCNConv(256, 128)
        self.bn1 = GraphNorm(128)
        # self.bn1 = BatchNorm1d(128)
        self.conv3 = GCNConv(128, 64)

    def forward(self, data_x, data_edge_index, data_edge_attr, batch, morgan_fp=None, rdkit_fp=None):
        x = self.conv1(data_x, data_edge_index)
        x = x.relu()
        x = self.conv2(x, data_edge_index)
        x = self.bn1(x,batch=batch)
        x = x.relu()
        x = self.conv3(x, data_edge_index)

        x = F.dropout(x, p=0.5, training=self.training)
        x = global_mean_pool(x, batch)

        return x


class SmilesCrossAttention(nn.Module):
    def __init__(self, in_dim1=64, in_dim2=64, k_dim=256, v_dim=256, num_heads=4, model_name="fusion"):
        super(SmilesCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)

        self.output = nn.Linear(v_dim * num_heads, 128)
        # self.kcat_o = nn.Linear(v_dim * num_heads, 1)
        # self.km_o = nn.Linear(v_dim * num_heads, 1)
        self.model_name = model_name

    def forward(self, x1, x2, mask=None):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size()[1]

        # q1(batch_size, num_heads, seq_len1, k_dim)
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        # k2(batch_size, num_heads, k_dim, seq_len2)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        # v2(batch_size, num_heads, seq_len2, v_dim)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        # attention(batch_size, num_heads, seq_len1, seq_len2)
        attention = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = F.softmax(attention, dim=1)
        # output(batch_size, num_heads, seq_len1, v_dim)=>(batch_size, seq_len1, num_heads*v_dim)
        hin = torch.matmul(attention, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1).squeeze(1)

        return self.output(hin)
        # output(batch_size, seq_len1, in_dim1)
        # if self.model_name == "2dgnn" or self.model_name == "3dgnn":
        #     kcat_output = self.kcat_o(hin).squeeze(1)
        #     km_output = self.km_o(hin).squeeze(1)
        #     return kcat_output, km_output
        # else:
        #     return self.output(hin)
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
    def __init__(self, input_features, model_name, num_experts=3,num_tasks=2):
        super(SmilesExtractor, self).__init__()
        # self.substrate_extractor = GCN_model(input_features)
        # self.product_extractor = GCN_model(input_features)
        self.cross_attention = SmilesCrossAttention(model_name=model_name)

        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.model_name = model_name

        # 共享专家模块（3）
        self.experts = nn.ModuleList([GCN_model(input_features) for _ in range(num_experts)])

        # 每个任务有独立的门控网络(2)
        self.gating_networks = nn.ModuleList([nn.Linear(input_features, num_experts) for _ in range(num_tasks)])

        # 每个任务不同输出头(2)
        self.task_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(self.num_tasks)])
        

    def forward(self, substrate, product, substrate_embedding, product_embedding):

        substrate_expert_outputs = torch.stack([
            expert(substrate.X, substrate.edge_index, substrate.edge_attr, substrate.batch)
            for expert in self.experts
        ], dim=1)  # (batch(因为平均池化), num_experts, 64)

        product_expert_outputs = torch.stack([
            expert(product.X, product.edge_index, product.edge_attr, product.batch)
            for expert in self.experts
        ], dim=1)  # (batch(因为平均池化), num_experts, 64)

        task_outputs = []

        for task_idx in range(self.num_tasks):
            # 计算专家权重
            substrate_gates = torch.softmax(self.gating_networks[task_idx](substrate_embedding), dim=-1)  # (batch, num_experts)
            product_gates = torch.softmax(self.gating_networks[task_idx](product_embedding), dim=-1)  # (batch, num_experts)

            # 根据权重组合专家输出 b-->batch_size  e-->num_experts d--->dim
            substrate_task_output = torch.einsum('be,bed->bd', substrate_gates, substrate_expert_outputs)  # (batch_size, 64)
            product_task_output = torch.einsum('be,bed->bd', product_gates, product_expert_outputs)  # (batch_size, 64)

            # 是否使用交叉注意力
            CrossOver = 1
            if CrossOver == 0:
                combined_output = torch.cat([substrate_task_output, product_task_output], dim=-1)  # (batch_size, 128)
                task_output = self.task_heads[task_idx](combined_output) # (batch_size, 1)
            else:     
                combined_output = self.cross_attention(substrate_task_output, product_task_output)  # (batch_size, 128)
                task_output = self.task_heads[task_idx](combined_output) # (batch_size, 1)

            task_outputs.append(task_output)

        if self.model_name == "2dgnn":
            return task_outputs[0].squeeze(1),task_outputs[1].squeeze(1)
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
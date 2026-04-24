from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphNorm
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool

from models.gnn_3d import EGNN


class rubisco(torch.nn.Module):
    def __init__(self, input_features, model_name):
        super(rubisco, self).__init__()
        self.substrate_extractor = GCN_model(input_features)
        self.product_extractor = GCN_model(input_features)
        self.cross_attention = SmilesCrossAttention(model_name=model_name)
        self.model_name = model_name

    def forward(self, substrate, product, substrate_embedding, product_embedding, condition):
        substrate_output = self.substrate_extractor(substrate.X, substrate.edge_index, substrate.edge_attr,
                                                    substrate.batch,
                                                    # substrate.morgan_fp, substrate.rdk_fp
                                                    )
        product_output = self.product_extractor(product.X, product.edge_index, product.edge_attr, product.batch,
                                                # product.morgan_fp,product.rdk_fp
                                                )
        kcat = self.cross_attention(substrate_output, product_output)
        return kcat


def edge_index_to_adjacency_matrix(edge_index, num_nodes):
    # 构建一个大小为 (num_nodes, num_nodes) 的零矩阵
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float)

    # 使用索引广播机制，一次性将边索引映射到邻接矩阵的相应位置上
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    adjacency_matrix[edge_index[1], edge_index[0]] = 1

    return adjacency_matrix


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
        self.conv3 = GCNConv(128, 512)

        self.bn1 = BatchNorm1d(128)

        self.rates = Linear(64, 1)

    def forward(self, data_x, data_edge_index, data_edge_attr, batch, morgan_fp=None, rdkit_fp=None):
        # e = self.edge_lin(data_edge_attr)
        x = self.conv1(data_x, data_edge_index)

        x = x.relu()
        x = self.conv2(x, data_edge_index)
        # x = self.bn1(x)
        x = x.relu()
        x = self.conv3(x, data_edge_index)

        x = F.dropout(x, p=0.5, training=self.training)
        x = global_mean_pool(x, batch)

        return x


class SmilesCrossAttention(nn.Module):
    def __init__(self, in_dim1=512, in_dim2=512, k_dim=128, v_dim=128, num_heads=4, model_name="fusion"):
        super(SmilesCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)

        self.output = nn.Linear(v_dim * num_heads, 128)
        ### 加的
        self.kcat_o = nn.Linear(512, 1)
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
        hin = torch.matmul(attention, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1).squeeze(1)
        # hin = self.output(hin)
        kcat_output = self.kcat_o(hin).squeeze(1)
        return kcat_output

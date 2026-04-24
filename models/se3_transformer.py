
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import SumAggregation, MessagePassing, Sequential
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms

from equitorch.nn import (
    SO2Linear,
    GaussianBasisExpansion,
    PolynomialCutoff,
    S2Act,
    MultiheadAttentionBlock,
    SE3TrAttention,
    DegreeWiseLinear,
    EquivariantLayerNorm,
    Separable

)
from equitorch.utils.geometries import rot_on

from equitorch.utils import num_degree_triplets, range_eq
from equitorch.typing import DegreeRange
from equitorch.transforms import RadiusGraph, AddEdgeAlignWignerD

from e3nn import o3

# Code borrowed and modified from https://github.com/e3nn/e3nn/blob/main/Example/tetris.py

class SE3TransformerBlock(MessagePassing):

    def __init__(self,
                in_channels: int,
                out_channels: int,
                L_in: DegreeRange,
                L_out: DegreeRange,
                num_heads: int = 1,
                k_channels: int = None,
                L_k: DegreeRange = None,
                ):
        super().__init__(node_dim=0)

        if k_channels is None:
            k_channels = in_channels
        if L_k is None:
            L_k = L_in

        self.num_weights_k = num_degree_triplets(L_in, L_k)
        self.num_weights_v = num_degree_triplets(L_in, L_out)
        self.att_msg = MultiheadAttentionBlock(
            num_heads=num_heads,
            attention_score_producer=SE3TrAttention(
                L_in, L_in, in_channels, k_channels, num_heads,
                nn.LazyLinear(self.num_weights_k*in_channels*num_heads*k_channels)
            ),
            v_producer=Sequential('x, edge_emb',[
                (nn.LazyLinear(self.num_weights_v*in_channels*out_channels), 'edge_emb -> weight'),
                (SO2Linear(L_in, L_out, in_channels, out_channels, True), 'x, weight -> x')
            ]) # The Sequential module provided by torch_geometric
        )
        self.self_interaction = DegreeWiseLinear(L_in, L_out, in_channels, out_channels)

        self.act = Separable(
            nn.Sequential(
                nn.LayerNorm(out_channels),
                nn.SiLU()
            ),
            nn.Sequential(
                EquivariantLayerNorm(range_eq(L_out), out_channels),
                S2Act(range_eq(L_out), nn.SiLU(), 8)
            )
        )

    def forward(self, x, edge_index,
                D_in, DT_out, edge_emb, edge_weight = None):
        out = self.propagate(edge_index, x=x,
                            edge_emb=edge_emb,
                            edge_weight=edge_weight,
                            D_in=D_in, DT_out=DT_out)
        out = out + self.self_interaction(x)
        return self.act(out)

    def message(self, x_j, x_i, edge_index,
                edge_emb, edge_weight,
                D_in, DT_out):
        x_i = rot_on(D_in, x_i)
        x_j = rot_on(D_in, x_j)
        out, _ = self.att_msg((x_i, x_j), x_j, edge_index[1],
                            edge_emb=edge_emb)
        out = rot_on(DT_out, out)
        if edge_weight is not None:
            return edge_weight.view(-1,1,1) * out
        else:
            return out

class SE3Transformer(nn.Module):

    def __init__(self, hidden=4, L=3, num_heads=1):

        super().__init__()
        self.hidden = hidden

        self.edge_embedding = GaussianBasisExpansion(0.1, 20, 0.7, 1.7)
        self.cutoff = PolynomialCutoff(1.5)

        self.layer1 = SE3TransformerBlock(in_channels=1, out_channels=hidden, k_channels=hidden//2,
                                        num_heads=1, L_in=0, L_out=L)
        self.layer2 = SE3TransformerBlock(in_channels=hidden, out_channels=hidden, k_channels=hidden//2,
                                        num_heads=num_heads, L_in=L, L_out=L)
        self.layer3 = SE3TransformerBlock(in_channels=hidden, out_channels=hidden, k_channels=hidden//2,
                                        num_heads=num_heads, L_in=L, L_out=L)
        self.layer4 = SE3TransformerBlock(in_channels=hidden, out_channels=hidden, k_channels=hidden//2,
                                        num_heads=num_heads, L_in=L, L_out=0)

        self.pool = SumAggregation()
        self.output = nn.Sequential(nn.Linear(hidden, 8), nn.Softmax(dim=-1))

    def forward(self, h, edges, x, edge_attr, batch, atoms=None):
        edge_len = edge_attr.norm(dim=-1)
        DT = x.transpose(-1,-2)
        D0 = DT0 = x[:,:1,:1]
        x = h.unsqueeze(-2)
        edge_emb = self.edge_embedding(edge_len)
        edge_weight = self.cutoff(edge_len)
        h = self.layer1(x, edges, D0, DT, edge_emb, edge_weight)
        h = self.layer2(h, edges, x, DT, edge_emb, edge_weight)
        h = self.layer3(h, edges, x, DT, edge_emb, edge_weight)
        h = self.layer4(h, edges, x, DT0, edge_emb, edge_weight)
        h = self.pool(h[:,0,:], batch, dim=0)
        h = self.output(h)
        return h




def tetris() -> None:
    pos = [
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)],  # chiral_shape_2
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # L
        [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # T
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],  # zigzag
    ]
    pos = torch.tensor(pos, dtype=torch.get_default_dtype())
    labels = torch.arange(8, dtype=torch.long)

    # apply random rotation
    pos = torch.einsum("zij,zaj->zai", o3.rand_matrix(len(pos)), pos)

    return pos, labels
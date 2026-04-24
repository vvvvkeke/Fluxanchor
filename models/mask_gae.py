from torch_geometric.resolver import resolver
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, HeteroConv, Sequential, global_mean_pool
import torch

from models.gnn_2d import SmilesCrossAttention


try:
    import torch_cluster  # noqa

    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

from copy import copy
from typing import Any, Optional, Union

from torch import nn, Tensor
from torch_geometric.nn import (GATConv, GATv2Conv, GCNConv, GINConv, Linear,
                                SAGEConv)
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import degree, sort_edge_index, to_undirected

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks):
        super(AutomaticWeightedLoss, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_tasks))

    def forward(self, *losses):
        loss = losses[0] + losses[1]
        # loss = 0
        # loss_num = losses.__len__()
        # for i in range(losses.__len__()):
        #     loss += losses[i]
        #     loss = loss + torch.exp(-self.weights[i]) * losses[i] / loss_num + self.weights[i]
        # return loss
        return loss

class FusedBCE(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder  # 用于计算正负样本的预测值。

    # left: 通常表示源节点的特征。
    # right: 通常表示目标节点的特征。
    # pairs: 正样本对，表示图中存在的边。
    # neg_pairs: 可选的负样本对，表示图中不存在的边。
    def forward(self, left, right, pairs, neg_pairs=None):
        pos_out = self.decoder(left, right, pairs)  # 使用解码器对正样本对进行预测，返回的输出表示每个正样本对的相似性分数或概率。
        labels = torch.ones_like(pos_out)  # 创建与 pos_out 相同形状的张量，所有值为 1，表示正样本的标签。
        loss = F.binary_cross_entropy(pos_out, labels)  # 使用 F.binary_cross_entropy 计算正样本的二元交叉熵损失，比较 pos_out 和 labels。

        if neg_pairs is not None:  # 如果提供了 neg_pairs，则进行负样本的计算：
            neg_out = self.decoder(left, right, neg_pairs)  # 使用解码器对负样本对进行预测，返回的输出表示每个负样本对的相似性分数或概率。
            neg_labels = torch.zeros_like(neg_out)  # 创建与 neg_out 相同形状的张量，所有值为 0，表示负样本的标签。
            loss += F.binary_cross_entropy(neg_out, neg_labels)  # 将负样本的损失加到总损失中。
        return loss

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def normalization_resolver(query: Optional[Union[Any, str]], *args, **kwargs):
    if query is None or query == 'none':
        return torch.nn.Identity()
    import torch_geometric.nn.norm as norm
    base_cls = torch.nn.Module
    base_cls_repr = 'Norm'
    norms = [
        norm for norm in vars(norm).values()
        if isinstance(norm, type) and issubclass(norm, base_cls)
    ]
    norm_dict = {}
    return resolver(norms, norm_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)


def layer_resolver(name, first_channels, second_channels, heads=1):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(nn.Sequential(Linear(first_channels, second_channels),
                                      nn.LayerNorm(second_channels),
                                      nn.PReLU(),
                                      Linear(second_channels, second_channels),
                                      # nn.LayerNorm(second_channels),
                                      ), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, second_channels, heads=heads)
    elif name == 'linear':
        layer = Linear(first_channels, second_channels)
    else:
        raise ValueError(name)
    return layer


def activation_resolver(query: Optional[Union[Any, str]] = 'relu', *args, **kwargs):
    if query is None or query == 'none':
        return torch.nn.Identity()
    base_cls = torch.nn.Module
    base_cls_repr = 'Act'
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(acts, act_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)


def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

NUM_CANDIDATES = 20


# 负采样
def negative_sampling(method,
                      x, edge_index,
                      num_neg_samples,
                      left,
                      right,
                      decoder,
                      num_nodes=None):
    if not isinstance(x, tuple):  # 如果 x 不是元组，则将其转换为 (x, x)，表示源节点和目标节点特征相同。
        x = (x, x)
    num_nodes = num_nodes or (x[0].size(0), x[1].size(0))
    device = x[0].device
    # 调用相应的负采样函数生成负边 neg_edges。
    if method == 'similarity':
        neg_edges = similarity_negative_sampler(
            x=x,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            device=device,
        )

    elif method == 'random':
        neg_edges = random_negative_sampler(
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            device=device,
        )
    elif method == 'degree':
        neg_edges = degree_negative_sampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            device=device,
        )
    elif method == 'hard_negative':
        neg_edges = hard_negative_sampler(
            x=(left, right),
            decoder=decoder,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples,
            device=device,
        )
    else:
        raise ValueError(f'Unknown negative sampler {method}')
    return neg_edges


def random_negative_sampler(num_nodes, num_neg_samples, device):
    src = torch.randint(0, num_nodes[0], size=(num_neg_samples,), device=device)  # 随机生成的源节点索引。
    dst = torch.randint(0, num_nodes[1], size=(num_neg_samples,), device=device)  # 随机生成的目标节点索引。
    neg_edges = torch.stack([src, dst], dim=0)  # 使用 torch.stack 将源和目标节点组合成一个边张量 neg_edges，其形状为 [2, num_neg_samples]。
    return neg_edges  # 返回生成的负边。


def degree_negative_sampler(edge_index, num_nodes, num_neg_samples, device):
    candidates = random_negative_sampler(num_nodes, num_neg_samples=num_neg_samples * NUM_CANDIDATES, device=device)
    d = degree(edge_index[1], num_nodes)
    row, col = candidates
    score = (d[row] - d[col]).abs()
    k = score.topk(num_neg_samples, largest=False).indices
    neg_edges = candidates[:, k]
    return neg_edges


def similarity_negative_sampler(x, num_nodes, num_neg_samples, device):
    left, right = x
    candidates = random_negative_sampler(num_nodes, num_neg_samples=num_neg_samples * NUM_CANDIDATES, device=device)
    row, col = candidates
    score = F.cosine_similarity(left[row], right[col])
    k = score.topk(num_neg_samples, largest=False).indices
    neg_edges = candidates[:, k]
    return neg_edges


def hard_negative_sampler(x, decoder, num_nodes, num_neg_samples, device):
    left, right = x
    candidates = random_negative_sampler(num_nodes, num_neg_samples=num_neg_samples * NUM_CANDIDATES, device=device)
    row, col = candidates
    with torch.no_grad():
        score = decoder(left, right, candidates).squeeze()
    k = score.topk(num_neg_samples, largest=False).indices
    neg_edges = candidates[:, k]
    return neg_edges


class DotProductEdgeDecoder(nn.Module):
    """Dot-Product Edge Decoder"""

    def __init__(self, left=2, right=2, *args, **kwargs):
        super().__init__()
        self.left = left
        self.right = right

    def reset_parameters(self):
        return

    def forward(self, left, right, pairs, sigmoid=True):
        x = left[pairs[0]] * right[pairs[1]]
        x = x.sum(-1)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class EdgeDecoder(nn.Module):
    """MLP Edge Decoder"""

    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels=1,
            num_layers=2,
            dropout=0.5,
            activation="relu",
            norm="none",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        network = []
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if is_last_layer else hidden_channels
            layer = layer_resolver("linear", first_channels, second_channels)

            if not is_last_layer and dropout > 0:
                network.append((nn.Dropout(dropout), "x -> x"))
            network.append((layer, "x -> x"))
            if not is_last_layer and norm != "none":
                network.append(
                    (normalization_resolver(norm, second_channels), "x -> x")
                )
            if not is_last_layer and activation != "none":
                # whether to add last activation
                network.append((activation_resolver(activation), "x -> x"))

        self.network = Sequential("x", network)

    def reset_parameters(self):
        for layer in self.network:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, left, right, pairs, sigmoid=True):
        x = left[pairs[0]] * right[pairs[1]]
        x = self.network(x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class FeatureDecoder(nn.Module):
    """MLP Feature Decoder"""

    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels=1,
            num_layers=2,
            dropout=0.5,
            activation="relu",
            norm="none",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        network = []
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if is_last_layer else hidden_channels
            layer = layer_resolver("linear", first_channels, second_channels)

            if not is_last_layer and dropout > 0:
                network.append((nn.Dropout(dropout), "x -> x"))
            network.append((layer, "x -> x"))

            if not is_last_layer and norm != "none":
                network.append(
                    (normalization_resolver(norm, second_channels), "x -> x")
                )
            if not is_last_layer and activation != "none":
                # whether to add last activation
                network.append((activation_resolver(activation), "x -> x"))

        self.network = Sequential("x", network)

    def reset_parameters(self):
        for layer in self.network:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        x = self.network(x)
        return x


def mask_edge(edge_index: Tensor, p: float = 0.7):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]


class MaskEdge(nn.Module):
    def __init__(self, p: float = 0.7, undirected: bool = True):
        super().__init__()
        self.p = p  # p: 代表掩蔽边的概率，默认值为 0.7，表示 70% 的边将被掩蔽。
        self.undirected = undirected  # undirected: 布尔值，指示图是否为无向图，默认值为 True。

    def forward(self, data):
        edge_index = data.edge_index  # 从 data 中提取 edge_index，然后调用 mask_edge 函数进行边的掩蔽，返回未掩蔽的边 remaining_edges 和掩蔽的边 masked_edges。
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        # 使用 copy 函数创建 data 的两个副本：
        # remaining_graph: 包含未掩蔽的边。
        # masked_graph: 包含掩蔽的边。
        # 将掩蔽的边信息存储在图的 masked_edges 属性中。
        remaining_graph = copy(data)
        masked_graph = copy(data)
        remaining_graph.masked_edges = masked_edges
        masked_graph.masked_edges = remaining_edges

        if self.undirected:  # 如果图是无向的，则调用 to_undirected 函数将边转换为无向边。
            remaining_edges = to_undirected(remaining_edges)
            masked_edges = to_undirected(masked_edges)

        masked_graph.edge_index = masked_edges  # 更新两个图的边索引，分别为掩蔽的和未掩蔽的边。
        remaining_graph.edge_index = remaining_edges
        return remaining_graph, masked_graph

    def extra_repr(self):
        return f"p={self.p}, undirected={self.undirected}"


class GNNEncoder(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels=None,
            num_heads=4,
            num_layers=2,
            dropout=0.5,
            norm='batchnorm',
            layer="gcn",
            activation="elu",
            add_last_act=True,
            add_last_bn=True,
    ):

        super().__init__()

        out_channels = out_channels or hidden_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.add_last_act = add_last_act
        self.add_last_bn = add_last_bn

        networks = []
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if is_last_layer else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else num_heads
            conv = layer_resolver(layer, first_channels,
                                  second_channels, heads)

            block = []
            if dropout > 0:
                block.append((nn.Dropout(dropout), 'x -> x'))
            block.append((conv, 'x, edge_index -> x'))
            if not is_last_layer or (is_last_layer and add_last_bn):
                # whether to add last BN
                if norm != 'none':
                    block.append((normalization_resolver(norm, second_channels * heads), 'x -> x'))
            if not is_last_layer or (is_last_layer and add_last_act):
                # whether to add last activation
                if activation != 'none':
                    block.append((activation_resolver(activation), 'x -> x'))
            networks.append(Sequential('x, edge_index', block))

        self.network = nn.Sequential(*networks)

    def reset_parameters(self):
        for block in self.network:
            for layer in block:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = to_sparse_tensor(edge_index, num_nodes=x.size(0))
        out = [x]
        for block in self.network:
            x = block(x, edge_index)
            out.append(x)
        return out


class MaskGAE(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            mask,
            degree_decoder=None,
            negative_sampler='random',
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.degree_decoder = degree_decoder
        self.mask = mask
        self.loss_fn = FusedBCE(decoder)

        assert negative_sampler in ['random', 'similarity', 'degree', 'hard_negative']
        self.negative_sampler = negative_sampler

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        if self.degree_decoder is not None:
            self.degree_decoder.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        return self.encoder(x, edge_index, **kwargs)

    def train_step(self, graph: Union[Data, HeteroData], alpha: float = 0.) -> torch.Tensor:
        return self.train_step_homo(graph, alpha=alpha)

    def train_step_homo(self, graph: Data, alpha: float = 0.) -> torch.Tensor:
        remaining_graph, masked_graph = self.mask(graph)  # 更新两个图的边索引，分别为掩蔽的和未掩蔽的边。
        x, remaining_edge_index = remaining_graph.X, remaining_graph.edge_index
        masked_edges = remaining_graph.masked_edges

        z = self.encoder(x, remaining_edge_index)
        left = right = z[-1]  # 返回最后一层encoder的特征
        neg_edges = negative_sampling(self.negative_sampler,  # 生成的负边
                                      x=graph.X,
                                      edge_index=graph.edge_index,
                                      num_neg_samples=masked_edges.size(1),
                                      left=left,
                                      right=right,
                                      decoder=self.decoder,
                                      )
        loss = self.loss_fn(left, right, masked_edges, neg_edges)
        if self.degree_decoder is not None and alpha > 0:
            deg = degree(masked_edges[1].flatten(), graph.num_nodes).float()
            deg = (deg - deg.mean()) / (deg.std() + 1e-6)
            loss += alpha * \
                    F.mse_loss(self.degree_decoder(left).squeeze(), deg)
        return loss

class SmilesExtractor2(torch.nn.Module):
    def __init__(self, input_features, model_name):
        super(SmilesExtractor2, self).__init__()
        encoder = GNNEncoder(in_channels=512,
                             hidden_channels=256,
                             out_channels=512,
                             num_layers=2,
                             dropout=0.3, #0.
                             norm="batchnorm",
                             layer="gcn",
                             activation="elu")
        decoder = EdgeDecoder(in_channels=512,
                              hidden_channels=32,
                              num_layers=2,
                              dropout=0.2,
                              norm="batchnorm")
        degree_decoder = FeatureDecoder(in_channels=512,
                                        hidden_channels=32,
                                        num_layers=2,
                                        dropout=0.2,
                                        norm="batchnorm")
        mask = MaskEdge(p=0.5)
        self.maskGAE = MaskGAE(encoder, decoder, mask,
                        degree_decoder=degree_decoder)
        # self.gcn = GCN_model(input_features)
        self.substrate_extractor = encoder
        self.product_extractor = encoder
        self.cross_attention = SmilesCrossAttention(model_name=model_name)

        self.condition_model = nn.Linear(2, 16)
        self.kcat_o = nn.Linear(128, 1)
        self.km_o = nn.Linear(128, 1)
        self.model_name = model_name
        self.auto_weighted_loss = AutomaticWeightedLoss(num_tasks=2)
        # self.lin1 = nn.Linear(512 * 2, 512)
        # self.lin2 = nn.Linear(512 * 2, 512)
        # self.SLG = SubstrateLGCrossAttention()
        # self.PLG = ProductLGCrossAttention()

    def pretrain(self, substrate, product):
        loss1 = self.maskGAE.train_step(substrate)
        loss2 = self.maskGAE.train_step(product)
        return loss1 + loss2

    def frozen_encoder(self):
        # 冻结 substrate_extractor 的参数
        for param in self.substrate_extractor.parameters():
            param.requires_grad = False
        # 冻结 product_extractor 的参数
        for param in self.product_extractor.parameters():
            param.requires_grad = False
    # def train_clf(self, substrate, product, substrate_embedding, product_embedding, condition):
    #     # 冻结 substrate_extractor 的参数
    #     for param in self.substrate_extractor.parameters():
    #         param.requires_grad = False
    #     # 冻结 product_extractor 的参数
    #     for param in self.product_extractor.parameters():
    #         param.requires_grad = False
    #     substrate_output = self.substrate_extractor(substrate)
    #     product_output = self.product_extractor(product)
    #     hin = self.cross_attention(substrate_output, product_output)
    #     kcat = self.kcat_o(hin).squeeze(1)
    #     km = self.km_o(hin).squeeze(1)
    #     return kcat, km

    def get_loss(self, pred_kcat, pred_km, true_kcat, true_km):
        loss_kcat = F.mse_loss(pred_kcat, true_kcat)
        loss_km = F.mse_loss(pred_km, true_km)
        # loss = self.auto_weighted_loss(loss_kcat, loss_km, self.r_squared(true_kcat, pred_kcat), self.r_squared(true_km, pred_km))
        loss = self.auto_weighted_loss(loss_kcat, loss_km)
        return loss
    def forward(self, substrate, product, substrate_embedding, product_embedding, condition, true_kcat, true_km):


        substrate_output = self.substrate_extractor(substrate.X, substrate.edge_index)
        product_output = self.product_extractor(product.X, product.edge_index)
        # condition_embedding = self.condition_model(condition)
        # substrate_output = self.lin1(torch.cat([substrate_output, substrate_embedding], dim=1))
        # product_output = self.lin2(torch.cat([product_output, product_embedding], dim=1))
        # substrate_output = self.SLG(substrate_output, substrate_embedding)
        # product_output = self.PLG(product_output, product_embedding)
        # if self.model_name == "2dgnn":
        substrate_output = global_mean_pool(substrate_output[-1], substrate.batch)
        product_output = global_mean_pool(product_output[-1], product.batch)
        hin = self.cross_attention(substrate_output, product_output)
        # hin = torch.cat([hin, condition_embedding], dim=1)
        kcat = self.kcat_o(hin).squeeze(1)
        km = self.km_o(hin).squeeze(1)
        loss = self.get_loss(kcat, km, true_kcat, true_km)
        return kcat, km, loss
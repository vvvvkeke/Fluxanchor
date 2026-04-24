import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import TransformerConv
import numpy as np


def get_geo_feat(X, edge_index):
    """
    get geometric node features and edge features
    """
    pos_embeddings = _positional_embeddings(edge_index)
    node_angles = _get_angle(X)
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

    geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

    return geo_node_feat, geo_edge_feat


def _get_direction_orientation(X, edge_index):  # N, CA, C, O, R
    """
    get the direction features
    """
    X_N = X[:, 0]  # [L, 3]
    X_Ca = X[:, 1]
    X_C = X[:, 2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v, dim=-1), dim=-1)  # ????
    local_frame = torch.stack([b, n, torch.cross(b, n, dim=-1)], dim=-1)  # [L, 3, 3] (3 column vectors)

    node_j, node_i = edge_index

    t = F.normalize(X[:, [0, 2, 3, 4]] - X_Ca.unsqueeze(1), dim=-1)  # [L, 4, 3]
    try:
        node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1)  # [L, 4 * 3]
    except Exception as ex:
        print(t.size())
        print(local_frame.size())
        print('except')
        print(ex)

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1)  # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1)  # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1)  # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1)  # [E, 5 * 3]
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim=-1)  # [E, 2 * 5 * 3]

    r = torch.matmul(local_frame[node_i].transpose(-1, -2), local_frame[node_j])  # [E, 3, 3]
    edge_orientation = _quaternions(r)  # [E, 4]

    return node_direction, edge_direction, edge_orientation


def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [N,3,3]
        Q [N,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
        Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i, j: R[:, i, j]
    signs = torch.sign(torch.stack([
        _R(2, 1) - _R(1, 2),
        _R(0, 2) - _R(2, 0),
        _R(1, 0) - _R(0, 1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q


def _positional_embeddings(edge_index, num_embeddings=16):
    """
    get the positional embeddings
    """
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE


def _get_angle(X, eps=1e-7):
    """
    get the angle features
    """
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]  ## torch.Size([1274, 3])
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2])  # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1)  # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles  # dim = 12


def _get_distance(X, edge_index):
    """
    get the distance features
    """
    atom_N = X[:, 0]  # [L, 3]
    atom_Ca = X[:, 1]
    atom_C = X[:, 2]
    atom_O = X[:, 3]
    atom_R = X[:, 4]
    # node_list 定义了一组原子对，表示要计算距离的节点对。
    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    node_dist = []  # node_dist 用于存储计算得到的节点距离特征。
    for pair in node_list:  # 对于 node_list 中的每一对原子
        atom1, atom2 = pair.split('-')  # 使用 split('-') 将原子对分割成两个原子
        E_vectors = vars()['atom_' + atom1] - vars()[
            'atom_' + atom2]  # 通过 vars() 函数动态获取对应原子的坐标，然后计算这两个原子之间的向量差 E_vectors
        rbf = _rbf(E_vectors.norm(dim=-1))  # 计算向量的范数，得到原子之间的距离
        node_dist.append(rbf)  # 使用 _rbf 函数计算径向基函数（RBF）特征，并将结果添加到 node_dist 列表中
    node_dist = torch.cat(node_dist,
                          dim=-1)  # 将所有的节点距离特征在最后一个维度上拼接，形成一个形状为 [N, 10 * 16] 的张量，其中 10 是节点对的数量，16 是 RBF 特征的维度

    atom_list = ["N", "Ca", "C", "O", "R"]  # atom_list 定义了所有原子的类型。
    edge_dist = []  ## 41 edge_dist 用于存储计算得到的边距离特征
    for atom1 in atom_list:  # 对于每一对原子，计算连接这两个原子的边的距离
        for atom2 in atom_list:
            # 使用 edge_index 获取对应的原子坐标
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]  # 计算边的向量差，并求出距离
            rbf = _rbf(E_vectors.norm(dim=-1))  # 使用 _rbf 计算 RBF 特征，并将结果添加到 edge_dist 列表中
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1)  # dim = [E, 25 * 16]
    # 将所有边距离特征在最后一个维度上拼接，形成一个形状为 [E, 25 * 16] 的张量，其中 E 是边的数量，25 是原子对的数量（5个原子之间的所有组合）
    return node_dist, edge_dist


def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class GNNLayer(nn.Module):
    """
    define GNN layer for subsequent computations
    """

    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads),
                                         heads=num_heads, dropout=dropout, edge_dim=num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E, batch_id)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E


class EdgeMLP(nn.Module):
    """
    define MLP operation for edge updates
    """

    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.BatchNorm1d(num_hidden)
        # self.norm = GraphNorm(num_hidden)

        self.W11 = nn.Linear(3 * num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E, batch_id):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = h_E + self.dropout(h_message)

        # 根据节点的 batch_id 推导边的 batch_id
        edge_batch_id = batch_id[edge_index[0]]
        h_E = self.norm(h_E)
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid()
        )

    def forward(self, h_V, batch_id):
        # c_V = scatter_add(h_V, batch_id, dim=0)
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


class Graph_encoder(nn.Module):
    """
    construct the graph encoder module
    """

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim,
                 seq_in=False, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20

        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        # self.norm_nodes = GraphNorm(hidden_dim)
        # self.norm_edges = GraphNorm(hidden_dim)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)

        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
            GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))

    def forward(self, h_V, edge_index, h_E, seq, batch_id):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)

        # 节点特征归一化（按图）  # h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_V = self.node_embedding(h_V)
        h_V = self.norm_nodes(h_V)
        h_V = self.W_v(h_V)

        # 根据节点的 batch_id 推导边的 batch_id
        edge_batch_id = batch_id[edge_index[0]]

        # 边特征归一化（按图）        # h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))
        h_E = self.edge_embedding(h_E)
        h_E = self.norm_edges(h_E)
        h_E = self.W_e(h_E)

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)

        return h_V


class Attention(nn.Module):
    """
    define the attention module
    """

    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  # input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  # x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  # x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  # attention.shape = (1, attention_hops, seq_len)
        return attention


class ActivateSiteScore(nn.Module):
    def __init__(self, hidden_dim=256):
        super(ActivateSiteScore, self).__init__()
        self.hidden_dim = hidden_dim
        # 定义线性层用于计算注意力分数
        self.attention_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim] => [64, 2511, 8]

        # 计算注意力权重
        attn_scores = self.attention_layer(x)  # [64, 2511, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # 归一化权重

        # 返回注意力分数
        return attn_weights  # 返回形状为 [64, 2511]

class GraphEC(nn.Module):
    """
    construct the GraphEC model
    """

    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, dropout, augment_eps, device,
                 model_name):
        super(GraphEC, self).__init__()
        self.augment_eps = augment_eps
        self.device = device
        self.hidden_dim = hidden_dim
        self.node_input_dim = node_input_dim
        self.model_name = model_name
        # define the encoder layer
        self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim,
                                           hidden_dim=hidden_dim, seq_in=False, num_layers=num_layers,
                                           drop_rate=dropout)
        # define the attention layer
        self.attention = Attention(hidden_dim, dense_dim=16, n_heads=4)

        self.input_block = nn.Sequential(
            nn.LayerNorm(1536 + 9, eps=1e-6)
            , nn.Linear(1536 + 9, hidden_dim)
            , nn.LeakyReLU()
        )
        attention_heads = 8
        # self.out_block = nn.Sequential(
        #     nn.Linear((attention_heads + 1) * hidden_dim, (attention_heads + 1) * hidden_dim)
        #     , nn.LeakyReLU()
        #     , nn.LayerNorm((attention_heads + 1) * hidden_dim, eps=1e-6)
        #     , nn.Dropout(dropout)
        #     , nn.Linear((attention_heads + 1) * hidden_dim, 128)
        # )
        self.activate_site = ActivateSiteScore()
        self.kcat_out_block = nn.Sequential(
            nn.Linear((attention_heads) * hidden_dim, (attention_heads + 1) * hidden_dim)
            , nn.LeakyReLU()
            , nn.LayerNorm((attention_heads + 1) * hidden_dim, eps=1e-6)
            , nn.Dropout(dropout)
            , nn.Linear((attention_heads + 1) * hidden_dim, 1)
        )
        self.km_out_block = nn.Sequential(
            nn.Linear((attention_heads) * hidden_dim, (attention_heads + 1) * hidden_dim)
            , nn.LeakyReLU()
            , nn.LayerNorm((attention_heads + 1) * hidden_dim, eps=1e-6)
            , nn.Dropout(dropout)
            , nn.Linear((attention_heads + 1) * hidden_dim, 1)
        )
        num_emb_layers = 2
        self.hidden_block = []
        for i in range(num_emb_layers - 1):
            self.hidden_block.extend([
                nn.LayerNorm(hidden_dim, eps=1e-6)
                , nn.Dropout(dropout)
                , nn.Linear(hidden_dim, hidden_dim)
                , nn.LeakyReLU()
            ])
            if i == num_emb_layers - 2:
                self.hidden_block.extend([nn.LayerNorm(hidden_dim, eps=1e-6)])

        self.hidden_block = nn.Sequential(*self.hidden_block)

        # Attention pooling layer
        self.ATFC = nn.Sequential(
            nn.Linear(hidden_dim, 64)
            , nn.LeakyReLU()
            , nn.LayerNorm(64, eps=1e-6)
            , nn.Linear(64, attention_heads)  # num_heads
        )
        self.weight = 0.2
        self.add_module("FC_1", nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.add_module("FC_2", nn.Linear(hidden_dim, 5106, bias=True))

        # self.batch_activate_site = nn.Parameter(torch.randn(256, 256))
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def padding_ver1(self, x, batch_id, feature_dim):
        batch_size = max(batch_id) + 1
        max_len = max(torch.unique(batch_id, return_counts=True)[1])
        batch_data = torch.zeros([batch_size, max_len, feature_dim])
        mask = torch.zeros([batch_size, max_len])
        len_0 = 0
        len_1 = 0
        for i in range(batch_size):
            len_1 = len_0 + torch.unique(batch_id, return_counts=True)[1][i]
            batch_data[i][:torch.unique(batch_id, return_counts=True)[1][i]] = x[len_0:len_1]
            mask[i][:torch.unique(batch_id, return_counts=True)[1][i]] = 1
            len_0 += torch.unique(batch_id, return_counts=True)[1][i]
        return batch_data, mask

    ### X节点， h_V节点特征 edge_index边 batch.seq???
    def forward(self, X, h_V, edge_index, seq, batch_id, batch_data, mask_data):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)
        # X shape[5044, 5, 3], batch*N，5个残基特征节点(C N Ca O )，3的坐标 edge_index shape [2, 90348]
        # h_V [batch*node, feature]
        # batch_data [batch, max_node, feature]
        # mask_data [batch, max_node] 指示每个graph里面node的情况
        h_V_baseline, _ = batch_data, mask_data
        h_V_baseline = h_V_baseline.to(self.device)
        h_V_baseline = self.input_block(h_V_baseline)
        h_V_baseline = self.hidden_block(h_V_baseline)

        # get the geometric features
        h_V_geo, h_E = get_geo_feat(X, edge_index)  # h_V_geo shape[5044, 184]   h_v shape[90348, 450]
        try:  # h_V shape [5044, 1033]
            h_V = torch.cat([h_V, h_V_geo], dim=-1)  # h_V shape[5044, 1033+184 = 1217]
        except:
            print(h_V.size())
            print(h_V_geo.size())
            print(seq)
            h_V_geo = torch.ones([h_V.shape[0], 184]).to(self.device)
            h_V = torch.cat([h_V, h_V_geo], dim=-1)

        h_V = h_V.to(self.device)
        h_V = self.Graph_encoder(h_V, edge_index, h_E, seq, batch_id)  # [num_residue, hidden_dim]
        h_V_stru, mask_baseline = self.padding_ver1(h_V.cpu(), batch_id.cpu(), h_V.shape[1])
        h_V_stru = h_V_stru.to(self.device)
        mask_baseline = mask_baseline.to(self.device)

        h_V_baseline = self.weight * h_V_baseline + (1 - self.weight) * h_V_stru

        # Attention pooling
        att = self.ATFC(h_V_baseline)  # [B, L, hid] -> [B, L, att_heads]
        att = att.masked_fill(mask_baseline[:, :, None] == 0, -1e9)  # 混合精度fp16会溢出
        # att = att.masked_fill(mask_baseline[:, :, None] == 0, -1e5)
        att = F.softmax(att, dim=1)  # [B, L, att_heads] torch.Size([64, 2511, 8])

        # use the active sites @:矩阵乘法要求第一个矩阵的列数等于第二个矩阵的行数
        # activate_site_probabilities = self.activate_site(h_V_baseline)
        # active_pool = activate_site_probabilities.transpose(1,2) @ h_V_baseline  # ([64, 1, 2511]) @ ([64, 2511, 256]) -> [64,1,2511]

        att = att.transpose(1, 2)  # [B, L, att_heads] -> [B, att_heads, L]
        h_V_baseline = att @ h_V_baseline  # [B, att_heads, hid] torch.Size([64, 8, 2511]) @ torch.Size([64, 2511, 256]) ->torch.Size([64, 8, 256])

        h_V_baseline = h_V_baseline
        # h_V_baseline = torch.cat((h_V_baseline, active_pool), 1)

        h_V_baseline = torch.flatten(h_V_baseline, start_dim=1)  # [B, att_heads,hid] -> [B, att_heads*hid]

        kcat_out = self.kcat_out_block(h_V_baseline).squeeze(1)
        km_out = self.km_out_block(h_V_baseline).squeeze(1)
        return kcat_out, km_out

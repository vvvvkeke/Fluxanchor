from sklearn.metrics import r2_score
from torch import nn, autocast
import torch

from models.LLMExtractor import LLMExtractor
from models.gnn_2d import SmilesExtractor
from models.graphEC import GraphEC
import torch.nn.functional as F

from models.mask_gae import GNNEncoder, EdgeDecoder, MaskGAE, FeatureDecoder, MaskEdge, SmilesExtractor2


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_tasks):
        super(AutomaticWeightedLoss, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_tasks))
    def forward(self, *losses):
        loss = losses[0] + losses[1]
        return loss


def padding_ver1(x, batch_id, feature_dim):
    """x: 输入张量，通常是一个特征矩阵，形状为 [num_samples, feature_dim]。batch_id: 表示每个样本所属批次的标识符，是一个一维张量。
    organize the data into batches feature_dim: 特征的维度，用于指定输出张量的形状。
    """
    batch_size = max(batch_id) + 1
    max_len = max(torch.unique(batch_id, return_counts=True)[1])  # 获取每个批次中样本的最大数量。
    batch_data = torch.zeros(
        [batch_size, max_len, feature_dim])  # : 初始化一个张量，形状为 [batch_size, max_len, feature_dim]，用于存储每个批次的数据。
    mask = torch.zeros([batch_size, max_len])  # 初始化一个掩码张量，形状为 [batch_size, max_len]，用于指示每个位置是否有效。
    len_0 = 0
    for i in range(batch_size):
        len_1 = len_0 + torch.unique(batch_id, return_counts=True)[1][i]  # len_1: 计算当前批次样本的结束索引。
        batch_data[i][:torch.unique(batch_id, return_counts=True)[1][i]] = x[
                                                                           len_0:len_1]  # 将输入 x 中对应的样本填充到 batch_data 的第 i 行。
        mask[i][:torch.unique(batch_id, return_counts=True)[1][i]] = 1  # 更新掩码 mask，将有效位置标记为 1。
        len_0 += torch.unique(batch_id, return_counts=True)[1][i]  # 更新 len_0，为下一次迭代做准备。
    return batch_data, mask  # 这个代码，把pyg的mini batch形式[batch * node, feature] 变回 [batch, node, feature]


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        # self.kcat_weight = nn.Parameter(torch.randn(2))  # ablation
        # self.km_weight = nn.Parameter(torch.randn(2))
        self.kcat_weight = nn.Parameter(torch.randn(3))
        self.km_weight = nn.Parameter(torch.randn(3))

    def forward(self, p_output, s_output, l_output):
        kcat_probabilities = nn.functional.softmax(self.kcat_weight, dim=0)
        km_probabilities = nn.functional.softmax(self.km_weight, dim=0)
        kcat_output = p_output[0] * kcat_probabilities[0] + \
                      s_output[0] * kcat_probabilities[1] + \
                      l_output[0] * kcat_probabilities[2]
        km_output   = p_output[1] * km_probabilities[0] + \
                      s_output[1] * km_probabilities[1] + \
                      l_output[1] * km_probabilities[2]
        # ablation
        # kcat_output = p_output[0] * kcat_probabilities[0] + \
        #               s_output[0] * kcat_probabilities[1]
        # km_output = p_output[1] * km_probabilities[0] + \
        #             s_output[1] * km_probabilities[1]
        return kcat_output, km_output


class Model(torch.nn.Module):
    def __init__(self, input_features, device, model_name, predict=False, rubisco_km=False, rubisco_kcat=False):
        super(Model, self).__init__()
        self.protein_extractor = GraphEC(node_input_dim=1536 + 9 + 184, edge_input_dim=450, hidden_dim=256,
                                         num_layers=3,
                                         dropout=0.2, augment_eps=0, device=device, model_name=model_name)
        self.llm_extractor = LLMExtractor(input_dim=3584, model_name=model_name)
        self.smiles_extractor = SmilesExtractor(input_features, model_name)
        self.fusion_model = Fusion()
        self.device = device
        self.model_name = model_name
        self.auto_weighted_loss = AutomaticWeightedLoss(num_tasks=2)
        self.predict = predict
        self.rubisco_km = rubisco_km
        self.rubisco_kcat = rubisco_kcat

    def get_loss(self, pred_kcat, pred_km, true_kcat, true_km):
        loss_kcat = F.mse_loss(pred_kcat, true_kcat)
        loss_km = F.mse_loss(pred_km, true_km)
        if self.rubisco_km:
            loss = self.auto_weighted_loss(0, loss_km) + 0.026026
        elif self.rubisco_kcat:
            loss = self.auto_weighted_loss(loss_kcat, 0) + 0.026026
        else:
            loss = self.auto_weighted_loss(loss_kcat, loss_km)
        return loss

    def r_squared(self, y_true, y_pred):
        # 计算总平方和
        ss_total = ((y_true - y_true.mean()) ** 2).sum()
        # 计算残差平方和
        ss_residual = ((y_true - y_pred) ** 2).sum()
        # 计算 R²
        r2 = 1 - (ss_residual / ss_total)
        return r2

    def forward(self, batch, multi_gpu):
        if multi_gpu:
            true_kcat = batch[0]
            true_km = batch[1]
            if self.model_name == "smiles":
                substrate = batch[2]
                product = batch[3]
                substrate_embedding = batch[4]
                product_embedding = batch[5]
                condition = batch[6]
                index = batch[7]
            elif self.model_name == "protein":
                protein = batch[2]
                condition = batch[3]
                index = batch[4]
            elif self.model_name == "llm":
                language = batch[2]
                index = batch[3]
            else:
                substrate = batch[2]
                product = batch[3]
                substrate_embedding = batch[4]
                product_embedding = batch[5]
                protein = batch[6]
                condition = batch[7]
                language = batch[8]
                index = batch[9]
        else:
            true_kcat = batch[0].to(self.device)
            true_km = batch[1].to(self.device)
            if self.model_name == "smiles":
                substrate = batch[2].to(self.device)
                product = batch[3].to(self.device)
                substrate_embedding = batch[4].to(self.device)
                product_embedding = batch[5].to(self.device)
                condition = batch[6].to(self.device)
                index = batch[7]
            elif self.model_name == "protein":
                protein = batch[2].to(self.device)
                condition = batch[3].to(self.device)
                index = batch[4]
            elif self.model_name == "llm":
                language = batch[2].to(self.device)
                index = batch[3]
            else:
                substrate = batch[2].to(self.device)
                product = batch[3].to(self.device)
                substrate_embedding = batch[4].to(self.device)
                product_embedding = batch[5].to(self.device)
                protein = batch[6].to(self.device)
                condition = batch[7].to(self.device)
                language = batch[8].to(self.device)
                index = batch[9]
        # with autocast(device_type="cuda"): #混合精度会溢出
        if self.model_name == "protein":
            batch_data, mask_data = padding_ver1(protein.node_feat, protein.batch, protein.node_feat.shape[1])
            p_output = self.protein_extractor(X=protein.X, h_V=protein.node_feat, edge_index=protein.edge_index,
                                              seq=protein.seq, batch_id=protein.batch, batch_data=batch_data,
                                              mask_data=mask_data)
            loss = self.get_loss(p_output[0], p_output[1], true_kcat, true_km)
            return p_output[0], p_output[1], loss, index

        if self.model_name == "smiles":
            s_output = self.smiles_extractor(substrate, product, substrate_embedding, product_embedding, condition)
            loss = self.get_loss(s_output[0], s_output[1], true_kcat, true_km)
            return s_output[0], s_output[1], loss, index

        if self.model_name == "llm":
            l_output = self.llm_extractor(language)
            loss = self.get_loss(l_output[0], l_output[1], true_kcat, true_km)
            return l_output[0], l_output[1], loss, index

        if self.model_name == "fusion":
            batch_data, mask_data = padding_ver1(protein.node_feat, protein.batch, protein.node_feat.shape[1])
            p_output = self.protein_extractor(X=protein.X, h_V=protein.node_feat, edge_index=protein.edge_index,
                                              seq=protein.seq, batch_id=protein.batch, batch_data=batch_data,
                                              mask_data=mask_data)
            s_output = self.smiles_extractor(substrate, product, substrate_embedding, product_embedding, condition)
            l_output = self.llm_extractor(language)
            kcat_output, km_output = self.fusion_model(p_output, s_output, l_output)
            if self.predict:
                return kcat_output, km_output, index
            else:
                loss = self.get_loss(kcat_output, km_output, true_kcat, true_km)
                return p_output, s_output, l_output, kcat_output, km_output, loss, index
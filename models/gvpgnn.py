import torch
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool

from models.gnn_2d import SmilesCrossAttention
from models.mace_modules.blocks import RadialEmbeddingBlock
import models.layers.gvp_layer as gvp

class Smiles_3d_Extractor(torch.nn.Module):
    def __init__(self, input_features, model_name):
        super(Smiles_3d_Extractor, self).__init__()
        self.substrate_extractor = GVPGNNModel(input_features)
        self.product_extractor = GVPGNNModel(input_features)
        self.cross_attention = SmilesCrossAttention(model_name=model_name)

        self.model_name = model_name
        # self.lin1 = nn.Linear(512 * 2, 512)
        # self.lin2 = nn.Linear(512 * 2, 512)
        # self.SLG = SubstrateLGCrossAttention()
        # self.PLG = ProductLGCrossAttention()

    def forward(self, substrate, product, substrate_embedding, product_embedding):

        substrate_output = self.substrate_extractor(h=substrate.X, x=substrate.Z, edges=substrate.edge_index, edge_attr=substrate.edge_attr, batch=substrate.batch,
                                     # substrate.morgan_fp, substrate.rdk_fp
                                     )
        product_output = self.product_extractor(h=product.X, x=substrate.Z, edges=product.edge_index, edge_attr=product.edge_attr, batch=product.batch,
                                   # product.morgan_fp,product.rdk_fp
                                   )
        # substrate_output = self.lin1(torch.cat([substrate_output, substrate_embedding], dim=1))
        # product_output = self.lin2(torch.cat([product_output, product_embedding], dim=1))
        # substrate_output = self.SLG(substrate_output, substrate_embedding)
        # product_output = self.PLG(product_output, product_embedding)
        if self.model_name == "3dgnn":
            kcat, km, lin = self.cross_attention(substrate_output, product_output)
            return kcat, km, lin
        else:
            output = self.cross_attention(substrate_output, product_output)
            return output

class GVPGNNModel(torch.nn.Module):
    """
    GVP-GNN model from "Equivariant Graph Neural Networks for 3D Macromolecular Structure".
    """
    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_layers: int = 8,
        in_dim=10000,
        out_dim=512,
        s_dim: int = 512,
        v_dim: int = 16,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        pool: str = "mean",  # sum
        residual: bool = True,
        equivariant_pred: bool = False
    ):
        """
        Initializes an instance of the GVPGNNModel class with the provided parameters.

        Parameters:
        - r_max (float): Maximum distance for Bessel basis functions (default: 10.0)
        - num_bessel (int): Number of Bessel basis functions (default: 8)
        - num_polynomial_cutoff (int): Number of polynomial cutoff basis functions (default: 5)
        - num_layers (int): Number of layers in the model (default: 5)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - s_dim (int): Dimension of the node state embeddings (default: 128)
        - v_dim (int): Dimension of the node vector embeddings (default: 16)
        - s_dim_edge (int): Dimension of the edge state embeddings (default: 32)
        - v_dim_edge (int): Dimension of the edge vector embeddings (default: 1)
        - pool (str): Global pooling method to be used (default: "sum")
        - residual (bool): Whether to use residual connections (default: True)
        - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)
        """
        super().__init__()
        
        self.r_max = r_max
        self.num_layers = num_layers
        self.equivariant_pred = equivariant_pred
        self.s_dim = s_dim
        self.v_dim = v_dim
        
        activations = (F.leaky_relu, None)
        _DEFAULT_V_DIM = (s_dim, v_dim)
        _DEFAULT_E_DIM = (s_dim_edge, v_dim_edge) 

        # Node embedding
        self.emb_in = torch.nn.Embedding(in_dim, s_dim)
        self.W_v = torch.nn.Sequential(
            gvp.LayerNorm((s_dim, 0)),
            gvp.GVP((s_dim, 0), _DEFAULT_V_DIM,
                activations=(None, None), vector_gate=True)
        )

        # Edge embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.W_e = torch.nn.Sequential(
            gvp.LayerNorm((3, 1)),
        gvp.GVP((3, 1), _DEFAULT_E_DIM,
                activations=(None, None), vector_gate=True)
        )
        
        # Stack of GNN layers
        self.layers = torch.nn.ModuleList(
            gvp.GVPConvLayer(
                _DEFAULT_V_DIM, _DEFAULT_E_DIM, 
                activations=activations, vector_gate=True,
                residual=residual
            ) 
            for _ in range(num_layers)
        )
        
        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.equivariant_pred:
            # Linear predictor for equivariant tasks using geometric features
            self.pred = torch.nn.Linear(s_dim + v_dim * 3, out_dim)
        else:
            # MLP predictor for invariant tasks using only scalar features
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(s_dim, s_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(s_dim, out_dim)
            )
    
    def forward(self, h, x, edges, edge_attr=None, batch=None, atoms=None):
        # Edge features
        vectors = x[edges[0]] - x[edges[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]

        # vectors = x[edges[0]] - x[edges[1]]  # [n_edges, 3]
        # lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        h_V = h
        # h_V = self.emb_in(h)  # (n,) -> (n, d)
        h_E = (
            edge_attr,
            torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2)
        )
        # h_E = ()
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
    
        for layer in self.layers:
            h_V = layer(h_V, edges, h_E)

        out = self.pool(gvp._merge(*h_V), batch)  # (n, d) -> (batch_size, d)
        
        if not self.equivariant_pred:
            # Select only scalars for invariant prediction
            out = out[:,:self.s_dim]
        
        return self.pred(out)  # (batch_size, out_dim)
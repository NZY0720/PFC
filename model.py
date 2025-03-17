import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# 1. Graphormer (简化版) - 支持2维输出 (V_real, V_imag)
###############################################################################

class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        # NOTE: now we do edge_bias_proj = nn.Linear(4, num_heads)
        # because edge_attr has shape [E,4]
        self.edge_bias_proj = nn.Linear(4, num_heads)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        # 1) Q,K,V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(-1, self.num_heads, self.head_dim).transpose(0,1)
        K = K.view(-1, self.num_heads, self.head_dim).transpose(0,1)
        V = V.view(-1, self.num_heads, self.head_dim).transpose(0,1)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim**0.5)

        if edge_index is not None and edge_attr is not None:
            E = edge_index.size(1)
            # shape [E,4] => project to [E,num_heads]
            edge_bias = self.edge_bias_proj(edge_attr)  # => [E, num_heads]
            edge_bias = edge_bias.transpose(0,1)        # => [num_heads, E]
            N = x.size(0)
            edge_bias_full = torch.zeros(self.num_heads, N, N, device=x.device)
            for h in range(self.num_heads):
                edge_bias_full[h].index_put_(
                    (edge_index[0], edge_index[1]),
                    edge_bias[h],
                    accumulate=True
                )
            scores = scores + edge_bias_full

        if batch is not None:
            # mask cross-graph
            N = x.size(0)
            batch_i = batch.unsqueeze(0).unsqueeze(0).expand(self.num_heads, -1, -1)
            batch_j = batch_i.transpose(1,2)
            scores = scores.masked_fill((batch_i!=batch_j), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(0,1).contiguous().view(-1, self.embed_dim)
        out = self.out_proj(out)
        return out


class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = GraphormerMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        h = self.norm1(x)
        h = self.attn(h, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = x + self.dropout(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


class Graphormer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        h = self.norm(h)
        return h  # shape [N, hidden_dim]

###############################################################################
# 2. VirtualNodePredictor: 
#    - Outputs node_probs (existence) 
#    - Outputs node_feats_pred = (V_real, V_imag) for each node
###############################################################################
class VirtualNodePredictor(nn.Module):
    """
    A simplified model:
      1) node_probs => node label=0/1
      2) node_feats_pred => (V_real,V_imag)
      3) edge_feature_head => predict (VdiffR, VdiffI, S_real, S_imag) 
         only if child label=1
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        # a trivial encoder => node_emb
        self.encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # node_exist => [N,1]
        self.node_exist_head = nn.Linear(hidden_dim, 1)
        # node_feature => [N,2]
        self.node_feature_head = nn.Linear(hidden_dim, 2)
        # edge_feature => [2*hidden_dim ->4]
        self.edge_feature_head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, x, edge_index, edge_attr, candidate_nodes):
        """
        x => shape [N, node_in_dim]
        Return => node_probs: [num_candidate_nodes], node_feats_pred: [N,2], node_emb: [N,hidden_dim]
        We'll do edge feature predictions in train_step 
        """
        node_emb = self.encoder(x)  # [N, hidden_dim]

        # node_probs
        node_scores = self.node_exist_head(node_emb).squeeze(-1)
        node_probs  = torch.sigmoid(node_scores[candidate_nodes])

        # node_feats => (V_real, V_imag)
        node_feats_pred = self.node_feature_head(node_emb) # [N,2]

        return node_probs, node_feats_pred, node_emb


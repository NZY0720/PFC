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

        # 如果edge_attr是2维 (S_real,S_imag)，则in_features=2
        # 如果edge_attr是3维，就写3
        self.edge_bias_proj = nn.Linear(2, num_heads)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 多头拆分
        Q = Q.view(-1, self.num_heads, self.head_dim).transpose(0,1)
        K = K.view(-1, self.num_heads, self.head_dim).transpose(0,1)
        V = V.view(-1, self.num_heads, self.head_dim).transpose(0,1)

        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.head_dim**0.5)

        # Edge bias
        if edge_index is not None and edge_attr is not None:
            E = edge_index.size(1)
            edge_bias = self.edge_bias_proj(edge_attr)  # [E, num_heads]
            edge_bias = edge_bias.transpose(0,1)        # [num_heads, E]
            # 构造一个 [num_heads, N, N]
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
            # 防止跨图 attention
            N = x.size(0)
            batch_i = batch.unsqueeze(0).unsqueeze(0).expand(self.num_heads, -1, -1)
            batch_j = batch_i.transpose(1,2)
            mask = (batch_i == batch_j).float()
            scores = scores.masked_fill(mask==0, float('-inf'))

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
    让 Graphormer 输出 hidden_dim(中间表示),
    再接 2个头:
      - node_exist_head => [N,1] => 二分类 => node_probs
      - node_feature_head => [N,2] => (V_real, V_imag) 便于潮流约束
    """
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hidden_dim=64,
                 num_layers=3,
                 num_heads=4,
                 dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Graphormer主体
        self.encoder = Graphormer(
            input_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # 节点存在预测 (二分类)
        self.node_exist_head = nn.Linear(hidden_dim, 1)

        # 节点特征预测 => (V_real, V_imag)
        self.node_feature_head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, edge_attr, candidate_nodes):
        """
        x: [N, node_in_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_in_dim] (e.g. 2 => S_real,S_imag)
        candidate_nodes: [num_candidate_nodes]
        Return:
          node_probs: shape [num_candidate_nodes], Sigmoid( node_scores )
          node_feats_pred: shape [N,2], interpreted as (V_real, V_imag)
        """
        device = x.device
        N = x.size(0)

        # 构建 batch=0, 并可计算 degree(若需要)
        batch = torch.zeros(N, dtype=torch.long, device=device)  # 仅单图

        # 送进Graphormer => [N, hidden_dim]
        h = self.encoder(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        # 1) 节点存在 => node_probs
        node_scores = self.node_exist_head(h).squeeze(-1)  # [N]
        node_probs = torch.sigmoid(node_scores[candidate_nodes])  # [num_candidate_nodes]

        # 2) 节点潮流特征 => (V_real, V_imag)
        node_feats_pred = self.node_feature_head(h)  # [N, 2]

        return node_probs, node_feats_pred

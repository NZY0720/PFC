import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphStructureEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.position_embedding = nn.Embedding(100, hidden_dim // 4)  # 最多100个位置编码
        self.degree_embedding = nn.Embedding(50, hidden_dim // 4)     # 最大度数49
        self.centrality_projection = nn.Linear(3, hidden_dim // 4)    # 3种中心性度量
        self.structure_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, edge_index, num_nodes):
        # 计算节点度数
        node_degrees = torch.zeros(num_nodes, device=edge_index.device)
        if edge_index.size(1) > 0:  # 确保有边存在
            unique_nodes, degree_counts = torch.unique(edge_index[0], return_counts=True)
            node_degrees[unique_nodes] = degree_counts.float()
        node_degrees = torch.clamp(node_degrees, max=49)  # 限制最大度数
        
        # 计算节点位置（例如，基于谱聚类或布局算法的位置索引）
        # 这里简化为使用度数排名作为位置的代理
        if torch.sum(node_degrees) > 0:  # 确保有非零度数
            degree_rank = torch.argsort(torch.argsort(node_degrees, descending=True))
        else:
            degree_rank = torch.zeros(num_nodes, device=edge_index.device)
        position_indices = torch.clamp(degree_rank, max=99)  # 限制最大位置索引
        
        # 计算简化的中心性度量
        # 1. 度中心性（归一化）
        if num_nodes > 1:
            degree_centrality = node_degrees / (num_nodes - 1)
        else:
            degree_centrality = node_degrees
            
        # 2. 简化的接近中心性（使用度数的倒数作为代理）
        closeness_centrality = 1 / (node_degrees + 1)
        
        # 3. 简化的介数中心性（使用度数的平方作为代理）
        deg_max = node_degrees.max()
        if deg_max > 0:
            betweenness_centrality = (node_degrees / deg_max) ** 2
        else:
            betweenness_centrality = torch.zeros_like(node_degrees)
            
        # 组合中心性度量
        centrality_features = torch.stack([
            degree_centrality,
            closeness_centrality,
            betweenness_centrality
        ], dim=1)
        
        # 获取嵌入
        position_emb = self.position_embedding(position_indices.long())
        degree_emb = self.degree_embedding(node_degrees.long())
        centrality_emb = self.centrality_projection(centrality_features)
        
        # 拼接并投影所有结构特征
        structure_emb = torch.cat([
            position_emb,
            degree_emb,
            centrality_emb,
            torch.zeros(num_nodes, self.position_embedding.embedding_dim, device=edge_index.device)  # 填充到完整维度
        ], dim=1)
        
        return self.structure_projection(structure_emb)

class TemporalGraphormerMultiHeadAttention(nn.Module):
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

        # Edge bias projection
        # Edge attributes have shape [E,4]
        self.edge_bias_proj = nn.Linear(4, num_heads)
        
        # Temporal attention
        self.temporal_q_proj = nn.Linear(embed_dim, embed_dim)
        self.temporal_k_proj = nn.Linear(embed_dim, embed_dim)
        self.temporal_v_proj = nn.Linear(embed_dim, embed_dim)
        self.temporal_out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, time_index=None, temporal_embeddings=None):
        """
        Args:
            x: Node features [N, embed_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, 4]
            batch: Batch index for each node [N]
            time_index: Time step index for each node [N]
            temporal_embeddings: Temporal embeddings for each time step [T, embed_dim]
        """
        # Handle empty edge_index
        if edge_index is None or edge_index.size(1) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            if edge_attr is not None:
                edge_attr = torch.zeros((0, edge_attr.size(1)), dtype=torch.float, device=x.device)
            else:
                edge_attr = torch.zeros((0, 4), dtype=torch.float, device=x.device)
        
        # Get number of nodes
        num_nodes = x.size(0)
        
        # 1) Spatial Graph Attention
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N, head_dim]
        K = K.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N, head_dim]
        V = V.view(-1, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)  # [num_heads, N, N]

        # Apply edge biases to attention scores
        if edge_index is not None and edge_attr is not None and edge_index.size(1) > 0:
            # SAFE VERSION: Ensure all edge indices are valid
            valid_edge_mask = (edge_index[0] >= 0) & (edge_index[0] < num_nodes) & \
                             (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
            
            # Only process valid edges
            if valid_edge_mask.any():
                valid_edge_index = edge_index[:, valid_edge_mask]
                valid_edge_attr = edge_attr[valid_edge_mask]
                
                if valid_edge_index.size(1) > 0:  # Check if we have any valid edges left
                    # Project edge attributes to bias values
                    edge_bias = self.edge_bias_proj(valid_edge_attr)  # [valid_E, num_heads]
                    edge_bias = edge_bias.transpose(0, 1)  # [num_heads, valid_E]
                    
                    # Initialize bias tensor
                    edge_bias_full = torch.zeros(self.num_heads, num_nodes, num_nodes, device=x.device)
                    
                    # Apply biases edge by edge to avoid index errors
                    for e in range(valid_edge_index.size(1)):
                        src = valid_edge_index[0, e].item()
                        dst = valid_edge_index[1, e].item()
                        for h in range(self.num_heads):
                            edge_bias_full[h, src, dst] += edge_bias[h, e]

                    # Add edge biases to attention scores
                    scores = scores + edge_bias_full

        # Apply batch masking to prevent cross-graph attention
        if batch is not None:
            # Safe batch masking
            try:
                N = x.size(0)
                batch_i = batch.unsqueeze(0).unsqueeze(0).expand(self.num_heads, -1, -1)  # [num_heads, 1, N]
                batch_j = batch_i.transpose(1, 2)  # [num_heads, N, 1]
                mask = (batch_i != batch_j)  # [num_heads, N, N]
                scores = scores.masked_fill(mask, float('-inf'))
            except Exception as e:
                print(f"Warning: Error in batch masking: {e}")
                # Continue without masking if error occurs

        # Compute attention weights and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        spatial_out = torch.matmul(attn, V)  # [num_heads, N, head_dim]
        
        # 2) Temporal Attention (if temporal data provided)
        if time_index is not None and temporal_embeddings is not None:
            # Safe handling for time_index
            if time_index.max() >= len(temporal_embeddings):
                # Clamp time_index to valid range
                time_index = torch.clamp(time_index, 0, len(temporal_embeddings) - 1)
                
            # Get temporal embedding for each node based on time_index
            node_temporal_emb = temporal_embeddings[time_index]  # [N, embed_dim]
            
            # Project temporal features
            T_Q = self.temporal_q_proj(node_temporal_emb)
            T_K = self.temporal_k_proj(node_temporal_emb)
            T_V = self.temporal_v_proj(node_temporal_emb)
            
            T_Q = T_Q.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
            T_K = T_K.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
            T_V = T_V.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
            
            # Compute temporal attention scores
            t_scores = torch.matmul(T_Q, T_K.transpose(-2, -1)) / (self.head_dim**0.5)
            
            # Apply batch masking
            if batch is not None:
                try:
                    t_scores = t_scores.masked_fill(mask, float('-inf'))
                except Exception as e:
                    print(f"Warning: Error in temporal batch masking: {e}")
            
            t_attn = F.softmax(t_scores, dim=-1)
            t_attn = self.attn_drop(t_attn)
            temporal_out = torch.matmul(t_attn, T_V)
            
            # Combine spatial and temporal outputs
            combined_out = spatial_out + temporal_out
            combined_out = combined_out.transpose(0, 1).contiguous().view(-1, self.embed_dim)
            out = self.out_proj(combined_out)
        else:
            # Only spatial attention if no temporal data
            spatial_out = spatial_out.transpose(0, 1).contiguous().view(-1, self.embed_dim)
            out = self.out_proj(spatial_out)
            
        return out

class TemporalGraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = TemporalGraphormerMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, time_index=None, temporal_embeddings=None):
        # Normalize, apply attention, add residual
        h = self.norm1(x)
        h = self.attn(h, edge_index=edge_index, edge_attr=edge_attr, batch=batch, 
                    time_index=time_index, temporal_embeddings=temporal_embeddings)
        x = x + self.dropout(h)

        # Feedforward
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x

class TemporalEmbedding(nn.Module):
    """
    Generate embeddings for different time steps
    Can handle both minute level (0-1439) and hourly level (0-23) time steps
    """
    def __init__(self, max_time_steps=1440, hidden_dim=64):
        super().__init__()
        self.max_time_steps = max_time_steps
        self.time_embedding = nn.Embedding(max_time_steps, hidden_dim)
        
        # For hourly mode (max_time_steps=24), we don't need minute embeddings
        if max_time_steps <= 24:
            # Simplified embedding for hourly data
            self.is_hourly = True
            self.hour_embedding = nn.Embedding(24, hidden_dim)
            self.combine_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            # Detailed embeddings for minute-level data
            self.is_hourly = False
            self.hour_embedding = nn.Embedding(24, hidden_dim // 2)
            self.minute_embedding = nn.Embedding(60, hidden_dim // 2)
            self.combine_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, time_indices):
        """
        Args:
            time_indices: Time indices - either minutes since midnight [0-1439] 
                          or hours [0-23] depending on initialization
        Returns:
            Temporal embeddings [batch_size, hidden_dim]
        """
        # Ensure time indices are within bounds
        time_indices = torch.clamp(time_indices, 0, self.max_time_steps - 1)
        
        # Basic time embedding
        time_emb = self.time_embedding(time_indices)
        
        if self.is_hourly:
            # For hourly data, time_indices are directly hours [0-23]
            hour_emb = self.hour_embedding(time_indices)
            # Combine embeddings
            combined_emb = torch.cat([time_emb, hour_emb], dim=-1)
        else:
            # For minute data, extract hour and minute components
            hours = torch.div(time_indices, 60, rounding_mode='floor')
            minutes = time_indices % 60
            
            # Generate hour and minute embeddings
            hour_emb = self.hour_embedding(hours)
            minute_emb = self.minute_embedding(minutes)
            
            # Combine hour and minute embeddings
            combined_emb = torch.cat([hour_emb, minute_emb], dim=-1)
            # Combine with the basic time embedding
            combined_emb = torch.cat([time_emb, combined_emb], dim=-1)
        
        # Project to final embedding size
        final_emb = self.combine_proj(combined_emb)
        
        return final_emb

class TemporalGraphormer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=3, 
                 dropout=0.1, max_time_steps=24):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal embedding
        self.temporal_embedding = TemporalEmbedding(max_time_steps, hidden_dim)
        
        # Graphormer layers
        self.layers = nn.ModuleList([
            TemporalGraphormerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Optional LSTM layer for temporal processing
        self.use_lstm = False
        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, time_index=None):
        """
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, 4]
            batch: Batch index for each node [N]
            time_index: Time step index for each node [N]
        """
        # Handle empty edge_index case
        if edge_index is None or edge_index.size(1) == 0:
            # Create dummy edge_index and edge_attr
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
            if edge_attr is None:
                # Create empty edge_attr with correct feature dimension (4)
                edge_attr = torch.zeros((0, 4), dtype=torch.float, device=x.device)
                
        # Project node features
        h = self.input_proj(x)  # [N, hidden_dim]
        
        # Get temporal embeddings if time_index is provided
        temporal_embeddings = None
        if time_index is not None:
            # Get unique time indices and clamp to valid range
            max_time = self.temporal_embedding.time_embedding.num_embeddings - 1
            time_index = torch.clamp(time_index, 0, max_time)
            unique_times = torch.unique(time_index)
            unique_times = torch.clamp(unique_times, 0, max_time)
            temporal_embeddings = self.temporal_embedding(unique_times)  # [T, hidden_dim]
        
        # Apply Graphormer layers
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr, batch=batch,
                      time_index=time_index, temporal_embeddings=temporal_embeddings)
        
        # Normalize output
        h = self.norm(h)  # [N, hidden_dim]
        
        # Apply LSTM if enabled and time_index provided
        if self.use_lstm and time_index is not None:
            # Group node embeddings by time for LSTM processing
            # This requires reshaping data which depends on your exact batch structure
            # For simplicity, this implementation is left as a placeholder
            pass
            
        return h  # [N, hidden_dim]

class TemporalVirtualNodePredictor(nn.Module):
    """
    带有时间感知的虚拟节点预测模型：
      1) node_probs => 节点标签=0/1（存在性预测）
      2) node_feats_pred => (V_real, V_imag) 电压预测
      3) edge_feature_head => 预测 (VdiffR, VdiffI, S_real, S_imag) 
         仅对存在的子节点（标签=1）
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, num_layers=3, 
                 num_heads=4, max_time_steps=24, dropout=0.1, exist_bias=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.exist_bias = exist_bias  # 新增存在性预测的偏置，正值增加预测正样本的倾向
        
        # 使用TemporalGraphormer作为编码器
        self.encoder = TemporalGraphormer(
            input_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_time_steps=max_time_steps
        )
        
        # 添加图结构编码器
        self.structure_encoder = GraphStructureEncoder(hidden_dim)
        
        # 节点存在性预测头
        self.node_exist_head = nn.Linear(hidden_dim, 1)
        
        # 节点特征预测头（电压实部和虚部）
        self.node_feature_head = nn.Linear(hidden_dim, 2)
        
        # 边特征预测头
        self.edge_feature_head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # 可选: 时间趋势头，捕捉随时间的变化
        self.predict_temporal_trend = False
        if self.predict_temporal_trend:
            self.temporal_trend_head = nn.Linear(hidden_dim, 2)  # 预测V_real, V_imag的变化

    def forward(self, x, edge_index, edge_attr, candidate_nodes, time_index=None, batch=None):
        """
        模型前向传播
        
        Args:
            x: 节点特征 [N, node_in_dim]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, 4]
            candidate_nodes: 候选节点的索引
            time_index: 每个节点的时间步索引 [N]（可选）
            batch: 每个节点的批次索引 [N]（可选）
            
        Returns:
            node_probs: 节点存在概率 [num_candidate_nodes]
            node_feats_pred: 预测的节点特征 [N, 2] (V_real, V_imag)
            node_emb: 节点嵌入 [N, hidden_dim]
        """
        # 安全检查，确保candidate_nodes在有效范围内
        num_nodes = x.size(0)
        if candidate_nodes.max() >= num_nodes:
            candidate_nodes = torch.clamp(candidate_nodes, 0, num_nodes - 1)
        
        # 获取图结构嵌入
        structure_emb = self.structure_encoder(edge_index, num_nodes)
            
        # 使用TemporalGraphormer编码节点
        node_emb = self.encoder(
            x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            time_index=time_index,
            batch=batch
        )  # [N, hidden_dim]
        
        # 结合结构嵌入
        node_emb = node_emb + structure_emb

        # 节点存在性预测
        node_scores = self.node_exist_head(node_emb).squeeze(-1)  # [N]
        
        # 添加存在性偏置，正值增加预测1的倾向
        if self.exist_bias != 0:
            node_scores = node_scores + self.exist_bias
            
        node_probs = torch.sigmoid(node_scores[candidate_nodes])  # [num_candidate_nodes]

        # 节点特征预测（电压）
        node_feats_pred = self.node_feature_head(node_emb)  # [N, 2]
        
        # 时间趋势预测（可选）
        if self.predict_temporal_trend and time_index is not None:
            temporal_trends = self.temporal_trend_head(node_emb)  # [N, 2]
            # 趋势可用于基于时间调整预测
            # 此处仅为占位实现
            pass

        return node_probs, node_feats_pred, node_emb

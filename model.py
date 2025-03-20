import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Generate embeddings for different time steps"""
    def __init__(self, max_time_steps, hidden_dim):
        super().__init__()
        self.time_embedding = nn.Embedding(max_time_steps, hidden_dim)
        
    def forward(self, time_indices):
        """
        Args:
            time_indices: Time indices [batch_size]
        Returns:
            Temporal embeddings [batch_size, hidden_dim]
        """
        return self.time_embedding(time_indices)

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
    Enhanced model for predicting virtual nodes with temporal awareness:
      1) node_probs => node label=0/1 (existence prediction)
      2) node_feats_pred => (V_real, V_imag) for voltage predictions
      3) edge_feature_head => predict (VdiffR, VdiffI, S_real, S_imag) 
         only for existing child nodes (label=1)
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, num_layers=3, 
                 num_heads=4, max_time_steps=24, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Use TemporalGraphormer as encoder
        self.encoder = TemporalGraphormer(
            input_dim=node_in_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_time_steps=max_time_steps
        )
        
        # Node existence prediction head
        self.node_exist_head = nn.Linear(hidden_dim, 1)
        
        # Node feature prediction head (voltage real and imaginary parts)
        self.node_feature_head = nn.Linear(hidden_dim, 2)
        
        # Edge feature prediction head
        self.edge_feature_head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # Optional: Temporal trend head to capture changes over time
        self.predict_temporal_trend = False
        if self.predict_temporal_trend:
            self.temporal_trend_head = nn.Linear(hidden_dim, 2)  # Predict change in V_real, V_imag

    def forward(self, x, edge_index, edge_attr, candidate_nodes, time_index=None, batch=None):
        """
        Forward pass through the model
        
        Args:
            x: Node features [N, node_in_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, 4]
            candidate_nodes: Indices of candidate nodes
            time_index: Time step index for each node [N] (optional)
            batch: Batch index for each node [N] (optional)
            
        Returns:
            node_probs: Node existence probabilities [num_candidate_nodes]
            node_feats_pred: Predicted node features [N, 2] (V_real, V_imag)
            node_emb: Node embeddings [N, hidden_dim]
        """
        # Safety check for candidate_nodes to make sure they're in valid range
        num_nodes = x.size(0)
        if candidate_nodes.max() >= num_nodes:
            candidate_nodes = torch.clamp(candidate_nodes, 0, num_nodes - 1)
            
        # Encode nodes using the TemporalGraphormer
        node_emb = self.encoder(
            x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            time_index=time_index,
            batch=batch
        )  # [N, hidden_dim]

        # Node existence prediction
        node_scores = self.node_exist_head(node_emb).squeeze(-1)  # [N]
        node_probs = torch.sigmoid(node_scores[candidate_nodes])  # [num_candidate_nodes]

        # Node feature prediction (voltages)
        node_feats_pred = self.node_feature_head(node_emb)  # [N, 2]
        
        # Temporal trend prediction (optional)
        if self.predict_temporal_trend and time_index is not None:
            temporal_trends = self.temporal_trend_head(node_emb)  # [N, 2]
            # The trends could be used to adjust predictions based on time
            # This implementation is left as a placeholder
            pass

        return node_probs, node_feats_pred, node_emb

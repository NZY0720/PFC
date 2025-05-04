import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from constraints import power_flow_constraint 

def weighted_binary_cross_entropy(predictions, targets, pos_weight=None):
    """
    计算带权重的二元交叉熵损失，处理正负样本不平衡问题
    
    Args:
        predictions: 模型预测的概率 [batch_size]
        targets: 真实标签 [batch_size]
        pos_weight: 正样本权重，如果为None则自动计算
        
    Returns:
        加权后的BCE损失
    """
    if pos_weight is None:
        # 自动计算正样本权重：负样本数量/正样本数量
        num_positives = targets.sum()
        if num_positives > 0:
            num_negatives = len(targets) - num_positives
            pos_weight = num_negatives / num_positives
        else:
            pos_weight = 1.0
    
    # 计算带正样本权重的BCE损失
    loss = F.binary_cross_entropy_with_logits(
        torch.log(predictions / (1 - predictions + 1e-7)),  # 将概率转换为logits
        targets,
        pos_weight=torch.tensor([pos_weight], device=predictions.device)
    )
    
    return loss

def train_step(model, data_sequence, optimizer, lambda_edge=1.0, lambda_phy=10.0, lambda_temporal=1.0, 
              pos_weight=None, focal_gamma=2.0, use_focal_loss=False):
    """
    训练步骤，支持加权BCE损失或Focal Loss处理不平衡问题
    
    Args:
        model: TemporalVirtualNodePredictor模型
        data_sequence: PyG Data对象列表，表示时间序列
        optimizer: PyTorch优化器
        lambda_edge: 边特征损失权重
        lambda_phy: 物理约束损失权重
        lambda_temporal: 时间一致性损失权重
        pos_weight: 正样本权重，如果为None则自动计算
        focal_gamma: Focal Loss的gamma参数
        use_focal_loss: 是否使用Focal Loss代替加权BCE
        
    Returns:
        损失值字典
    """
    model.train()
    optimizer.zero_grad()
    
    # 处理单个数据点的情况
    if not isinstance(data_sequence, list):
        data_sequence = [data_sequence]
    
    total_loss = 0.0
    total_node_loss = 0.0
    total_edge_loss = 0.0
    total_phy_loss = 0.0
    total_temporal_loss = 0.0
    sequence_length = len(data_sequence)
    
    # 存储预测结果用于计算时间一致性损失
    all_node_probs = []
    all_node_feats = []
    
    # 处理序列中的每个时间步
    for t, data in enumerate(data_sequence):
        # 前向传播
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            print(f"Warning: NaN or Inf found in input features at time step {t}")
        if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
            print(f"Warning: NaN or Inf found in edge attributes at time step {t}")
        
        node_probs, node_feats_pred, node_emb = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.candidate_nodes,
            time_index=data.time_index
        )
        
        # 存储结果用于时间一致性计算
        all_node_probs.append(node_probs)
        all_node_feats.append(node_feats_pred)
        
        # 1) 节点存在性损失(带权重的二元交叉熵)
        # 计算当前批次的正负样本分布
        if use_focal_loss:
            # Focal Loss: 下调高置信度预测的权重，强调难例
            p_t = node_probs * data.candidate_nodes_label.float() + (1 - node_probs) * (1 - data.candidate_nodes_label.float())
            focal_weight = (1 - p_t) ** focal_gamma
            
            # 应用可选的正样本权重
            if pos_weight is not None:
                # 对正样本额外加权
                weight = torch.ones_like(data.candidate_nodes_label.float())
                weight[data.candidate_nodes_label == 1] = pos_weight
                focal_weight = focal_weight * weight
            
            # 计算带权重的损失
            node_loss = F.binary_cross_entropy(node_probs, data.candidate_nodes_label.float(), reduction='none')
            node_loss = (focal_weight * node_loss).mean()
        else:
            # 使用加权BCE损失
            node_loss = weighted_binary_cross_entropy(
                node_probs, 
                data.candidate_nodes_label.float(),
                pos_weight
            )
        
        # 2) 边特征回归损失
        edge_feat_loss = torch.tensor(0.0, device=data.x.device)
        if hasattr(data, 'fc_edges') and hasattr(data, 'fc_attr'):
            fc_edges = data.fc_edges  # [2, fc_count]
            fc_attr = data.fc_attr    # [fc_count, 4]
            mask = (data.candidate_nodes_label == 1)  # 只对真实子节点回归边
            
            if mask.any():
                father_idx = fc_edges[0, mask]
                child_idx = fc_edges[1, mask]
                fc_attr_sel = fc_attr[mask]
                
                father_emb = node_emb[father_idx]    # [sum(mask), hidden_dim]
                child_emb = node_emb[child_idx]      # [sum(mask), hidden_dim]
                edge_in = torch.cat([father_emb, child_emb], dim=1)  # => [sum(mask), 2*hidden_dim]
                pred_edge_feat = model.edge_feature_head(edge_in)    # => [sum(mask), 4]
                
                edge_feat_loss = F.mse_loss(pred_edge_feat, fc_attr_sel)
        
        # 3) 物理约束损失
        phy_loss = power_flow_constraint(
            node_feats_pred,
            data.edge_index,
            data.edge_attr,
            data.candidate_nodes,
            data.candidate_nodes_label
        )
        
        # 累加各损失项
        step_loss = node_loss + lambda_edge * edge_feat_loss + lambda_phy * phy_loss
        total_loss += step_loss
        
        # 跟踪各损失组件
        total_node_loss += node_loss.item()
        total_edge_loss += edge_feat_loss.item()
        total_phy_loss += phy_loss.item()
    
    # 4) 时间一致性损失（如果序列长度>1）
    temporal_loss = torch.tensor(0.0, device=data.x.device)
    if sequence_length > 1:
        # 应用时间一致性 - 节点存在概率应平滑变化
        for t in range(sequence_length - 1):
            if len(all_node_probs[t]) == len(all_node_probs[t+1]):
                # 概率差的L2损失
                temporal_loss += F.mse_loss(all_node_probs[t], all_node_probs[t+1])
                
                # 电压预测差的L2损失（应平滑变化）
                candidate_indices = data_sequence[t].candidate_nodes
                temporal_loss += 0.1 * F.mse_loss(
                    all_node_feats[t][candidate_indices], 
                    all_node_feats[t+1][candidate_indices]
                )
        
        temporal_loss = temporal_loss / (sequence_length - 1)
        total_loss += lambda_temporal * temporal_loss
        total_temporal_loss = temporal_loss.item()
    
    # 反向传播和优化
    total_loss.backward()
    optimizer.step()
    
    # 计算平均损失
    avg_node_loss = total_node_loss / sequence_length
    avg_edge_loss = total_edge_loss / sequence_length
    avg_phy_loss = total_phy_loss / sequence_length
    
    return {
        'total_loss': total_loss.item() / sequence_length,
        'node_loss': avg_node_loss,
        'edge_loss': avg_edge_loss,
        'phy_loss': avg_phy_loss,
        'temporal_loss': total_temporal_loss if sequence_length > 1 else 0.0
    }

def visualize_results(data, node_probs, node_feats_pred, iteration=0, threshold=0.5, save_path='./results'):
    """
    Visualize the power grid graph with predicted nodes.
    """
    G = nx.Graph()
    
    # Identify known nodes based on mask
    known_nodes = []
    if hasattr(data, 'node_known_mask'):
        known_nodes = data.node_known_mask.nonzero().view(-1).tolist()
    
    # Get candidate nodes
    candidate_nodes = data.candidate_nodes.tolist()
    
    # Identify predicted existing nodes
    pred_exist_nodes = [candidate_nodes[i] for i, p in enumerate(node_probs) if p >= threshold]
    
    # Add nodes to graph
    G.add_nodes_from([(i, {'type': 'known'}) for i in known_nodes])
    G.add_nodes_from([(i, {'type': 'candidate'}) for i in candidate_nodes])
    
    # Add edges if available and debug edge information
    edge_count = 0
    if data.edge_index is not None:
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        
        # Debug first few edges
        if data.edge_index.shape[1] > 0:
            print("First 10 edges:")
            for i in range(min(10, data.edge_index.shape[1])):
                src = data.edge_index[0, i].item()
                dst = data.edge_index[1, i].item()
                print(f"  Edge {i}: {src} -> {dst}")
        
        edge_list = data.edge_index.t().tolist()
        G.add_edges_from(edge_list, type='known')
        edge_count = len(edge_list)
    
    print(f"Added {edge_count} edges to visualization graph")
    print(f"NetworkX graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create layout and plot
    pos = nx.spring_layout(G, seed=42, k=0.5)  # Increased k for better spacing
    plt.figure(figsize=(12, 10))
    
    # Analyze edge types and create separate lists for different visualizations
    father_father_edges = []
    father_child_edges = []
    for u, v in G.edges():
        if u < len(known_nodes) and v < len(known_nodes):
            father_father_edges.append((u, v))
        else:
            father_child_edges.append((u, v))
    
    # Draw different edge types with increased visibility
    if father_father_edges:
        nx.draw_networkx_edges(G, pos, edgelist=father_father_edges, 
                              edge_color='black', alpha=0.9, width=2.0)
    
    if father_child_edges:
        nx.draw_networkx_edges(G, pos, edgelist=father_child_edges, 
                              edge_color='blue', alpha=0.7, width=1.5, style='dashed')
    
    # Draw different node types
    nx.draw_networkx_nodes(G, pos, nodelist=known_nodes, node_color='lightblue', 
                          label='Known Nodes', node_size=100)  # Increased node size
    
    # Mark candidate nodes based on predictions
    nx.draw_networkx_nodes(G, pos, nodelist=pred_exist_nodes, node_color='green', 
                          label='Predicted Exist', node_size=80)
    
    # Draw labels for the nodes
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Add voltage values as node attributes if node_feats_pred is provided
    if node_feats_pred is not None:
        node_labels = {}
        for i, node in enumerate(G.nodes()):
            if i < len(node_feats_pred):
                v_real = node_feats_pred[i, 0].item()
                v_imag = node_feats_pred[i, 1].item()
                node_labels[node] = f"{v_real:.2f}+{v_imag:.2f}i"
        
        # Only show labels for a subset to avoid clutter
        subset_nodes = known_nodes[:10] + pred_exist_nodes[:5]
        subset_labels = {node: node_labels[node] for node in subset_nodes if node in node_labels}
        pos_labels = {k: (v[0], v[1] + 0.08) for k, v in pos.items()}  # Offset labels
        nx.draw_networkx_labels(G, pos_labels, labels=subset_labels, font_size=8, font_color='red')
    
    # Convert time step from minutes to HHMM format for display
    time_info = ""
    if hasattr(data, 'time_step'):
        minutes = data.time_step.item()
        hours = minutes // 60
        mins = minutes % 60
        time_info = f" - Time {hours:02d}:{mins:02d}"
    
    plt.title(f'Power Grid{time_info} - Iteration {iteration}')
    plt.legend()
    plt.axis('off')
    
    # Save the visualization
    os.makedirs(save_path, exist_ok=True)
    outpath = os.path.join(save_path, f'iteration_{iteration}.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Visualization saved to {outpath}")
    
    # If very few edges are detected, try to reconstruct a more meaningful structure for a second visualization
    if G.number_of_edges() < G.number_of_nodes() / 4:  # If too few edges
        print("[WARN] Few edges detected, creating additional structural visualization")
        plt.figure(figsize=(12, 10))
        
        # Create a new graph with a structural layout
        G_structural = nx.Graph()
        G_structural.add_nodes_from(G.nodes(data=True))
        
        # Create a tree structure connecting known nodes
        known_nodes_sorted = sorted(known_nodes)
        for i in range(1, len(known_nodes_sorted)):
            G_structural.add_edge(known_nodes_sorted[0], known_nodes_sorted[i], type='structural')
        
        # Connect predicted nodes to nearest known node
        for pred_node in pred_exist_nodes:
            closest_node = min(known_nodes, key=lambda x: abs(x - pred_node))
            G_structural.add_edge(pred_node, closest_node, type='predicted')
        
        # Add original edges also
        G_structural.add_edges_from(G.edges(), type='original')
        
        # Create new layout
        pos_structural = nx.spring_layout(G_structural, seed=42, k=0.3)
        
        # Draw different edge types
        original_edges = [(u, v) for u, v, d in G_structural.edges(data=True) if d.get('type') == 'original']
        structural_edges = [(u, v) for u, v, d in G_structural.edges(data=True) if d.get('type') == 'structural']
        predicted_edges = [(u, v) for u, v, d in G_structural.edges(data=True) if d.get('type') == 'predicted']
        
        nx.draw_networkx_edges(G_structural, pos_structural, edgelist=original_edges, 
                              edge_color='black', width=2.0, alpha=0.9)
        nx.draw_networkx_edges(G_structural, pos_structural, edgelist=structural_edges, 
                              edge_color='gray', width=1.5, style='dashed', alpha=0.7)
        nx.draw_networkx_edges(G_structural, pos_structural, edgelist=predicted_edges, 
                              edge_color='green', width=1.5, alpha=0.7)
        
        # Draw nodes
        nx.draw_networkx_nodes(G_structural, pos_structural, nodelist=known_nodes, 
                              node_color='lightblue', label='Known Nodes', node_size=100)
        nx.draw_networkx_nodes(G_structural, pos_structural, nodelist=pred_exist_nodes, 
                              node_color='green', label='Predicted Exist', node_size=80)
        
        # Draw labels
        nx.draw_networkx_labels(G_structural, pos_structural, font_size=8)
        
        plt.title(f'Power Grid (Structural View){time_info} - Iteration {iteration}')
        plt.legend()
        plt.axis('off')
        
        # Save the structural visualization
        structural_outpath = os.path.join(save_path, f'structural_{iteration}.png')
        plt.savefig(structural_outpath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Structural visualization saved to {structural_outpath}")
    
    # Additional metrics visualization
    if node_probs is not None and hasattr(data, 'candidate_nodes_label'):
        # Get true labels
        true_labels = data.candidate_nodes_label.cpu().numpy()
        predicted_probs = node_probs.cpu().numpy()
        
        # Create histogram of probabilities
        plt.figure(figsize=(10, 6))
        
        # Separate into positive and negative examples
        pos_probs = predicted_probs[true_labels == 1] if len(true_labels[true_labels == 1]) > 0 else []
        neg_probs = predicted_probs[true_labels == 0] if len(true_labels[true_labels == 0]) > 0 else []
        
        if len(pos_probs) > 0:
            plt.hist(pos_probs, alpha=0.7, bins=20, color='green', label='True Positive')
        if len(neg_probs) > 0:
            plt.hist(neg_probs, alpha=0.7, bins=20, color='red', label='True Negative')
        
        plt.axvline(x=threshold, linestyle='--', color='black', label=f'Threshold ({threshold})')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title(f'Distribution of Node Existence Probabilities{time_info}')
        plt.legend()
        
        # Save histogram
        hist_path = os.path.join(save_path, f'prob_hist_{iteration}.png')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Probability histogram saved to {hist_path}")

def visualize_temporal_results(data_sequence, node_probs_sequence, node_feats_sequence, iteration=0, threshold=0.5, save_path='./results'):
    """
    Visualize temporal predictions
    
    Args:
        data_sequence: List of PyG Data objects representing a temporal sequence
        node_probs_sequence: List of node probabilities for each time step
        node_feats_sequence: List of node feature predictions for each time step
        iteration: Iteration number or identifier for the filename
        threshold: Probability threshold for considering a node to exist
        save_path: Directory to save visualization
    """
    sequence_length = len(data_sequence)
    
    # Individual time step visualizations
    for t in range(sequence_length):
        visualize_results(
            data_sequence[t],
            node_probs_sequence[t],
            node_feats_sequence[t],
            iteration=f"{iteration}_t{t}",
            threshold=threshold,
            save_path=save_path
        )
    
    # Temporal visualization - probability over time
    if sequence_length > 1:
        plt.figure(figsize=(12, 6))
        
        # Extract candidate node probabilities for each time step
        prob_data = []
        for t, (data, probs) in enumerate(zip(data_sequence, node_probs_sequence)):
            true_labels = data.candidate_nodes_label.cpu().numpy()
            for i, (prob, label) in enumerate(zip(probs.cpu().numpy(), true_labels)):
                prob_data.append({
                    'time': t,
                    'node': i,
                    'prob': prob,
                    'label': label
                })
        
        # Plot probabilities over time for a subset of nodes
        unique_nodes = np.unique([d['node'] for d in prob_data])
        selected_nodes = unique_nodes[:min(10, len(unique_nodes))]
        
        for node in selected_nodes:
            node_data = [d for d in prob_data if d['node'] == node]
            times = [d['time'] for d in node_data]
            probs = [d['prob'] for d in node_data]
            label = node_data[0]['label']
            
            color = 'green' if label == 1 else 'red'
            linestyle = 'solid' if label == 1 else 'dashed'
            
            plt.plot(times, probs, marker='o', linestyle=linestyle, color=color, 
                     label=f"Node {node} ({'Real' if label == 1 else 'Fake'})")
        
        plt.axhline(y=threshold, linestyle='--', color='black', label=f'Threshold ({threshold})')
        plt.xlabel('Time Step')
        plt.ylabel('Existence Probability')
        plt.title('Node Existence Probabilities Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        os.makedirs(save_path, exist_ok=True)
        temporal_path = os.path.join(save_path, f'temporal_probs_{iteration}.png')
        plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Temporal probability plot saved to {temporal_path}")
        
        # Voltage magnitude change over time
        plt.figure(figsize=(12, 6))
        
        voltage_data = []
        for t, (data, node_feats) in enumerate(zip(data_sequence, node_feats_sequence)):
            for i, node_idx in enumerate(data.candidate_nodes.cpu().numpy()):
                v_real = node_feats[node_idx, 0].item()
                v_imag = node_feats[node_idx, 1].item()
                magnitude = np.sqrt(v_real**2 + v_imag**2)
                label = data.candidate_nodes_label[i].item()
                
                voltage_data.append({
                    'time': t,
                    'node': i,
                    'magnitude': magnitude,
                    'label': label
                })
        
        # Plot voltage magnitudes over time for a subset of nodes
        for node in selected_nodes:
            node_data = [d for d in voltage_data if d['node'] == node]
            if not node_data:
                continue
                
            times = [d['time'] for d in node_data]
            magnitudes = [d['magnitude'] for d in node_data]
            label = node_data[0]['label']
            
            color = 'blue' if label == 1 else 'orange'
            linestyle = 'solid' if label == 1 else 'dashed'
            
            plt.plot(times, magnitudes, marker='o', linestyle=linestyle, color=color, 
                     label=f"Node {node} ({'Real' if label == 1 else 'Fake'})")
        
        plt.xlabel('Time Step')
        plt.ylabel('Voltage Magnitude')
        plt.title('Voltage Magnitude Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        voltage_path = os.path.join(save_path, f'voltage_magnitudes_{iteration}.png')
        plt.savefig(voltage_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Voltage magnitude plot saved to {voltage_path}")

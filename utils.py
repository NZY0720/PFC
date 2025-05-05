import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

from constraints import power_flow_constraint 

def weighted_binary_cross_entropy(predictions, targets, pos_weight=None):
    """Calculate weighted binary cross entropy loss for handling class imbalance"""
    if pos_weight is None:
        # Calculate weight automatically if not provided
        num_positives = targets.sum()
        if num_positives > 0:
            num_negatives = len(targets) - num_positives
            pos_weight = num_negatives / num_positives
        else:
            pos_weight = 1.0
    
    # Convert probabilities to logits for weighted BCE loss
    loss = F.binary_cross_entropy_with_logits(
        torch.log(predictions / (1 - predictions + 1e-7)),
        targets,
        pos_weight=torch.tensor([pos_weight], device=predictions.device)
    )
    
    return loss

def train_step(model, data_sequence, optimizer, lambda_edge=1.0, lambda_phy=10.0, lambda_temporal=1.0, 
              pos_weight=None, focal_gamma=2.0, use_focal_loss=False):
    """Training step with support for weighted loss functions"""
    model.train()
    optimizer.zero_grad()
    
    # Handle single data point
    if not isinstance(data_sequence, list):
        data_sequence = [data_sequence]
    
    total_loss = 0.0
    total_node_loss = 0.0
    total_edge_loss = 0.0
    total_phy_loss = 0.0
    total_temporal_loss = 0.0
    sequence_length = len(data_sequence)
    
    # Store predictions for temporal loss calculation
    all_node_probs = []
    all_node_feats = []
    
    # Process each time step in the sequence
    for t, data in enumerate(data_sequence):
        # Forward pass
        node_probs, node_feats_pred, node_emb = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.candidate_nodes,
            time_index=data.time_index
        )
        
        all_node_probs.append(node_probs)
        all_node_feats.append(node_feats_pred)
        
        # 1) Node existence loss
        if use_focal_loss:
            # Focal Loss calculation
            p_t = node_probs * data.candidate_nodes_label.float() + (1 - node_probs) * (1 - data.candidate_nodes_label.float())
            focal_weight = (1 - p_t) ** focal_gamma
            
            if pos_weight is not None:
                weight = torch.ones_like(data.candidate_nodes_label.float())
                weight[data.candidate_nodes_label == 1] = pos_weight
                focal_weight = focal_weight * weight
            
            node_loss = F.binary_cross_entropy(node_probs, data.candidate_nodes_label.float(), reduction='none')
            node_loss = (focal_weight * node_loss).mean()
        else:
            # Weighted BCE loss
            node_loss = weighted_binary_cross_entropy(
                node_probs, 
                data.candidate_nodes_label.float(),
                pos_weight
            )
        
        # 2) Edge feature loss - only for true child nodes
        edge_feat_loss = torch.tensor(0.0, device=data.x.device)
        if hasattr(data, 'fc_edges') and hasattr(data, 'fc_attr'):
            fc_edges = data.fc_edges
            fc_attr = data.fc_attr
            mask = (data.candidate_nodes_label == 1)
            
            if mask.any():
                father_idx = fc_edges[0, mask]
                child_idx = fc_edges[1, mask]
                fc_attr_sel = fc_attr[mask]
                
                father_emb = node_emb[father_idx]
                child_emb = node_emb[child_idx]
                edge_in = torch.cat([father_emb, child_emb], dim=1)
                pred_edge_feat = model.edge_feature_head(edge_in)
                
                edge_feat_loss = F.mse_loss(pred_edge_feat, fc_attr_sel)
        
        # 3) Physics constraint loss
        phy_loss = power_flow_constraint(
            node_feats_pred,
            data.edge_index,
            data.edge_attr,
            data.candidate_nodes,
            data.candidate_nodes_label
        )
        
        # Combine losses for this time step
        step_loss = node_loss + lambda_edge * edge_feat_loss + lambda_phy * phy_loss
        total_loss += step_loss
        
        # Track loss components
        total_node_loss += node_loss.item()
        total_edge_loss += edge_feat_loss.item()
        total_phy_loss += phy_loss.item()
    
    # 4) Temporal consistency loss for sequences
    temporal_loss = torch.tensor(0.0, device=data.x.device)
    if sequence_length > 1:
        # Calculate temporal consistency for node probabilities and voltages
        for t in range(sequence_length - 1):
            if len(all_node_probs[t]) == len(all_node_probs[t+1]):
                # Probability smoothness
                temporal_loss += F.mse_loss(all_node_probs[t], all_node_probs[t+1])
                
                # Voltage smoothness
                candidate_indices = data_sequence[t].candidate_nodes
                temporal_loss += 0.1 * F.mse_loss(
                    all_node_feats[t][candidate_indices], 
                    all_node_feats[t+1][candidate_indices]
                )
        
        temporal_loss = temporal_loss / (sequence_length - 1)
        total_loss += lambda_temporal * temporal_loss
        total_temporal_loss = temporal_loss.item()
    else:
        total_temporal_loss = 0.0
    
    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()
    
    # Calculate average losses
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
    """Visualize the power grid with predictions"""
    # Create the graph
    G = nx.Graph()
    
    # Number of original visible (father) nodes
    num_father_nodes = data.node_known_mask.sum().item()
    num_virtual_nodes = len(data.candidate_nodes)
    
    # Create node indices
    father_indices = range(num_father_nodes)  # Original visible nodes
    virtual_indices = range(num_father_nodes, num_father_nodes + num_virtual_nodes)  # Virtual nodes
    
    # Add nodes to graph
    G.add_nodes_from([(i, {'type': 'father'}) for i in father_indices])
    G.add_nodes_from([(i, {'type': 'virtual'}) for i in virtual_indices])
    
    # Get predicted existing virtual nodes
    virtual_pred_exist = [
        virtual_indices[i] for i, p in enumerate(node_probs) if p >= threshold
    ]
    
    # Add edges between father nodes (original grid structure)
    father_father_edges = []
    if data.edge_index is not None and data.edge_index.size(1) > 0:
        for i in range(data.edge_index.size(1)):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            
            # Only include edges between father nodes
            if src < num_father_nodes and dst < num_father_nodes:
                if (src, dst) not in father_father_edges and (dst, src) not in father_father_edges:
                    father_father_edges.append((src, dst))
    
    G.add_edges_from(father_father_edges, type='father-father')
    
    # Add edges between father and virtual nodes
    father_virtual_edges = []
    for i in range(num_virtual_nodes):
        father_idx = i  # Each father maps to its corresponding virtual node
        virtual_idx = num_father_nodes + i
        father_virtual_edges.append((father_idx, virtual_idx))
    
    G.add_edges_from(father_virtual_edges, type='father-virtual')
    
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Father-father edges: {len(father_father_edges)}")
    print(f"Father-virtual edges: {len(father_virtual_edges)}")
    print(f"Predicted existing virtual nodes: {len(virtual_pred_exist)}")
    
    # Create a spring layout
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Draw the different edge types
    nx.draw_networkx_edges(G, pos, edgelist=father_father_edges, 
                          edge_color='black', alpha=0.6, width=1.0)
    
    # Highlight edges to predicted existing virtual nodes
    pred_edges = [(i, num_father_nodes + i) for i in range(num_father_nodes) 
                 if num_father_nodes + i in virtual_pred_exist]
    
    nx.draw_networkx_edges(G, pos, edgelist=pred_edges, 
                          edge_color='green', alpha=0.8, width=2.0)
    
    # Other father-virtual edges as dashed
    other_edges = [(i, j) for i, j in father_virtual_edges if (i, j) not in pred_edges]
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, 
                          edge_color='gray', alpha=0.3, width=0.5, style='dashed')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=father_indices, 
                          node_color='lightblue', label='Visible Nodes', node_size=100)
    
    # Predicted existing virtual nodes
    nx.draw_networkx_nodes(G, pos, nodelist=virtual_pred_exist, 
                          node_color='green', label='Predicted Existing Nodes', node_size=80)
    
    # Other virtual nodes
    other_virtual = [n for n in virtual_indices if n not in virtual_pred_exist]
    nx.draw_networkx_nodes(G, pos, nodelist=other_virtual, 
                          node_color='lightgray', alpha=0.3, node_size=50)
    
    # Add node labels for father nodes
    labels = {i: str(i) for i in father_indices}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
    # Add time info if available
    time_info = ""
    if hasattr(data, 'time_step'):
        minutes = data.time_step.item()
        hours = minutes // 60
        mins = minutes % 60
        time_info = f" - Time {hours:02d}:{mins:02d}"
    
    plt.title(f'Power Grid{time_info} - Iteration {iteration}')
    plt.legend()
    plt.axis('off')
    
    # Save visualization
    os.makedirs(save_path, exist_ok=True)
    outpath = os.path.join(save_path, f'iteration_{iteration}.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Visualization saved to {outpath}")
    
    # Create probability histogram
    if node_probs is not None and hasattr(data, 'candidate_nodes_label'):
        # Get ground truth labels
        true_labels = data.candidate_nodes_label.cpu().numpy()
        pred_probs = node_probs.cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        
        # Split by positive/negative examples
        pos_probs = pred_probs[true_labels == 1] if len(true_labels[true_labels == 1]) > 0 else []
        neg_probs = pred_probs[true_labels == 0] if len(true_labels[true_labels == 0]) > 0 else []
        
        if len(pos_probs) > 0:
            plt.hist(pos_probs, alpha=0.7, bins=20, color='green', label='True Positive')
        if len(neg_probs) > 0:
            plt.hist(neg_probs, alpha=0.7, bins=20, color='red', label='True Negative')
        
        plt.axvline(x=threshold, linestyle='--', color='black', label=f'Threshold ({threshold})')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.title(f'Distribution of Virtual Node Existence Probabilities{time_info}')
        plt.legend()
        
        # Save histogram
        hist_path = os.path.join(save_path, f'prob_hist_{iteration}.png')
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Probability histogram saved to {hist_path}")

def visualize_temporal_results(data_sequence, node_probs_sequence, node_feats_sequence, iteration=0, threshold=0.5, save_path='./results'):
    """Visualize results across multiple time steps"""
    sequence_length = len(data_sequence)
    
    # Create individual visualizations for each time step
    for t in range(sequence_length):
        visualize_results(
            data_sequence[t],
            node_probs_sequence[t],
            node_feats_sequence[t],
            iteration=f"{iteration}_t{t}",
            threshold=threshold,
            save_path=save_path
        )
    
    # Create temporal plots for sequences
    if sequence_length > 1:
        # Probability trend chart
        plt.figure(figsize=(12, 6))
        
        # Collect data for plotting
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
        
        # Plot trends for a subset of nodes
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
                     label=f"Node {node} ({'Exists' if label == 1 else 'Does Not Exist'})")
        
        plt.axhline(y=threshold, linestyle='--', color='black', label=f'Threshold ({threshold})')
        plt.xlabel('Time Step')
        plt.ylabel('Existence Probability')
        plt.title('Virtual Node Existence Probabilities Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the temporal probability plot
        os.makedirs(save_path, exist_ok=True)
        temporal_path = os.path.join(save_path, f'temporal_probs_{iteration}.png')
        plt.savefig(temporal_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Temporal probability plot saved to {temporal_path}")
        
        # Voltage magnitude chart
        plt.figure(figsize=(12, 6))
        
        # Collect voltage data
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
        
        # Plot voltage trends
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
                     label=f"Node {node} ({'Exists' if label == 1 else 'Does Not Exist'})")
        
        plt.xlabel('Time Step')
        plt.ylabel('Voltage Magnitude')
        plt.title('Voltage Magnitude Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the voltage plot
        voltage_path = os.path.join(save_path, f'voltage_magnitudes_{iteration}.png')
        plt.savefig(voltage_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Voltage magnitude plot saved to {voltage_path}")

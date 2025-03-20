import torch

def power_flow_constraint(
    node_feats_pred,   # [N,2], interpret as (V_real, V_imag)
    edge_index,        # [2,E]
    edge_attr,         # [E,4] => [Vdiff_Real, Vdiff_Imag, S_Real, S_Imag]
    candidate_nodes,
    candidate_nodes_label,
    lambda_child=1.0
):
    """
    Implements physics-based power flow constraints for the graph model.
    
    Args:
        node_feats_pred: Predicted node features (voltage real and imaginary parts)
        edge_index: Edge indices
        edge_attr: Edge attributes (voltage differences and power flow values)
        candidate_nodes: Indices of candidate nodes
        candidate_nodes_label: Binary labels for candidate nodes (1=exists, 0=doesn't exist)
        lambda_child: Weight for father-child constraints
        
    Returns:
        Physics constraint loss
    """
    # Extract real and imaginary voltage components
    V_real = node_feats_pred[:, 0]
    V_imag = node_feats_pred[:, 1]

    # Get source and target nodes for each edge
    src = edge_index[0]
    tgt = edge_index[1]

    # Get voltage values for source and target nodes
    V_real_src = V_real[src]
    V_imag_src = V_imag[src]
    V_real_tgt = V_real[tgt]
    V_imag_tgt = V_imag[tgt]

    # True values from the edge attributes
    Vdiff_Real_true = edge_attr[:, 0]
    Vdiff_Imag_true = edge_attr[:, 1]
    S_real_true = edge_attr[:, 2]
    S_imag_true = edge_attr[:, 3]

    # Predict voltage differences (V_src - V_tgt)
    Vdiff_Real_pred = V_real_src - V_real_tgt
    Vdiff_Imag_pred = V_imag_src - V_imag_tgt
    
    # Calculate squared error for voltage differences
    Vdiff_loss = (Vdiff_Real_pred - Vdiff_Real_true)**2 + (Vdiff_Imag_pred - Vdiff_Imag_true)**2

    # Predict power flow: S = V_src * conj(I)
    # For power flow S, we use S = V_src * V_tgt* (assuming unit impedance)
    # Real part: V_real_src * V_real_tgt + V_imag_src * V_imag_tgt
    # Imaginary part: V_imag_src * V_real_tgt - V_real_src * V_imag_tgt
    S_real_pred = V_real_src * V_real_tgt + V_imag_src * V_imag_tgt
    S_imag_pred = V_imag_src * V_real_tgt - V_real_src * V_imag_tgt
    
    # Calculate squared error for power flow
    S_loss = (S_real_pred - S_real_true)**2 + (S_imag_pred - S_imag_true)**2

    # Combine voltage and power losses
    all_loss = Vdiff_loss + S_loss  # Shape = [E]

    # Distinguish between father-father edges and father-child edges
    # Father-child edges are edges where at least one endpoint is a candidate node with label=1
    candi_label_1 = set(candidate_nodes[(candidate_nodes_label == 1)].tolist())
    
    # Create mask for child-related edges
    child_mask = []
    for i in range(edge_index.size(1)):
        s = src[i].item()
        t = tgt[i].item()
        if (s in candi_label_1) or (t in candi_label_1):
            child_mask.append(True)
        else:
            child_mask.append(False)
    child_mask = torch.tensor(child_mask, dtype=torch.bool, device=all_loss.device)

    # Calculate separate losses for father-father and father-child edges
    loss_father_father = torch.mean(all_loss[~child_mask]) if (~child_mask).any() else torch.tensor(0.0, device=all_loss.device)
    loss_father_child = torch.mean(all_loss[child_mask]) if child_mask.any() else torch.tensor(0.0, device=all_loss.device)

    # Combine losses with weighting
    loss = loss_father_father + lambda_child * loss_father_child
    
    return loss

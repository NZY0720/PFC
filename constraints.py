import torch

def power_flow_constraint(
    node_feats_pred,   
    edge_index,
    edge_attr,
    candidate_nodes,
    candidate_nodes_label,
    lambda_child=1.0
):
    
    src = edge_index[0]
    tgt = edge_index[1]

    V_real = node_feats_pred[:,0]
    V_imag = node_feats_pred[:,1]
    V_real_src, V_imag_src = V_real[src], V_imag[src]
    V_real_tgt, V_imag_tgt = V_real[tgt], V_imag[tgt]

    S_real_pred = V_real_src * V_real_tgt + V_imag_src * V_imag_tgt
    S_imag_pred = V_imag_src * V_real_tgt - V_real_src * V_imag_tgt

    S_real_true = edge_attr[:,0]
    S_imag_true = edge_attr[:,1]

    
    all_loss = (S_real_pred - S_real_true)**2 + (S_imag_pred - S_imag_true)**2
   
    candi_label_1 = set(
        candidate_nodes[(candidate_nodes_label == 1)].tolist()
    )
    child_mask = []
    for i in range(edge_index.size(1)):
        s = src[i].item()
        t = tgt[i].item()
        if (s in candi_label_1) or (t in candi_label_1):
            child_mask.append(True)
        else:
            child_mask.append(False)

    child_mask = torch.tensor(child_mask, dtype=torch.bool, device=all_loss.device)

    loss_father_father = torch.mean(all_loss[~child_mask])  
    loss_father_child  = torch.mean(all_loss[child_mask])   
    loss = loss_father_father + lambda_child*loss_father_child

    return loss

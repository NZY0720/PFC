# constraints.py
import torch

def power_flow_constraint(V_real_all, V_imag_all, edge_index, S_real_true, S_imag_true):
    src_nodes, tgt_nodes = edge_index[0], edge_index[1]

    V_real_src, V_imag_src = V_real_all[src_nodes], V_imag_all[src_nodes]
    V_real_tgt, V_imag_tgt = V_real_all[tgt_nodes], V_imag_all[tgt_nodes]

    S_real_pred = V_real_src * V_real_tgt + V_imag_src * V_imag_tgt
    S_imag_pred = V_imag_src * V_real_tgt - V_real_src * V_imag_tgt

    loss_real = torch.mean((S_real_pred - S_real_true)**2)
    loss_imag = torch.mean((S_imag_pred - S_imag_true)**2)

    return loss_real + loss_imag

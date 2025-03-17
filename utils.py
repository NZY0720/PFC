import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import os

from constraints import power_flow_constraint  # 你贴出的版本

def train_step(model, data, optimizer, lambda_child=1.0):
    """
    单次训练:
      - 前向 => (node_probs, node_feats_pred)
      - BCE(node_probs, candidate_nodes_label)
      - phy_loss => power_flow_constraint(node_feats_pred, data, ...)
      - total_loss => sum
    """
    model.train()
    optimizer.zero_grad()

    # forward
    node_probs, node_feats_pred = model(
        data.x,
        data.edge_index,
        data.edge_attr,
        data.candidate_nodes
    )
    # 1) 节点二分类损失
    node_loss = torch.tensor(0.0, device=data.x.device)
    if hasattr(data, 'candidate_nodes_label'):
        node_loss = F.binary_cross_entropy(node_probs, data.candidate_nodes_label.float())

    # 2) 潮流约束 => power_flow_constraint
    #   node_feats_pred形状 [N,2], interpret as V_real, V_imag
    phy_loss = torch.tensor(0.0, device=data.x.device)
    if data.edge_index is not None and data.edge_attr is not None:
        # 传入 candidate_nodes和candidate_nodes_label, 以区分父-子
        phy_loss = power_flow_constraint(
            node_feats_pred,
            data.edge_index,
            data.edge_attr,
            data.candidate_nodes,
            data.candidate_nodes_label,
            lambda_child=1.0   # 若想加大对父-子边的约束，可改
        )

    total_loss = node_loss + lambda_child * phy_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), node_loss.item(), phy_loss.item()

def visualize_results(data, node_probs, node_feats_pred, iteration=0, threshold=0.5, save_path='./results'):
    """
    简单可视化: 
      - known_nodes(浅蓝), pred_exist_nodes(绿色)
      - 不处理 edge_probs
    """
    G = nx.Graph()
    known_nodes = []
    if hasattr(data, 'node_known_mask'):
        known_nodes = data.node_known_mask.nonzero().view(-1).tolist()
    candidate_nodes = data.candidate_nodes.tolist()

    # 预测存在节点
    pred_exist_nodes = [candidate_nodes[i] for i, p in enumerate(node_probs) if p>=threshold]

    # 构造图
    G.add_nodes_from(known_nodes, type='known')
    G.add_nodes_from(candidate_nodes, type='candidate')
    if data.edge_index is not None:
        edge_list = data.edge_index.t().tolist()
        G.add_edges_from(edge_list, type='known')

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10,8))
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=known_nodes, node_color='lightblue', label='Known', node_size=50)
    nx.draw_networkx_nodes(G, pos, nodelist=pred_exist_nodes, node_color='green', label='Predicted exist', node_size=50)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f'Iteration {iteration}')
    plt.axis('off')

    os.makedirs(save_path, exist_ok=True)
    outpath = os.path.join(save_path, f'iteration_{iteration}.png')
    plt.savefig(outpath)
    plt.close()
    print(f"[INFO] Visualization saved to {outpath}")

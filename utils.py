import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import os

from constraints import power_flow_constraint  # 你贴出的版本

def train_step(model, data, optimizer, lambda_edge=1.0, lambda_phy=10.0):
    model.train()
    optimizer.zero_grad()

    # forward => (node_probs, node_feats_pred, node_emb)
    node_probs, node_feats_pred, node_emb = model(
        data.x,
        data.edge_index,
        data.edge_attr,
        data.candidate_nodes
    )

    # 1) 节点loss(BCE)
    node_loss = F.binary_cross_entropy(node_probs, data.candidate_nodes_label.float())

    # 2) 边特征回归 loss（示例，与原代码保持一致）
    edge_feat_loss = torch.tensor(0.0, device=data.x.device)
    if hasattr(data, 'fc_edges') and hasattr(data, 'fc_attr'):
        fc_edges = data.fc_edges  # [2, fc_count]
        fc_attr  = data.fc_attr   # [fc_count,4]
        mask = (data.candidate_nodes_label==1)  # 只对label=1的child进行回归
        if mask.any():
            father_idx    = fc_edges[0, mask]
            child_idx     = fc_edges[1, mask]
            fc_attr_sel   = fc_attr[mask]

            father_emb = node_emb[father_idx]    # [sum(mask), hidden_dim]
            child_emb  = node_emb[child_idx]     # [sum(mask), hidden_dim]
            edge_in    = torch.cat([father_emb, child_emb], dim=1)  # => [sum(mask), 2*hidden_dim]
            pred_edge_feat = model.edge_feature_head(edge_in)        # => [sum(mask),4]

            edge_feat_loss = F.mse_loss(pred_edge_feat, fc_attr_sel)

    # 3) 物理约束损失 (phy_loss)
    #    node_feats_pred => [N,2], 其中 0,1列可以表示 (V_real, V_imag) 
    #    如果你的模型输出不是 (V_real, V_imag)，请相应做转换
    phy_loss = power_flow_constraint(
        node_feats_pred,
        data.edge_index,
        data.edge_attr,
        data.candidate_nodes,
        data.candidate_nodes_label
    )

    total_loss = node_loss + lambda_edge*edge_feat_loss + lambda_phy*phy_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), node_loss.item(), edge_feat_loss.item(), phy_loss.item()

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

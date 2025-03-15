import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import os
from constraints import power_flow_constraint

def train_step(model, data, optimizer, lambda_phy=1.0):
    """
    训练单步:
      1) 前向传播 -> (node_probs, edge_probs, candidate_edges, node_feats_pred)
      2) 节点、边二分类损失 + 节点特征回归损失 + 物理约束损失
      3) 反向传播 & 更新
    """
    model.train()
    optimizer.zero_grad()

    # 模型输出: node_probs, edge_probs, candidate_edges, node_feats_pred
    node_probs, edge_probs, candidate_edges, node_feats_pred = model(
        data.x, data.edge_index, data.edge_attr, data.candidate_nodes, data.candidate_edges
    )

    # (a) 节点二分类损失
    node_loss = F.binary_cross_entropy(node_probs, data.candidate_nodes_label.float())
    # (b) 边二分类损失
    edge_loss = F.binary_cross_entropy(edge_probs, data.candidate_edges_label.float())

    # (c) 节点特征回归损失 (示例：对 candidate_nodes 的 (V_real, V_imag) 做 MSE)
    candidate_feats_true = data.x[data.candidate_nodes, :2]       # shape: [num_candidate_nodes, 2]
    candidate_feats_pred = node_feats_pred[data.candidate_nodes]  # shape: [num_candidate_nodes, 2]
    feature_loss = F.mse_loss(candidate_feats_pred, candidate_feats_true)

    # (d) 物理约束损失
    phy_loss = power_flow_constraint(
        data.V_real, data.V_imag,
        data.edge_index, data.known_S_real, data.known_S_imag
    )

    total_loss = node_loss + edge_loss + feature_loss + lambda_phy * phy_loss
    total_loss.backward()
    optimizer.step()

    return (
        total_loss.item(),
        node_loss.item(),
        edge_loss.item(),
        feature_loss.item(),
        phy_loss.item()
    )


def visualize_results(
    data, node_probs, edge_probs, candidate_edges,
    iteration, threshold=0.5, save_path='./results'
):
    """
    可视化:
      - 已知边(灰色)              : data.edge_index
      - 预测边(蓝色虚线)          : edge_probs >= threshold & label=1
      - 漏检边(橙色虚线)          : edge_probs < threshold & label=1  (真实存在却没预测到)
      - 父节点-子节点边(绿色实线) : candidate_source_nodes (若存在)
      - 节点:
          * lightblue: 已知节点
          * green    : 正确预测节点 (真实存在 & prob>=threshold)
          * red      : 错误预测节点 (真实不存在 & prob>=threshold)
          * orange   : 漏检节点 (真实存在 & prob<threshold)
    """
    G = nx.Graph()

    # 1) 基础节点分类
    known_nodes = data.node_known_mask.nonzero().view(-1).tolist()
    candidate_nodes = data.candidate_nodes.tolist()

    # 正确预测节点
    pred_exist_nodes = [
        candidate_nodes[i]
        for i, p in enumerate(node_probs)
        if p >= threshold and data.candidate_nodes_label[i] == 1
    ]
    # 错误预测节点
    false_nodes = [
        candidate_nodes[i]
        for i, p in enumerate(node_probs)
        if p >= threshold and data.candidate_nodes_label[i] == 0
    ]
    # 漏检节点 (真实存在但没被预测出来)
    missed_nodes = [
        candidate_nodes[i]
        for i, p in enumerate(node_probs)
        if p < threshold and data.candidate_nodes_label[i] == 1
    ]

    # 2) 基础边分类
    # 已知边(灰色)
    known_edges = data.edge_index.t().tolist()
    # 预测到的候选边(蓝色虚线)
    predicted_edges = [
        candidate_edges.t().tolist()[i]
        for i, p in enumerate(edge_probs)
        if p >= threshold and data.candidate_edges_label[i] == 1
    ]
    # 漏检边(真实存在但模型没预测)
    missed_edges = [
        candidate_edges.t().tolist()[i]
        for i, p in enumerate(edge_probs)
        if p < threshold and data.candidate_edges_label[i] == 1
    ]

    # 3) 添加节点进Graph
    G.add_nodes_from(known_nodes, type='known')
    G.add_nodes_from(pred_exist_nodes, type='correct')
    G.add_nodes_from(false_nodes, type='false')
    G.add_nodes_from(missed_nodes, type='missed')

    # 4) 添加边进Graph
    G.add_edges_from(known_edges, type='known')
    G.add_edges_from(predicted_edges, type='predicted')
    G.add_edges_from(missed_edges, type='missed')

    # 5) 父节点-子节点边(绿色实线)
    #    如果你的 dataset.py 中含有 candidate_source_nodes
    virtual_node_edges = []
    if hasattr(data, 'candidate_source_nodes'):
        candidate_source_nodes = data.candidate_source_nodes.tolist()
        node2source = dict(zip(candidate_nodes, candidate_source_nodes))

        # 仅给“真实存在并且被预测存在”的节点画出父节点连线
        for node in pred_exist_nodes:
            source_node = node2source[node]
            virtual_node_edges.append((source_node, node))

        G.add_edges_from(virtual_node_edges, type='virtual')

    # 6) 绘图
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))

    # (a) 已知边(灰色)
    nx.draw_networkx_edges(
        G, pos, edgelist=known_edges,
        edge_color='gray', alpha=0.5, label='Known Edges'
    )
    # (b) 预测到的边(蓝色虚线)
    nx.draw_networkx_edges(
        G, pos, edgelist=predicted_edges,
        edge_color='blue', style='dashed', alpha=0.7, label='Predicted Edges'
    )
    # (c) 漏检边(橙色虚线)
    nx.draw_networkx_edges(
        G, pos, edgelist=missed_edges,
        edge_color='orange', style='dotted', alpha=0.9, label='Missed True Edges'
    )
    # (d) 父节点-子节点(绿色实线)
    nx.draw_networkx_edges(
        G, pos, edgelist=virtual_node_edges,
        edge_color='green', style='solid', width=2, alpha=0.8, label='Parent-Child Edges'
    )

    # 节点
    nx.draw_networkx_nodes(
        G, pos, nodelist=known_nodes,
        node_color='lightblue', label='Known Nodes'
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=pred_exist_nodes,
        node_color='green', label='Correctly Predicted Nodes'
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=false_nodes,
        node_color='red', label='Incorrectly Predicted Nodes'
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=missed_nodes,
        node_color='orange', label='Missed True Nodes'
    )

    # 节点标签
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.legend()
    plt.title(f'Iteration {iteration}')
    plt.axis('off')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/iteration_{iteration}.png')
    plt.close()

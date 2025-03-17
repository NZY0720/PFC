import torch
import os
import pandas as pd

from dataset import PowerGridDataset
from model import VirtualNodePredictor
from utils import train_step, visualize_results

def main():
    # 1) 加载训练集
    train_dataset = PowerGridDataset(
        node_path='./train_data/NodeVoltages.csv',
        branch_path='./train_data/BranchFlows.csv',
        hide_ratio=0.2,
        is_train=True
    )
    train_data = train_dataset.load_data()

    # 2) 创建模型 & 优化器
    node_in_dim = train_data.x.size(1)  # 例如4
    edge_in_dim = train_data.edge_attr.size(1) if train_data.edge_attr is not None else 0
    model = VirtualNodePredictor(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=32,
        num_layers=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    # 3) 训练
    epochs = 3000
    for epoch in range(epochs):
        total_loss, node_loss, edge_loss, phy_loss = train_step(model, train_data, optimizer,
                                                               lambda_edge=1.0, 
                                                               lambda_phy=10.0)
        if epoch % 10 == 0:
            print(f'[Epoch {epoch}] total={total_loss:.4f}, node={node_loss:.4f}, phy={phy_loss:.4f}')

    # 4) 加载测试集
    test_dataset = PowerGridDataset(
        node_path='./test_data/NodeVoltages.csv',
        branch_path='./test_data/BranchFlows.csv',
        hide_ratio=0,
        is_train=False
    )
    test_data = test_dataset.load_data()

    # 5) 测试阶段: 推断 node_probs
    model.eval()
    with torch.no_grad():
        node_probs, _ ,_ = model(  
            test_data.x,
            test_data.edge_index,
            test_data.edge_attr,
            test_data.candidate_nodes
        )

    # 6) 保存 CSV，包括：NodeID, RealPart, ImagPart, Magnitude, Phase_deg, node_prob, (可选) label
    candidate_indices = test_data.candidate_nodes.cpu().numpy()       # [num_candidate_nodes]
    node_probs_np = node_probs.cpu().numpy()                          # [num_candidate_nodes]

    # 取原始特征 (形状: [N,4])，把它搬到CPU -> numpy
    x_original = test_data.x.cpu().numpy()                            # [N,4]

    # 逐个 candidate_node 找它对应的 4 维特征
    # 例如: RealPart=x_original[node_id,0], ImagPart=x_original[node_id,1], ...
    real_parts  = x_original[candidate_indices, 0]
    imag_parts  = x_original[candidate_indices, 1]
    magnitude   = x_original[candidate_indices, 2]
    phase_deg   = x_original[candidate_indices, 3]

    # 若有真实标签
    labels = None
    if hasattr(test_data, 'candidate_nodes_label'):
        labels = test_data.candidate_nodes_label.cpu().numpy()  # [num_candidate_nodes]

    # 构造DataFrame
    df_data = {
        'node_id'   : candidate_indices + 1
        'RealPart'  : real_parts,
        'ImagPart'  : imag_parts,
        'Magnitude' : magnitude,
        'Phase_deg' : phase_deg,
        'node_prob' : node_probs_np,
    }
    if labels is not None:
        df_data['label'] = labels

    df = pd.DataFrame(df_data)

    out_csv = './results/test_predictions.csv'
    df.to_csv(out_csv, index=False)
    print(f"[INFO] 已保存测试结果到 {out_csv}")

    # 7) 可视化
    visualize_results(
        test_data,
        node_probs=node_probs,
        node_feats_pred=None,
        iteration='test',
        threshold=0.5
    )

    print("Done.")

if __name__ == '__main__':
    os.makedirs('./results', exist_ok=True)
    main()

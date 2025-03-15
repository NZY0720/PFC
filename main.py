import torch, os
import pandas as pd
from dataset import PowerGridDataset
from model import VirtualNodePredictor
from utils import train_step, visualize_results

def main():
    # 1) 加载数据
    dataset = PowerGridDataset('./data/NodeVoltages.csv', './data/BranchFlows.csv')
    data = dataset.load_data()

    # 2) 初始化模型 & 优化器
    node_in_dim = data.x.size(1)      # 例如4: [RealPart, ImagPart, Magnitude, Phase_deg]
    edge_in_dim = data.edge_attr.size(1)  # 例如2: [S_Real, S_Imag]
    model = VirtualNodePredictor(node_in_dim, edge_in_dim, hidden_dim=64, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3) 训练超参
    epochs = 200
    iterations = 2
    lambda_phy = 10.0
    threshold = 0.5

    save_path = './results'
    os.makedirs(save_path, exist_ok=True)

    # 4) 训练循环
    for iteration in range(iterations):
        print(f'--- Iteration {iteration+1}/{iterations} ---')
        for epoch in range(epochs):
            (total_loss, node_loss, edge_loss, feature_loss, phy_loss) = train_step(model, data, optimizer, lambda_phy)
            print(f'[Iter {iteration+1} | Epoch {epoch}] '
                  f'total={total_loss:.4f} | node={node_loss:.4f} | edge={edge_loss:.4f} '
                  f'| feat={feature_loss:.4f} | phy={phy_loss:.4f}')

        # 5) 可视化
        model.eval()
        with torch.no_grad():
            node_probs, edge_probs, candidate_edges, node_feats_pred = model(
                data.x, data.edge_index, data.edge_attr, data.candidate_nodes, data.candidate_edges
            )

        visualize_results(
            data, node_probs, edge_probs, candidate_edges,
            iteration=iteration+1, threshold=threshold, save_path=save_path
        )

        # 6) 导出当前迭代下「预测节点特征 vs 真实节点特征」对比CSV
        candidate_nodes = data.candidate_nodes
        real_features = data.x[candidate_nodes, :2].cpu().numpy()      # (V_real, V_imag) 真实
        pred_features = node_feats_pred[candidate_nodes].cpu().numpy() # (V_real_pred, V_imag_pred)

        df = pd.DataFrame({
            'node_id': candidate_nodes.cpu().tolist(),
            'v_real_pred': pred_features[:, 0],
            'v_real_true': real_features[:, 0],
            'v_imag_pred': pred_features[:, 1],
            'v_imag_true': real_features[:, 1],
        })
        csv_path = os.path.join(save_path, f'feature_comparison_iter{iteration+1}.csv')
        df.to_csv(csv_path, index=False)
        print(f'==> Saved feature comparison CSV to: {csv_path}')


if __name__ == '__main__':
    main()

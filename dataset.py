import torch
from torch_geometric.data import Data
import pandas as pd

class PowerGridDataset:
    def __init__(self, node_path, branch_path, unknown_ratio=0.2):
        self.node_df = pd.read_csv(node_path).dropna().reset_index(drop=True)
        self.branch_df = pd.read_csv(branch_path).dropna().reset_index(drop=True)
        self.unknown_ratio = unknown_ratio

    def load_data(self):
        """
        加载节点电压和支路数据，构建已知节点、候选节点及对应的图结构。
        同时，添加 candidate_source_nodes 字段，用于记录每个候选节点的父节点(来源节点)。
        """
        # 节点特征: RealPart, ImagPart, Magnitude, Phase_deg
        node_features = torch.tensor(
            self.node_df[['RealPart', 'ImagPart', 'Magnitude', 'Phase_deg']].values, dtype=torch.float
        )
        num_nodes = len(self.node_df)

        # 构建所有边的索引和特征
        edge_index_all = torch.tensor(
            self.branch_df[['FromNode', 'ToNode']].values.T - 1, dtype=torch.long
        )
        edge_attr_all = torch.tensor(
            self.branch_df[['S_Real', 'S_Imag']].values, dtype=torch.float
        )

        # 已知节点: 前 (1 - unknown_ratio) * num_nodes
        known_node_num = int(num_nodes * (1 - self.unknown_ratio))
        known_nodes = torch.arange(known_node_num)
        candidate_nodes = torch.arange(known_node_num, num_nodes)

        # 已知节点掩码
        node_known_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_known_mask[known_nodes] = True

        # 候选节点的标签(这里设为1，表示这些节点为"潜在新节点")
        candidate_nodes_label = torch.ones(candidate_nodes.size(0), dtype=torch.long)

        # 过滤出已知边
        edge_mask = node_known_mask[edge_index_all[0]] & node_known_mask[edge_index_all[1]]
        edge_index_known = edge_index_all[:, edge_mask]
        edge_attr_known = edge_attr_all[edge_mask]

        known_S_real = edge_attr_known[:, 0]
        known_S_imag = edge_attr_known[:, 1]

        # 构建候选边: "所有candidate_nodes与known_nodes的组合"
        candidate_edges = torch.cartesian_prod(candidate_nodes, known_nodes).t()
        existing_edges_set = set(tuple(sorted(e)) for e in edge_index_all.t().tolist())

        # 候选边的标签(1表示此候选边在真实网络中已存在，0表示不存在)
        candidate_edges_label = torch.tensor([
            1 if tuple(sorted(edge)) in existing_edges_set else 0
            for edge in candidate_edges.t().tolist()
        ], dtype=torch.long)

        # 电压信息
        V_real = torch.tensor(self.node_df['RealPart'].values, dtype=torch.float)
        V_imag = torch.tensor(self.node_df['ImagPart'].values, dtype=torch.float)

        # "虚拟节点"的父节点: 这里简单示例使用随机父节点
        # 如果你有更明确的来源映射关系，请自行替换此处
        candidate_source_nodes = torch.randint(
            low=0,
            high=known_node_num,
            size=(candidate_nodes.size(0),),
            dtype=torch.long
        )

        data = Data(
            x=node_features,
            edge_index=edge_index_known,
            edge_attr=edge_attr_known,
            candidate_nodes=candidate_nodes,
            candidate_nodes_label=candidate_nodes_label,
            candidate_edges=candidate_edges,
            candidate_edges_label=candidate_edges_label,
            node_known_mask=node_known_mask,
            V_real=V_real,
            V_imag=V_imag,
            known_S_real=known_S_real,
            known_S_imag=known_S_imag,
            # 新增: 为每个候选节点指定一个父节点
            candidate_source_nodes=candidate_source_nodes
        )

        return data

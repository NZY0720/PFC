import torch
import pandas as pd
from torch_geometric.data import Data

class PowerGridDataset:
    def __init__(self, node_path, branch_path, hide_ratio=0.2, is_train=True):
        """
        node_path  : 节点CSV(含 RealPart,ImagPart,Magnitude,Phase_deg)
        branch_path: 边CSV(含 FromNode,ToNode, Vdiff_Real, Vdiff_Imag, S_Real, S_Imag 等)
        hide_ratio : 隐藏最低度节点比例
        is_train   : True => 训练集, 生成(真实+假)子节点; False => 测试集, 仅1个子节点
        """
        self.node_df = pd.read_csv(node_path).dropna().reset_index(drop=True)
        self.branch_df = pd.read_csv(branch_path).dropna().reset_index(drop=True)
        self.hide_ratio = hide_ratio
        self.is_train = is_train

        # 事先把原始 edges 放入字典方便查询
        # 例如 edge_map[(src, tgt)] = [Vdiff_Real, Vdiff_Imag, S_Real, S_Imag]
        self.edge_map = {}
        for i, row in self.branch_df.iterrows():
            src = int(row['FromNode']) - 1
            tgt = int(row['ToNode'])   - 1
            attr = [
                row['Vdiff_Real'],
                row['Vdiff_Imag'],
                row['S_Real'],
                row['S_Imag']
            ]
            self.edge_map[(src, tgt)] = attr
            self.edge_map[(tgt, src)] = attr  # 无向图，可双向查询

    def load_data(self):
        num_nodes = len(self.node_df)

        # 1) 过滤越界
        self.branch_df = self.branch_df[
            (self.branch_df['FromNode'] >=1) & (self.branch_df['FromNode']<=num_nodes) &
            (self.branch_df['ToNode']   >=1) & (self.branch_df['ToNode']  <=num_nodes)
        ].reset_index(drop=True)

        # 2) 节点特征
        father_feats = torch.tensor(
            self.node_df[['RealPart','ImagPart','Magnitude','Phase_deg']].values,
            dtype=torch.float
        )

        # 3) 原始 edges
        edge_index_all = torch.tensor(
            self.branch_df[['FromNode','ToNode']].values.T -1,
            dtype=torch.long
        )
        edge_attr_all = torch.tensor(
            self.branch_df[['Vdiff_Real','Vdiff_Imag','S_Real','S_Imag']].values,
            dtype=torch.float
        )

        # 4) 统计度数 => 隐藏低度数节点
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(edge_index_all.size(1)):
            s = edge_index_all[0,i]
            t = edge_index_all[1,i]
            degrees[s]+=1
            degrees[t]+=1

        sorted_nodes = torch.argsort(degrees)
        hide_count = int(num_nodes*self.hide_ratio)
        hidden_set = sorted_nodes[:hide_count].tolist()
        known_set  = sorted_nodes[hide_count:].tolist()

        hidden_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for h in hidden_set:
            hidden_mask[h] = True

        father_nodes = torch.tensor(known_set, dtype=torch.long)

        # 保留 father-father edges
        e_mask = (~hidden_mask[edge_index_all[0]]) & (~hidden_mask[edge_index_all[1]])
        edge_index_father = edge_index_all[:, e_mask]
        edge_attr_father  = edge_attr_all[e_mask]

        # adjacency
        adjacency = [[] for _ in range(num_nodes)]
        for i in range(edge_index_all.size(1)):
            s = edge_index_all[0,i].item()
            t = edge_index_all[1,i].item()
            adjacency[s].append(t)
            adjacency[t].append(s)

        father_feats_known = father_feats[father_nodes]
        Nf = father_nodes.size(0)

        ################################################################################
        # (A) 训练集 => 真实子节点(可label=1/0) + 假子节点(label=0)
        ################################################################################
        if self.is_train:
            # 1) 生成真实子标签
            real_labels = []
            for i,f in enumerate(father_nodes):
                has_hidden = any(n in hidden_set for n in adjacency[f.item()])
                real_labels.append(1 if has_hidden else 0)
            real_labels = torch.tensor(real_labels, dtype=torch.long)

            # 2) 假子 => label=0
            child_feats_real = father_feats_known.clone()
            child_feats_fake = torch.zeros_like(father_feats_known)
            fake_labels = torch.zeros(Nf, dtype=torch.long)

            # 拼接节点: father + child_real + child_fake
            node_features = torch.cat([father_feats_known, child_feats_real, child_feats_fake], dim=0)
            total_num_nodes = 3*Nf

            candidate_real_nodes = torch.arange(Nf, 2*Nf)
            candidate_fake_nodes = torch.arange(2*Nf, 3*Nf)
            candidate_nodes = torch.cat([candidate_real_nodes, candidate_fake_nodes], dim=0)
            candidate_nodes_label = torch.cat([real_labels, fake_labels], dim=0)

            # father->child-real/fake edges
            fc_edges_real = []
            fc_attr_real  = []

            # ⚠ 关键修改：从 CSV 查询 father→(某 hidden neighbor) 的特征，如果不存在就填0
            for i in range(Nf):
                father_new_idx = i              # 在 "新图" 中 father 的编号
                child_new_idx  = Nf + i         # 在 "新图" 中 child 的编号
                fc_edges_real.append((father_new_idx, child_new_idx))

                # 判断父节点在原图中的下标
                father_original_idx = father_nodes[i].item()

                if real_labels[i] == 1:
                    # father 有 hidden 邻居 => 找到一个 hidden 邻居
                    hidden_neighbor = None
                    for nbr in adjacency[father_original_idx]:
                        if nbr in hidden_set:
                            hidden_neighbor = nbr
                            break
                    if hidden_neighbor is not None:
                        # 查字典 edge_map[(father_original_idx, hidden_neighbor)]
                        if (father_original_idx, hidden_neighbor) in self.edge_map:
                            fc_attr_real.append(self.edge_map[(father_original_idx, hidden_neighbor)])
                            continue

                # 如果走到这里，说明要么 real_labels[i]==0，要么 CSV 没找到
                fc_attr_real.append([0,0,0,0])

            fc_edges_real = torch.tensor(fc_edges_real, dtype=torch.long).T
            fc_attr_real  = torch.tensor(fc_attr_real,  dtype=torch.float)

            # 假子节点就直接挂0
            fc_edges_fake = []
            fc_attr_fake  = []
            for i in range(Nf):
                fc_edges_fake.append((i, 2*Nf + i))
                fc_attr_fake.append([0,0,0,0])
            fc_edges_fake = torch.tensor(fc_edges_fake, dtype=torch.long).T
            fc_attr_fake  = torch.tensor(fc_attr_fake,  dtype=torch.float)

            # father->father => old2new
            old2new = {}
            for i,f in enumerate(father_nodes):
                old2new[f.item()] = i

            father_edges_mapped = []
            father_attrs_mapped = []
            for i in range(edge_index_father.size(1)):
                s_old = edge_index_father[0,i].item()
                t_old = edge_index_father[1,i].item()
                s_new = old2new[s_old]
                t_new = old2new[t_old]
                father_edges_mapped.append((s_new, t_new))
                father_attrs_mapped.append(edge_attr_father[i].tolist())
            father_edges_mapped = torch.tensor(father_edges_mapped, dtype=torch.long).T
            father_attrs_mapped = torch.tensor(father_attrs_mapped, dtype=torch.float)

            # 合并 edge
            new_edge_index = torch.cat([father_edges_mapped, fc_edges_real, fc_edges_fake], dim=1)
            new_edge_attr  = torch.cat([father_attrs_mapped, fc_attr_real, fc_attr_fake], dim=0)

            # node_known_mask
            node_known_mask = torch.zeros(total_num_nodes, dtype=torch.bool)
            node_known_mask[:Nf] = True

            # 准备 V_real, V_imag （如果后面要用）
            father_real_vec = self.node_df['RealPart'].values[father_nodes]
            father_imag_vec = self.node_df['ImagPart'].values[father_nodes]
            child_real_vec  = father_real_vec.copy()
            child_imag_vec  = father_imag_vec.copy()
            zero_vec        = torch.zeros_like(torch.tensor(father_real_vec))

            V_real = torch.cat([
                torch.tensor(father_real_vec),
                torch.tensor(child_real_vec),
                zero_vec
            ], dim=0).float()

            V_imag = torch.cat([
                torch.tensor(father_imag_vec),
                torch.tensor(child_imag_vec),
                zero_vec
            ], dim=0).float()

            known_S_real = new_edge_attr[:,2]
            known_S_imag = new_edge_attr[:,3]

            # father->child edges 合并
            fc_edges = torch.cat([fc_edges_real, fc_edges_fake], dim=1) 
            fc_attr  = torch.cat([fc_attr_real, fc_attr_fake], dim=0)

            data = Data(
                x=node_features,
                edge_index=new_edge_index,
                edge_attr=new_edge_attr,
                candidate_nodes=candidate_nodes,
                candidate_nodes_label=candidate_nodes_label,
                node_known_mask=node_known_mask,
                V_real=V_real,
                V_imag=V_imag,
                known_S_real=known_S_real,
                known_S_imag=known_S_imag,
                fc_edges=fc_edges,
                fc_attr=fc_attr
            )
            return data

        else:
            # (B) 测试/验证 => 仅生成1个子节点 => label=1/0 => father->child => fc_attr

            father_feats_known = father_feats[father_nodes]
            Nf = father_nodes.size(0)

            # 仅1个子节点
            child_feats = father_feats_known.clone()
            label_list = []
            for i,f in enumerate(father_nodes):
                has_hidden = any(n in hidden_set for n in adjacency[f.item()])
                label_list.append(1 if has_hidden else 0)
            label_list = torch.tensor(label_list, dtype=torch.long)

            node_features = torch.cat([father_feats_known, child_feats], dim=0)
            total_num_nodes = 2*Nf

            candidate_nodes = torch.arange(Nf,2*Nf)
            candidate_nodes_label = label_list

            # father->child edges
            fc_edges = []
            fc_attr  = []
            for i in range(Nf):
                fc_edges.append((i, Nf+i))
                if label_list[i]==1:
                    fc_attr.append([0.001,0.002,0.1,0.05]) # or real data
                else:
                    fc_attr.append([0,0,0,0])

            fc_edges = torch.tensor(fc_edges, dtype=torch.long).T
            fc_attr  = torch.tensor(fc_attr, dtype=torch.float)

            # father->father => old2new
            old2new={}
            for i,f in enumerate(father_nodes):
                old2new[f.item()] = i
            father_edges_mapped=[]
            father_attrs_mapped=[]
            for i in range(edge_index_father.size(1)):
                s_old=edge_index_father[0,i].item()
                t_old=edge_index_father[1,i].item()
                s_new=old2new[s_old]
                t_new=old2new[t_old]
                father_edges_mapped.append((s_new,t_new))
                father_attrs_mapped.append(edge_attr_father[i].tolist())
            father_edges_mapped=torch.tensor(father_edges_mapped,dtype=torch.long).T
            father_attrs_mapped=torch.tensor(father_attrs_mapped,dtype=torch.float)

            new_edge_index=torch.cat([father_edges_mapped, fc_edges],dim=1)
            new_edge_attr=torch.cat([father_attrs_mapped, fc_attr],dim=0)

            node_known_mask=torch.zeros(total_num_nodes,dtype=torch.bool)
            node_known_mask[:Nf]=True

            father_real_vec=self.node_df['RealPart'].values[father_nodes]
            father_imag_vec=self.node_df['ImagPart'].values[father_nodes]
            child_real_vec=father_real_vec.copy()
            child_imag_vec=father_imag_vec.copy()

            V_real=torch.cat([
                torch.tensor(father_real_vec), 
                torch.tensor(child_real_vec)
            ],dim=0).float()
            V_imag=torch.cat([
                torch.tensor(father_imag_vec),
                torch.tensor(child_imag_vec)
            ],dim=0).float()

            known_S_real=new_edge_attr[:,2]
            known_S_imag=new_edge_attr[:,3]

            data=Data(
                x=node_features,
                edge_index=new_edge_index,
                edge_attr=new_edge_attr,
                candidate_nodes=candidate_nodes,
                candidate_nodes_label=candidate_nodes_label,
                node_known_mask=node_known_mask,
                V_real=V_real,
                V_imag=V_imag,
                known_S_real=known_S_real,
                known_S_imag=known_S_imag,

                fc_edges=fc_edges,
                fc_attr=fc_attr
            )
            return data

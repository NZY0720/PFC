import torch
import pandas as pd
from torch_geometric.data import Data

class PowerGridDataset:
    def __init__(self, node_path, branch_path, hide_ratio=0.2, is_train=True):
        """
        node_path  : 节点CSV (含 RealPart,ImagPart等)
        branch_path: 边CSV   (含 FromNode,ToNode,S_Real,S_Imag)
        hide_ratio : 手动隐藏节点比例(最低度)
        is_train   : True=训练集, False=测试集(逻辑可略不同 if needed)
        """
        self.node_df = pd.read_csv(node_path).dropna().reset_index(drop=True)
        self.branch_df = pd.read_csv(branch_path).dropna().reset_index(drop=True)
        self.hide_ratio = hide_ratio
        self.is_train = is_train

    def load_data(self):
        """
        1) 读原始节点/边
        2) 选度最小 hide_ratio => hidden_set
        3) 余者 => father_nodes
        4) father_nodes each => child_node => label=1 if father has hidden neighbor
        5) 合并 father-father + father-child into new_edge_index
        6) No candidate_edges => only father-child edges in edge_index
        7) Return Data(...), with candidate_nodes & candidate_nodes_label
        """
        num_nodes = len(self.node_df)
        self.branch_df = self.branch_df[
            (self.branch_df['FromNode'] >=1) & (self.branch_df['FromNode'] <=num_nodes) &
            (self.branch_df['ToNode']   >=1) & (self.branch_df['ToNode']   <=num_nodes)
        ].reset_index(drop=True)

        # father feats
        father_feats = torch.tensor(
            self.node_df[['RealPart','ImagPart','Magnitude','Phase_deg']].values,
            dtype=torch.float
        )

        # edges => [2,E]
        edge_index_all = torch.tensor(
            self.branch_df[['FromNode','ToNode']].values.T -1,
            dtype=torch.long
        )
        edge_attr_all = torch.tensor(
            self.branch_df[['S_Real','S_Imag']].values,
            dtype=torch.float
        )

        # compute degrees
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(edge_index_all.size(1)):
            s = edge_index_all[0,i]
            t = edge_index_all[1,i]
            degrees[s]+=1
            degrees[t]+=1

        sorted_nodes = torch.argsort(degrees)
        hide_count = int(num_nodes * self.hide_ratio)
        hidden_set = sorted_nodes[:hide_count].tolist()  # H
        known_set  = sorted_nodes[hide_count:].tolist()  # K

        hidden_mask = torch.zeros(num_nodes, dtype=torch.bool)
        hidden_mask[hidden_set] = True

        # father_nodes => known
        father_nodes = torch.tensor(known_set, dtype=torch.long)

        # keep father-father edges only
        e_mask = (~hidden_mask[edge_index_all[0]]) & (~hidden_mask[edge_index_all[1]])
        edge_index_father = edge_index_all[:, e_mask]
        edge_attr_father  = edge_attr_all[e_mask]

        # create child_nodes => father_nodes.size(0)
        Nf = father_nodes.size(0)
        father_feats_known = father_feats[father_nodes]  # shape=[Nf,4]
        child_feats = father_feats_known.clone()
        node_features = torch.cat([father_feats_known, child_feats], dim=0)  # shape=[2Nf,4]

        # label => if father has neighbor in hidden_set => child=1
        adjacency = [[] for _ in range(num_nodes)]
        for i in range(edge_index_all.size(1)):
            s = edge_index_all[0,i].item()
            t = edge_index_all[1,i].item()
            adjacency[s].append(t)
            adjacency[t].append(s)

        candidate_nodes_label = []
        for i, f in enumerate(father_nodes):
            neighs = adjacency[f.item()]
            has_hidden = any(n in hidden_set for n in neighs)
            candidate_nodes_label.append(1 if has_hidden else 0)
        candidate_nodes_label = torch.tensor(candidate_nodes_label, dtype=torch.long)

        # father-child edges
        fc_edges = []
        for i in range(Nf):
            fc_edges.append((i, Nf+i))
        fc_edges = torch.tensor(fc_edges, dtype=torch.long).T  # shape=[2,Nf]
        fc_edge_attr = torch.zeros(Nf, 2, dtype=torch.float)

        # reindex father-father edges => father old-> new [0..Nf-1]
        old2new = {}
        for i, f in enumerate(father_nodes):
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

        # merge father-father + father-child
        new_edge_index = torch.cat([father_edges_mapped, fc_edges], dim=1)
        new_edge_attr  = torch.cat([father_attrs_mapped, fc_edge_attr], dim=0)

        # node_known_mask => [Nf..2Nf-1] = child
        new_num_nodes = 2*Nf
        node_known_mask = torch.zeros(new_num_nodes, dtype=torch.bool)
        node_known_mask[:Nf] = True

        candidate_nodes = torch.arange(Nf, 2*Nf)  # child
        # build V_real, V_imag
        father_real = self.node_df['RealPart'].values[father_nodes]
        father_imag = self.node_df['ImagPart'].values[father_nodes]
        child_real = father_real.copy()
        child_imag = father_imag.copy()
        V_real = torch.tensor(list(father_real)+list(child_real), dtype=torch.float)
        V_imag = torch.tensor(list(father_imag)+list(child_imag), dtype=torch.float)

        known_S_real = new_edge_attr[:,0]
        known_S_imag = new_edge_attr[:,1]

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
            known_S_imag=known_S_imag
        )
        return data

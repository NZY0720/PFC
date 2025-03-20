import torch
import pandas as pd
import numpy as np
import re
import os
import glob
from torch_geometric.data import Data
from torch.utils.data import Dataset

class TemporalPowerGridDataset(Dataset):
    def __init__(self, data_dir, hide_ratio=0.2, is_train=True, max_time_steps=24, sequence_length=1):
        """
        Temporal dataset for power grid data
        
        Args:
            data_dir: Directory containing Imeas_data_*.csv and Vmeas_data_*.csv files
            hide_ratio: Percentage of lowest-degree nodes to hide
            is_train: If True, generate training data with real/fake child nodes; otherwise testing data
            max_time_steps: Maximum number of time steps in the dataset
            sequence_length: Number of consecutive time steps to include in each sample
        """
        super().__init__()
        self.data_dir = data_dir
        self.hide_ratio = hide_ratio
        self.is_train = is_train
        self.max_time_steps = max_time_steps
        self.sequence_length = sequence_length
        
        # Find all data files
        self.i_data_paths = sorted(glob.glob(os.path.join(data_dir, 'Imeas_data_*.csv')))
        self.v_data_paths = sorted(glob.glob(os.path.join(data_dir, 'Vmeas_data_*.csv')))
        
        if not self.i_data_paths or not self.v_data_paths:
            raise ValueError(f"No data files found in {data_dir}")
        
        # Extract time steps from file names
        self.time_steps = self._extract_time_steps()
        print(f"Found {len(self.time_steps)} time steps in {data_dir}")
        
        # Create node and edge maps
        self._initialize_dataset()
        
        # Create valid sequence indices
        self.valid_indices = list(range(len(self.time_steps) - sequence_length + 1))
        
        # Cache for loaded data
        self.data_cache = {}

    def _extract_time_steps(self):
        """Extract time step indices from file names"""
        time_steps = []
        for path in self.i_data_paths:
            # Extract number from filename (e.g., Imeas_data_5.csv -> 5)
            match = re.search(r'Imeas_data_(\d+)\.csv', os.path.basename(path))
            if match:
                time_steps.append(int(match.group(1)))
        return sorted(time_steps)

    def _parse_complex(self, complex_str):
        """Parse complex numbers from string format like '1.03-0.01i'"""
        pattern = r'([-+]?\d*\.?\d+(?:e[-+]?\d+)?)([-+]\d*\.?\d+(?:e[-+]?\d+)?i)'
        match = re.match(pattern, complex_str)
        if match:
            real = float(match.group(1))
            imag = float(match.group(2).replace('i', ''))
            return real, imag
        return 0, 0  # Default if parsing fails

    def _initialize_dataset(self):
        """Initialize dataset by loading the first time step to get node and branch info"""
        self.node_ids = set()
        self.branch_ids = set()
        
        # Load first time step to get structure
        if self.time_steps:
            t = self.time_steps[0]
            i_path = os.path.join(self.data_dir, f'Imeas_data_{t}.csv')
            v_path = os.path.join(self.data_dir, f'Vmeas_data_{t}.csv')
            
            if os.path.exists(i_path) and os.path.exists(v_path):
                i_df = pd.read_csv(i_path)
                v_df = pd.read_csv(v_path)
                
                # Collect all node IDs
                for node_id in v_df['Node_ID']:
                    self.node_ids.add(node_id)
                
                # Collect all branch IDs
                for branch_id in i_df['Branch_ID']:
                    self.branch_ids.add(branch_id)
                    
                # Extract node pairs from branch IDs
                self.branch_to_nodes = {}
                for branch_id in self.branch_ids:
                    from_node, to_node = map(int, branch_id.split('_'))
                    self.branch_to_nodes[branch_id] = (from_node, to_node)
                
                # Create mapping from node ID to index
                self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(list(self.node_ids)))}
                
                # Build adjacency information
                self.adjacency = {node_id: [] for node_id in self.node_ids}
                for branch_id, (from_node, to_node) in self.branch_to_nodes.items():
                    self.adjacency[from_node].append(to_node)
                    self.adjacency[to_node].append(from_node)
                
                print(f"Initialized dataset with {len(self.node_ids)} nodes and {len(self.branch_ids)} branches")
            else:
                raise FileNotFoundError(f"Could not find data files for time step {t}")

    def _load_time_step_data(self, time_step):
        """Load data for a specific time step"""
        i_path = os.path.join(self.data_dir, f'Imeas_data_{time_step}.csv')
        v_path = os.path.join(self.data_dir, f'Vmeas_data_{time_step}.csv')
        
        if not os.path.exists(i_path) or not os.path.exists(v_path):
            raise FileNotFoundError(f"Missing data for time step {time_step}")
        
        i_df = pd.read_csv(i_path)
        v_df = pd.read_csv(v_path)
        
        # Process voltage data
        node_features = {}
        for _, row in v_df.iterrows():
            node_id = row['Node_ID']
            if node_id not in self.node_ids:
                continue
                
            # Extract complex components for each phase
            v_real_a, v_imag_a = self._parse_complex(row['V_A'])
            
            # Use magnitude and phase as additional features
            magnitude = np.sqrt(v_real_a**2 + v_imag_a**2)
            phase_deg = np.degrees(np.arctan2(v_imag_a, v_real_a))
            
            # Store features
            node_features[node_id] = [v_real_a, v_imag_a, magnitude, phase_deg]
        
        # Process current data
        edge_features = {}
        for _, row in i_df.iterrows():
            branch_id = row['Branch_ID']
            if branch_id not in self.branch_ids:
                continue
                
            from_node, to_node = self.branch_to_nodes[branch_id]
            
            # Parse complex current
            i_real_a, i_imag_a = self._parse_complex(row['I_A'])
            
            # Get voltage values for the endpoints
            if from_node in node_features and to_node in node_features:
                from_v_real = node_features[from_node][0]
                from_v_imag = node_features[from_node][1]
                to_v_real = node_features[to_node][0]
                to_v_imag = node_features[to_node][1]
                
                # Calculate voltage difference
                vdiff_real = from_v_real - to_v_real
                vdiff_imag = from_v_imag - to_v_imag
                
                # Calculate complex power S = V * I*
                s_real = from_v_real * i_real_a + from_v_imag * i_imag_a
                s_imag = from_v_imag * i_real_a - from_v_real * i_imag_a
                
                # Store edge features [Vdiff_Real, Vdiff_Imag, S_Real, S_Imag]
                edge_features[branch_id] = [vdiff_real, vdiff_imag, s_real, s_imag]
            
        return node_features, edge_features

    def _prepare_pyg_data(self, time_step, node_features, edge_features, hidden_nodes=None):
        """
        Prepare PyTorch Geometric data object for a specific time step
        
        Args:
            time_step: Time step index
            node_features: Dictionary mapping node ID to features
            edge_features: Dictionary mapping branch ID to features
            hidden_nodes: Set of node IDs to hide (if None, will be computed)
            
        Returns:
            Data object for PyTorch Geometric
        """
        num_nodes = len(self.node_ids)
        
        # Convert node features to tensor
        x = torch.zeros((num_nodes, 4), dtype=torch.float)
        for node_id, features in node_features.items():
            if node_id in self.node_id_to_idx:
                idx = self.node_id_to_idx[node_id]
                x[idx] = torch.tensor(features, dtype=torch.float)
        
        # Prepare edges
        edge_list = []
        edge_attr_list = []
        
        for branch_id, (from_node, to_node) in self.branch_to_nodes.items():
            if branch_id in edge_features:
                # Convert node IDs to indices
                if from_node in self.node_id_to_idx and to_node in self.node_id_to_idx:
                    from_idx = self.node_id_to_idx[from_node]
                    to_idx = self.node_id_to_idx[to_node]
                    
                    # Add edge in both directions (undirected graph)
                    edge_list.append((from_idx, to_idx))
                    edge_list.append((to_idx, from_idx))
                    
                    # Add edge attributes for both directions
                    edge_attr_list.append(edge_features[branch_id])
                    edge_attr_list.append(edge_features[branch_id])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float) if edge_attr_list else torch.zeros((0, 4), dtype=torch.float)
        
        # Determine nodes to hide if not provided
        if hidden_nodes is None:
            # Calculate node degrees
            degrees = {node_id: len(neighbors) for node_id, neighbors in self.adjacency.items()}
            
            # Sort nodes by degree
            sorted_nodes = sorted(degrees.keys(), key=lambda n: degrees[n])
            
            # Hide the specified ratio of lowest-degree nodes
            num_to_hide = int(len(sorted_nodes) * self.hide_ratio)
            hidden_nodes = set(sorted_nodes[:num_to_hide])
        
        # Create masks for known and hidden nodes
        known_nodes = set(self.node_ids) - hidden_nodes
        
        # Convert node IDs to indices
        known_indices = [self.node_id_to_idx[n] for n in known_nodes if n in self.node_id_to_idx]
        hidden_indices = [self.node_id_to_idx[n] for n in hidden_nodes if n in self.node_id_to_idx]
        
        father_nodes = torch.tensor(known_indices, dtype=torch.long)
        
        # Create time index for each node
        time_index = torch.full((num_nodes,), time_step, dtype=torch.long)
        
        if self.is_train:
            # Features for known nodes
            father_feats = x[father_nodes]
            Nf = father_nodes.size(0)
            
            # Generate labels for candidate nodes
            real_labels = []
            for i, idx in enumerate(known_indices):
                node_id = list(sorted(self.node_ids))[idx]
                # Check if this node has any hidden neighbors
                has_hidden_neighbor = any(neighbor in hidden_nodes for neighbor in self.adjacency[node_id])
                real_labels.append(1 if has_hidden_neighbor else 0)
            real_labels = torch.tensor(real_labels, dtype=torch.long)
            
            # Create features for real and fake children
            child_feats_real = father_feats.clone()
            child_feats_fake = torch.zeros_like(father_feats)
            fake_labels = torch.zeros(Nf, dtype=torch.long)
            
            # Combine features for all nodes
            node_features_combined = torch.cat([father_feats, child_feats_real, child_feats_fake], dim=0)
            total_num_nodes = 3 * Nf
            
            # Create indices for candidate nodes
            candidate_real_nodes = torch.arange(Nf, 2 * Nf)
            candidate_fake_nodes = torch.arange(2 * Nf, 3 * Nf)
            candidate_nodes = torch.cat([candidate_real_nodes, candidate_fake_nodes], dim=0)
            candidate_nodes_label = torch.cat([real_labels, fake_labels], dim=0)
            
            # Create time index for all nodes (including candidates)
            time_index_combined = torch.full((total_num_nodes,), time_step, dtype=torch.long)
            
            # Create father-child edges
            fc_edges_real = []
            fc_attr_real = []
            
            for i in range(Nf):
                father_new_idx = i
                child_new_idx = Nf + i
                fc_edges_real.append((father_new_idx, child_new_idx))
                
                if real_labels[i] == 1:
                    # This father has a hidden neighbor
                    father_node_id = list(sorted(self.node_ids))[known_indices[i]]
                    
                    # Find a hidden neighbor
                    hidden_neighbor = None
                    for neighbor in self.adjacency[father_node_id]:
                        if neighbor in hidden_nodes:
                            hidden_neighbor = neighbor
                            break
                    
                    if hidden_neighbor is not None:
                        # Find the branch connecting these nodes
                        branch_key = f"{father_node_id}_{hidden_neighbor}"
                        alt_branch_key = f"{hidden_neighbor}_{father_node_id}"
                        
                        if branch_key in edge_features:
                            fc_attr_real.append(edge_features[branch_key])
                            continue
                        elif alt_branch_key in edge_features:
                            fc_attr_real.append(edge_features[alt_branch_key])
                            continue
                
                # Default zero attributes if no matching edge found
                fc_attr_real.append([0, 0, 0, 0])
            
            fc_edges_real = torch.tensor(fc_edges_real, dtype=torch.long).T
            fc_attr_real = torch.tensor(fc_attr_real, dtype=torch.float)
            
            # Create fake child edges
            fc_edges_fake = []
            fc_attr_fake = []
            for i in range(Nf):
                fc_edges_fake.append((i, 2 * Nf + i))
                fc_attr_fake.append([0, 0, 0, 0])
            
            fc_edges_fake = torch.tensor(fc_edges_fake, dtype=torch.long).T
            fc_attr_fake = torch.tensor(fc_attr_fake, dtype=torch.float)
            
            # Filter original edges to keep only father-father edges
            father_edges = []
            father_attrs = []
            
            # Create a mapping from old to new indices for father nodes
            old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(known_indices)}
            
            for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1])):
                s_item, t_item = s.item(), t.item()
                if s_item in old2new and t_item in old2new:
                    father_edges.append((old2new[s_item], old2new[t_item]))
                    father_attrs.append(edge_attr[i].tolist())
            
            father_edges = torch.tensor(father_edges, dtype=torch.long).T if father_edges else torch.zeros((2, 0), dtype=torch.long)
            father_attrs = torch.tensor(father_attrs, dtype=torch.float) if father_attrs else torch.zeros((0, 4), dtype=torch.float)
            
            # Combine all edges
            new_edge_index = torch.cat([father_edges, fc_edges_real, fc_edges_fake], dim=1) if father_edges.size(1) > 0 else torch.cat([fc_edges_real, fc_edges_fake], dim=1)
            new_edge_attr = torch.cat([father_attrs, fc_attr_real, fc_attr_fake], dim=0) if father_attrs.size(0) > 0 else torch.cat([fc_attr_real, fc_attr_fake], dim=0)
            
            # Create mask for known nodes
            node_known_mask = torch.zeros(total_num_nodes, dtype=torch.bool)
            node_known_mask[:Nf] = True
            
            # Extract voltage components
            V_real = node_features_combined[:, 0]
            V_imag = node_features_combined[:, 1]
            
            # Extract power components from edge attributes
            known_S_real = new_edge_attr[:, 2]
            known_S_imag = new_edge_attr[:, 3]
            
            # Combine father-child edges
            fc_edges = torch.cat([fc_edges_real, fc_edges_fake], dim=1)
            fc_attr = torch.cat([fc_attr_real, fc_attr_fake], dim=0)
            
            # Create PyG data object
            data = Data(
                x=node_features_combined,
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
                fc_attr=fc_attr,
                time_index=time_index_combined,
                time_step=torch.tensor([time_step], dtype=torch.long)
            )
            
            return data, hidden_nodes
            
        else:
            # Testing data with a single child node per father
            father_feats = x[father_nodes]
            Nf = father_nodes.size(0)
            
            # Clone father features for child
            child_feats = father_feats.clone()
            
            # Generate labels
            label_list = []
            for i, idx in enumerate(known_indices):
                node_id = list(sorted(self.node_ids))[idx]
                # Check if this node has any hidden neighbors
                has_hidden_neighbor = any(neighbor in hidden_nodes for neighbor in self.adjacency[node_id])
                label_list.append(1 if has_hidden_neighbor else 0)
            label_list = torch.tensor(label_list, dtype=torch.long)
            
            # Combine features
            node_features_combined = torch.cat([father_feats, child_feats], dim=0)
            total_num_nodes = 2 * Nf
            
            # Create candidate nodes
            candidate_nodes = torch.arange(Nf, 2 * Nf)
            candidate_nodes_label = label_list
            
            # Create time index for all nodes
            time_index_combined = torch.full((total_num_nodes,), time_step, dtype=torch.long)
            
            # Create father-child edges
            fc_edges = []
            fc_attr = []
            
            for i in range(Nf):
                fc_edges.append((i, Nf + i))
                
                if label_list[i] == 1:
                    # This father has a hidden neighbor
                    father_node_id = list(sorted(self.node_ids))[known_indices[i]]
                    
                    # Find a hidden neighbor
                    found_edge = False
                    for neighbor in self.adjacency[father_node_id]:
                        if neighbor in hidden_nodes:
                            # Find the branch connecting these nodes
                            branch_key = f"{father_node_id}_{neighbor}"
                            alt_branch_key = f"{neighbor}_{father_node_id}"
                            
                            if branch_key in edge_features:
                                fc_attr.append(edge_features[branch_key])
                                found_edge = True
                                break
                            elif alt_branch_key in edge_features:
                                fc_attr.append(edge_features[alt_branch_key])
                                found_edge = True
                                break
                    
                    if not found_edge:
                        # No matching edge found
                        fc_attr.append([0.001, 0.002, 0.1, 0.05])
                else:
                    fc_attr.append([0, 0, 0, 0])
            
            fc_edges = torch.tensor(fc_edges, dtype=torch.long).T
            fc_attr = torch.tensor(fc_attr, dtype=torch.float)
            
            # Filter original edges to keep only father-father edges
            father_edges = []
            father_attrs = []
            
            # Create a mapping from old to new indices for father nodes
            old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(known_indices)}
            
            for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1])):
                s_item, t_item = s.item(), t.item()
                if s_item in old2new and t_item in old2new:
                    father_edges.append((old2new[s_item], old2new[t_item]))
                    father_attrs.append(edge_attr[i].tolist())
            
            father_edges = torch.tensor(father_edges, dtype=torch.long).T if father_edges else torch.zeros((2, 0), dtype=torch.long)
            father_attrs = torch.tensor(father_attrs, dtype=torch.float) if father_attrs else torch.zeros((0, 4), dtype=torch.float)
            
            # Combine all edges
            new_edge_index = torch.cat([father_edges, fc_edges], dim=1) if father_edges.size(1) > 0 else fc_edges
            new_edge_attr = torch.cat([father_attrs, fc_attr], dim=0) if father_attrs.size(0) > 0 else fc_attr
            
            # Create mask for known nodes
            node_known_mask = torch.zeros(total_num_nodes, dtype=torch.bool)
            node_known_mask[:Nf] = True
            
            # Extract voltage components
            V_real = node_features_combined[:, 0]
            V_imag = node_features_combined[:, 1]
            
            # Extract power components from edge attributes
            known_S_real = new_edge_attr[:, 2]
            known_S_imag = new_edge_attr[:, 3]
            
            # Create PyG data object
            data = Data(
                x=node_features_combined,
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
                fc_attr=fc_attr,
                time_index=time_index_combined,
                time_step=torch.tensor([time_step], dtype=torch.long)
            )
            
            return data, hidden_nodes

    def __len__(self):
        """Return the number of valid sequences in the dataset"""
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        """
        Get a sequence of data starting at the specified index
        
        Args:
            idx: Index of the sequence
            
        Returns:
            List of Data objects representing a sequence of time steps
        """
        if idx in self.data_cache:
            return self.data_cache[idx]
            
        start_idx = self.valid_indices[idx]
        sequence = []
        hidden_nodes = None
        
        for i in range(self.sequence_length):
            if start_idx + i >= len(self.time_steps):
                break
                
            time_step = self.time_steps[start_idx + i]
            
            # Load data for this time step
            node_features, edge_features = self._load_time_step_data(time_step)
            
            # For the first time step, we need to determine hidden nodes
            # For subsequent time steps, use the same hidden nodes for consistency
            data, hidden_nodes = self._prepare_pyg_data(time_step, node_features, edge_features, hidden_nodes)
            sequence.append(data)
        
        # Cache the result
        self.data_cache[idx] = sequence
        return sequence

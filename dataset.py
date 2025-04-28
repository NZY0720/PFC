import torch
import pandas as pd
import numpy as np
import re
import os
import glob
from torch_geometric.data import Data
from torch.utils.data import Dataset

class TemporalPowerGridDataset(Dataset):
    def __init__(self, data_dir, hide_ratio=0.2, is_train=True, max_time_steps=1440, sequence_length=1, hourly=False):
        """
        Temporal dataset for power grid data with the new format
        
        Args:
            data_dir: Directory containing time-based CSV files (HHMM_line_currents.csv and HHMM_node_voltages.csv)
            hide_ratio: Percentage of lowest-degree nodes to hide
            is_train: If True, generate training data with real/fake child nodes; otherwise testing data
            max_time_steps: Maximum number of time steps in the dataset (default: 1440 for 24hrs in minutes)
            sequence_length: Number of consecutive time steps in each sample
            hourly: If True, only use data from hourly intervals (HH00 files)
        """
        super().__init__()
        self.data_dir = data_dir
        self.hide_ratio = hide_ratio
        self.is_train = is_train
        self.max_time_steps = max_time_steps
        self.sequence_length = sequence_length
        self.hourly = hourly
        
        # Find all data files
        self.line_current_paths = sorted(glob.glob(os.path.join(data_dir, '????_line_currents.csv')))
        self.node_voltage_paths = sorted(glob.glob(os.path.join(data_dir, '????_node_voltages.csv')))
        
        if not self.line_current_paths or not self.node_voltage_paths:
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
        """Extract time step indices from file names (HHMM format)"""
        time_steps = []
        for path in self.line_current_paths:
            # Extract HHMM from filename (e.g., 0000_line_currents.csv -> 0000)
            match = re.search(r'(\d{4})_line_currents\.csv', os.path.basename(path))
            if match:
                time_str = match.group(1)
                
                # If hourly option is enabled, only include files with minutes="00"
                if self.hourly and time_str[2:] != "00":
                    continue
                    
                # Convert HHMM to minutes since midnight for numerical ordering
                hours = int(time_str[:2])
                minutes = int(time_str[2:])
                time_minutes = hours * 60 + minutes
                time_steps.append((time_str, time_minutes))
        
        # Sort by minutes and return the original time strings
        sorted_time_steps = [t[0] for t in sorted(time_steps, key=lambda x: x[1])]
        
        if self.hourly:
            print(f"Hourly mode enabled. Using {len(sorted_time_steps)} hourly time steps.")
        
        return sorted_time_steps

    def _parse_complex_from_mag_ang(self, magnitude, angle_deg):
        """Convert magnitude and angle (in degrees) to complex number components"""
        angle_rad = np.radians(angle_deg)
        real = magnitude * np.cos(angle_rad)
        imag = magnitude * np.sin(angle_rad)
        return real, imag

    def _initialize_dataset(self):
        """Initialize dataset by loading the first time step to get node and branch info"""
        self.node_ids = set()
        self.branch_ids = set()
        
        # Load first time step to get structure
        if self.time_steps:
            t = self.time_steps[0]
            line_path = os.path.join(self.data_dir, f'{t}_line_currents.csv')
            node_path = os.path.join(self.data_dir, f'{t}_node_voltages.csv')
            
            if os.path.exists(line_path) and os.path.exists(node_path):
                line_df = pd.read_csv(line_path)
                node_df = pd.read_csv(node_path)
                
                # Print data types for debugging
                print("\n===== DATA TYPE CHECK =====")
                print(f"Node column type in node_df: {node_df['Node'].dtype}")
                print(f"Sample node from node_df: {node_df['Node'].iloc[0]} (type: {type(node_df['Node'].iloc[0])})")
                print(f"Line column type in line_df: {line_df['Line'].dtype}")
                print(f"Sample line from line_df: {line_df['Line'].iloc[0]} (type: {type(line_df['Line'].iloc[0])})")
                
                # Collect all node IDs - ALWAYS CONVERT TO STRINGS
                for node_id in node_df['Node']:
                    self.node_ids.add(str(node_id))  # Convert to string
                
                # Collect all branch IDs and extract node pairs
                self.branch_to_nodes = {}
                
                for _, row in line_df.iterrows():
                    branch_id = row['Line']
                    # Parse branch_id to get from_node and to_node
                    match = re.match(r'(.+)->(.+)', branch_id)
                    if match:
                        from_node = str(match.group(1))  # Convert to string
                        to_node = str(match.group(2))    # Convert to string
                        self.branch_ids.add(branch_id)
                        self.branch_to_nodes[branch_id] = (from_node, to_node)
                    elif '->' in branch_id:
                        # Fallback if regex doesn't match but arrow is present
                        parts = branch_id.split('->')
                        if len(parts) == 2:
                            from_node = str(parts[0])  # Convert to string 
                            to_node = str(parts[1])    # Convert to string
                            self.branch_ids.add(branch_id)
                            self.branch_to_nodes[branch_id] = (from_node, to_node)
                
                # Create mapping from node ID to index - handle non-numeric IDs
                sorted_nodes = sorted(list(self.node_ids), key=lambda x: str(x))  # Sort as strings
                self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted_nodes)}
                
                # Build adjacency information
                self.adjacency = {node_id: [] for node_id in self.node_ids}
                edges_added = 0
                
                for branch_id, (from_node, to_node) in self.branch_to_nodes.items():
                    if from_node in self.node_ids and to_node in self.node_ids:
                        self.adjacency[from_node].append(to_node)
                        self.adjacency[to_node].append(from_node)  # Undirected graph
                        edges_added += 1
                    else:
                        # Only print first few warnings to avoid excessive output
                        if len(self.adjacency.get(from_node, [])) < 2 and len(self.adjacency.get(to_node, [])) < 2:
                            if from_node not in self.node_ids:
                                print(f"Warning: from_node '{from_node}' not in node_ids")
                            if to_node not in self.node_ids:
                                print(f"Warning: to_node '{to_node}' not in node_ids")
                
                # Print connectivity stats after modifications
                isolated_nodes = sum(1 for neighbors in self.adjacency.values() if len(neighbors) == 0)
                connected_nodes = sum(1 for neighbors in self.adjacency.values() if len(neighbors) > 0)
                print(f"\nAfter type conversion: Connected nodes: {connected_nodes}, Isolated nodes: {isolated_nodes}")
                print(f"Total edges added: {edges_added}")
                print(f"Initialized dataset with {len(self.node_ids)} nodes and {len(self.branch_ids)} branches")

    def _debug_node_connectivity(self, hidden_nodes):
        """Debug function to analyze connectivity of hidden nodes"""
        print("\n===== DEBUGGING NODE CONNECTIVITY =====")
        
        # Count hidden nodes and their connections
        hidden_count = len(hidden_nodes)
        hidden_with_neighbors = 0
        
        # Count father nodes that have hidden neighbors
        fathers_with_hidden_neighbors = 0
        known_nodes = set(self.node_ids) - hidden_nodes
        
        hidden_neighbor_counts = []
        
        # Check each hidden node's connectivity
        for node_id in hidden_nodes:
            neighbors = self.adjacency[node_id]
            known_neighbors = [n for n in neighbors if n in known_nodes]
            
            if known_neighbors:
                hidden_with_neighbors += 1
                hidden_neighbor_counts.append(len(known_neighbors))
        
        # Check each known node's connectivity to hidden nodes
        for node_id in known_nodes:
            neighbors = self.adjacency[node_id]
            has_hidden_neighbor = any(n in hidden_nodes for n in neighbors)
            if has_hidden_neighbor:
                fathers_with_hidden_neighbors += 1
        
        # Print statistics
        print(f"Total nodes: {len(self.node_ids)}")
        print(f"Hidden nodes: {hidden_count} ({hidden_count/len(self.node_ids):.4f})")
        
        # Fix the format specifier error
        if hidden_count > 0:
            hidden_ratio = hidden_with_neighbors / hidden_count
        else:
            hidden_ratio = 0
        print(f"Hidden nodes with known neighbors: {hidden_with_neighbors} ({hidden_ratio:.4f})")
        
        if len(known_nodes) > 0:
            father_ratio = fathers_with_hidden_neighbors / len(known_nodes)
        else:
            father_ratio = 0
        print(f"Father nodes with hidden neighbors: {fathers_with_hidden_neighbors} ({father_ratio:.4f})")
        
        if hidden_neighbor_counts:
            print(f"Avg. known neighbors per hidden node: {sum(hidden_neighbor_counts)/len(hidden_neighbor_counts):.2f}")
        
        print("=======================================\n")

    def _load_time_step_data(self, time_step):
        """Load data for a specific time step"""
        line_path = os.path.join(self.data_dir, f'{time_step}_line_currents.csv')
        node_path = os.path.join(self.data_dir, f'{time_step}_node_voltages.csv')
        
        if not os.path.exists(line_path) or not os.path.exists(node_path):
            raise FileNotFoundError(f"Missing data for time step {time_step}")
        
        line_df = pd.read_csv(line_path)
        node_df = pd.read_csv(node_path)
        
        # Process voltage data
        node_features = {}
        for _, row in node_df.iterrows():
            node_id = str(row['Node'])  # Convert to string
            if node_id not in self.node_ids:
                continue
            
            # Extract phase A voltage (complex components)
            va_mag = row['Va_mag']
            va_ang = row['Va_ang_deg']
            v_real, v_imag = self._parse_complex_from_mag_ang(va_mag, va_ang)
            
            # Use magnitude and phase as additional features
            magnitude = va_mag  # Already have magnitude
            phase_deg = va_ang  # Already have phase in degrees
            
            # Store features: [v_real, v_imag, magnitude, phase_deg]
            node_features[node_id] = [v_real, v_imag, magnitude, phase_deg]
        
        # Process current data
        edge_features = {}
        for _, row in line_df.iterrows():
            branch_id = row['Line']
            if branch_id not in self.branch_ids:
                continue
            
            from_node, to_node = self.branch_to_nodes[branch_id]
            # Strings already by this point, no need to convert
            
            # Parse phase A current (complex components)
            ia_mag = row['Ia_mag']
            ia_ang = row['Ia_ang_deg']
            i_real, i_imag = self._parse_complex_from_mag_ang(ia_mag, ia_ang)
            
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
                s_real = from_v_real * i_real + from_v_imag * i_imag
                s_imag = from_v_imag * i_real - from_v_real * i_imag
                
                # Store edge features [Vdiff_Real, Vdiff_Imag, S_Real, S_Imag]
                edge_features[branch_id] = [vdiff_real, vdiff_imag, s_real, s_imag]
            
        return node_features, edge_features

    def _prepare_pyg_data(self, time_step, node_features, edge_features, hidden_nodes=None):
        """
        Prepare PyTorch Geometric data object for a specific time step with a single virtual child node
        
        Args:
            time_step: Time step string (HHMM format)
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
                # Both from_node and to_node are already strings
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
        
        # Debug edge information
        print(f"Original edge_index shape: {edge_index.shape}")
        
        # Determine nodes to hide if not provided
        if hidden_nodes is None:
            # Calculate node degrees
            degrees = {node_id: len(neighbors) for node_id, neighbors in self.adjacency.items()}
            
            # Filter out isolated nodes (degree 0)
            non_isolated_nodes = [node_id for node_id, degree in degrees.items() if degree > 0]
            
            if not non_isolated_nodes:
                print("WARNING: All nodes are isolated! Cannot create meaningful hiding strategy.")
                # Fall back to original strategy
                sorted_nodes = sorted(degrees.keys(), key=lambda n: degrees[n])
            else:
                # Sort non-isolated nodes by degree
                sorted_nodes = sorted(non_isolated_nodes, key=lambda n: degrees[n])
            
            # Hide the specified ratio of lowest-degree CONNECTED nodes
            num_to_hide = int(len(sorted_nodes) * self.hide_ratio)
            hidden_nodes = set(sorted_nodes[:num_to_hide])
            
            # Debug the connectivity of hidden nodes
            self._debug_node_connectivity(hidden_nodes)
        
        # Create masks for known and hidden nodes
        known_nodes = set(self.node_ids) - hidden_nodes
        
        # Convert node IDs to indices
        known_indices = [self.node_id_to_idx[n] for n in known_nodes if n in self.node_id_to_idx]
        hidden_indices = [self.node_id_to_idx[n] for n in hidden_nodes if n in self.node_id_to_idx]
        
        father_nodes = torch.tensor(known_indices, dtype=torch.long)
        
        # Convert time step from HHMM format to minutes since midnight for numerical representation
        time_minutes = int(time_step[:2]) * 60 + int(time_step[2:])
        
        # Create time index for each node
        time_index = torch.full((num_nodes,), time_minutes, dtype=torch.long)
        
        # The rest of the method remains largely the same as before
        # Handling the training vs. testing data generation
        if self.is_train:
            # Features for known nodes
            father_feats = x[father_nodes]
            Nf = father_nodes.size(0)
            
            # Generate labels for candidate nodes
            real_labels = []
            for i, idx in enumerate(known_indices):
                node_id = sorted(list(self.node_ids), key=lambda x: str(x))[idx]
                # Check if this node has any hidden neighbors
                has_hidden_neighbor = any(neighbor in hidden_nodes for neighbor in self.adjacency[node_id])
                real_labels.append(1 if has_hidden_neighbor else 0)
            real_labels = torch.tensor(real_labels, dtype=torch.long)
            
            # Print positive example stats
            pos_count = sum(real_labels).item()
            pos_ratio = pos_count / len(real_labels) if len(real_labels) > 0 else 0
            print(f"Positive examples: {pos_count}/{len(real_labels)} ({pos_ratio:.4f})")
            
            # Create features for real child
            child_feats_real = father_feats.clone()
            
            # Combine features for all nodes
            node_features_combined = torch.cat([father_feats, child_feats_real], dim=0)
            total_num_nodes = 2 * Nf
            
            # Create indices for candidate nodes
            candidate_real_nodes = torch.arange(Nf, 2 * Nf)
            candidate_nodes = candidate_real_nodes
            candidate_nodes_label = real_labels
            
            # Create time index for all nodes (including candidates)
            time_index_combined = torch.full((total_num_nodes,), time_minutes, dtype=torch.long)
            
            # Create father-child edges
            fc_edges_real = []
            fc_attr_real = []

            for i in range(Nf):
                father_new_idx = i
                child_new_idx = Nf + i
                fc_edges_real.append((father_new_idx, child_new_idx))
                
                if real_labels[i] == 1:
                    # This father has a hidden neighbor
                    father_node_id = sorted(list(self.node_ids), key=lambda x: str(x))[known_indices[i]]
                    # father_node_id is already a string
                    
                    # Find a hidden neighbor
                    hidden_neighbor = None
                    for neighbor in self.adjacency[father_node_id]:
                        if neighbor in hidden_nodes:
                            hidden_neighbor = neighbor
                            break
                    
                    if hidden_neighbor is not None:
                        # Find the branch connecting these nodes
                        branch_key = f"{father_node_id}->{hidden_neighbor}"
                        alt_branch_key = f"{hidden_neighbor}->{father_node_id}"
                        
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
            
            # Filter original edges to keep only father-father edges
            father_edges = []
            father_attrs = []
            
            # Create a mapping from old to new indices for father nodes
            old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(known_indices)}
            
            # Debug the mapping
            print(f"Number of known indices: {len(known_indices)}")
            print(f"old2new mapping size: {len(old2new)}")
            
            if edge_index.size(1) > 0:
                max_source = edge_index[0].max().item()
                max_target = edge_index[1].max().item()
                print(f"Max source idx: {max_source}, Max target idx: {max_target}")
                
                # Find edges between father nodes
                father_father_edges = 0
                for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1])):
                    s_item, t_item = s.item(), t.item()
                    if s_item in old2new and t_item in old2new:
                        father_edges.append((old2new[s_item], old2new[t_item]))
                        father_attrs.append(edge_attr[i].tolist())
                        father_father_edges += 1
                
                print(f"Father-father edges found: {father_father_edges}")
            
            father_edges = torch.tensor(father_edges, dtype=torch.long).T if father_edges else torch.zeros((2, 0), dtype=torch.long)
            father_attrs = torch.tensor(father_attrs, dtype=torch.float) if father_attrs else torch.zeros((0, 4), dtype=torch.float)
            
            print(f"Father edges shape: {father_edges.shape}")
            print(f"FC edges shape: {fc_edges_real.shape}")
            
            # Combine all edges
            new_edge_index = torch.cat([father_edges, fc_edges_real], dim=1) if father_edges.size(1) > 0 else fc_edges_real
            new_edge_attr = torch.cat([father_attrs, fc_attr_real], dim=0) if father_attrs.size(0) > 0 else fc_attr_real
            
            # Debug final edge information
            print(f"Combined edge_index shape: {new_edge_index.shape}")
            
            # Create mask for known nodes
            node_known_mask = torch.zeros(total_num_nodes, dtype=torch.bool)
            node_known_mask[:Nf] = True
            
            # Extract voltage components
            V_real = node_features_combined[:, 0]
            V_imag = node_features_combined[:, 1]
            
            # Extract power components from edge attributes
            known_S_real = new_edge_attr[:, 2] if new_edge_attr.size(0) > 0 else torch.tensor([])
            known_S_imag = new_edge_attr[:, 3] if new_edge_attr.size(0) > 0 else torch.tensor([])
            
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
                fc_edges=fc_edges_real,
                fc_attr=fc_attr_real,
                time_index=time_index_combined,
                time_step=torch.tensor([time_minutes], dtype=torch.long)
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
                node_id = sorted(list(self.node_ids), key=lambda x: str(x))[idx]
                # Check if this node has any hidden neighbors
                has_hidden_neighbor = any(neighbor in hidden_nodes for neighbor in self.adjacency[node_id])
                label_list.append(1 if has_hidden_neighbor else 0)
            label_list = torch.tensor(label_list, dtype=torch.long)
            
            # Print positive example stats
            pos_count = sum(label_list).item()
            pos_ratio = pos_count / len(label_list) if len(label_list) > 0 else 0
            print(f"Positive examples (test): {pos_count}/{len(label_list)} ({pos_ratio:.4f})")
            
            # Combine features
            node_features_combined = torch.cat([father_feats, child_feats], dim=0)
            total_num_nodes = 2 * Nf
            
            # Create candidate nodes
            candidate_nodes = torch.arange(Nf, 2 * Nf)
            candidate_nodes_label = label_list
            
            # Create time index for all nodes
            time_index_combined = torch.full((total_num_nodes,), time_minutes, dtype=torch.long)
            
            # Create father-child edges
            fc_edges = []
            fc_attr = []
            
            for i in range(Nf):
                fc_edges.append((i, Nf + i))
                
                if label_list[i] == 1:
                    # This father has a hidden neighbor
                    father_node_id = sorted(list(self.node_ids), key=lambda x: str(x))[known_indices[i]]
                    
                    # Find a hidden neighbor
                    found_edge = False
                    for neighbor in self.adjacency[father_node_id]:
                        if neighbor in hidden_nodes:
                            # Find the branch connecting these nodes
                            branch_key = f"{father_node_id}->{neighbor}"
                            alt_branch_key = f"{neighbor}->{father_node_id}"
                            
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
            
            if edge_index.size(1) > 0:
                # Find edges between father nodes
                for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1])):
                    s_item, t_item = s.item(), t.item()
                    if s_item in old2new and t_item in old2new:
                        father_edges.append((old2new[s_item], old2new[t_item]))
                        father_attrs.append(edge_attr[i].tolist())
            
            father_edges = torch.tensor(father_edges, dtype=torch.long).T if father_edges else torch.zeros((2, 0), dtype=torch.long)
            father_attrs = torch.tensor(father_attrs, dtype=torch.float) if father_attrs else torch.zeros((0, 4), dtype=torch.float)
            
            # Debug edge information for test data
            print(f"Test father edges shape: {father_edges.shape}")
            print(f"Test FC edges shape: {fc_edges.shape}")
            
            # Combine all edges
            new_edge_index = torch.cat([father_edges, fc_edges], dim=1) if father_edges.size(1) > 0 else fc_edges
            new_edge_attr = torch.cat([father_attrs, fc_attr], dim=0) if father_attrs.size(0) > 0 else fc_attr
            
            # Debug final edge information
            print(f"Test combined edge_index shape: {new_edge_index.shape}")
            
            # Create mask for known nodes
            node_known_mask = torch.zeros(total_num_nodes, dtype=torch.bool)
            node_known_mask[:Nf] = True
            
            # Extract voltage components
            V_real = node_features_combined[:, 0]
            V_imag = node_features_combined[:, 1]
            
            # Extract power components from edge attributes
            known_S_real = new_edge_attr[:, 2] if new_edge_attr.size(0) > 0 else torch.tensor([])
            known_S_imag = new_edge_attr[:, 3] if new_edge_attr.size(0) > 0 else torch.tensor([])
            
            # Create PyG data object
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
                time_step=torch.tensor([time_minutes], dtype=torch.long)
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

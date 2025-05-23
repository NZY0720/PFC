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
        Temporal dataset for power grid data
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
        
        # Initialize dataset 
        self._initialize_dataset()
        
        # Create valid sequence indices
        self.valid_indices = list(range(len(self.time_steps) - sequence_length + 1))
        
        # Cache for loaded data
        self.data_cache = {}

    def _extract_time_steps(self):
        """Extract time step indices from file names (HHMM format)"""
        time_steps = []
        for path in self.line_current_paths:
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
        """Load the first time step and build node/edge mapping"""
        # Start with empty collections
        self.node_ids = set()
        self.branch_ids = set()
        
        # Pick the first time step
        if not self.time_steps:
            return
        
        first_t = self.time_steps[0]
        node_path = os.path.join(self.data_dir, f'{first_t}_node_voltages.csv')
        line_path = os.path.join(self.data_dir, f'{first_t}_line_currents.csv')

        if not (os.path.exists(line_path) and os.path.exists(node_path)):
            raise FileNotFoundError(f'data files missing for {first_t}')

        # Load the data files
        node_df = pd.read_csv(node_path)
        line_df = pd.read_csv(line_path)
        
        print(f"Voltage CSV has {len(node_df)} rows")
        print(f"Line CSV has {len(line_df)} rows")
        print(f"Voltage CSV columns: {node_df.columns.tolist()}")
        print(f"Line CSV columns: {line_df.columns.tolist()}")
        
        # Print sample data for debugging
        print("\nSample voltage data:")
        print(node_df.head(3).to_string())
        print("\nSample line data:")
        print(line_df.head(3).to_string())
        
        # Extract nodes from voltage file (store as integers)
        for nid in node_df['Node']:
            self.node_ids.add(int(nid))
        
        print(f"Found {len(self.node_ids)} unique nodes in voltage CSV")
        print(f"First few nodes: {sorted(list(self.node_ids))[:5]}")
        
        # Define 'from' and 'to' columns for branch connectivity
        from_col, to_col = None, None
        
        # Check for standard column naming 
        if 'From_Node' in line_df.columns and 'To_Node' in line_df.columns:
            from_col, to_col = 'From_Node', 'To_Node'
            print("Using From_Node and To_Node columns directly")
        elif 'from_node' in line_df.columns and 'to_node' in line_df.columns:
            from_col, to_col = 'from_node', 'to_node'
            print("Using from_node and to_node columns (lowercase)")
        else:
            raise ValueError("Could not find connection information in line data. Require From_Node and To_Node columns.")
        
        # Process branch connectivity directly from the line CSV
        self.branch_to_nodes = {}
        
        # Check data types for debugging
        print("\n--- Connectivity Debug ---")
        print(f"From_Node column type: {line_df[from_col].dtype}")
        print(f"To_Node column type: {line_df[to_col].dtype}")
        print(f"Node IDs type: {type(next(iter(self.node_ids)))}")
        
        # Extract branch connectivity directly from line CSV
        print("Extracting branch connectivity from line CSV")
        
        for i, row in line_df.iterrows():
            branch_id = int(row['Line'])
            from_node = int(row[from_col])
            to_node = int(row[to_col])
            
            # Store all branches regardless of whether nodes are in node_ids
            # We'll filter later if needed
            self.branch_ids.add(branch_id)
            self.branch_to_nodes[branch_id] = (from_node, to_node)
            
            # Add to node_ids if not already present
            self.node_ids.add(from_node)
            self.node_ids.add(to_node)
            
            # Debug output for first few branches
            if i < 5:
                print(f"Added branch {branch_id}: {from_node} -> {to_node}")
        
        print(f"Extracted {len(self.branch_ids)} branches from line CSV")
        print(f"Updated node count: {len(self.node_ids)}")
        print(f"First few branch IDs: {sorted(list(self.branch_ids))[:5]}")
        
        # Create node index mapping - ensure sorted for deterministic indexing
        sorted_nodes = sorted(list(self.node_ids))
        self.node_id_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
        
        # Create adjacency list for connectivity
        self.adjacency = {n: [] for n in self.node_ids}
        for branch_id, (frm, to) in self.branch_to_nodes.items():
            self.adjacency[frm].append(to)
            self.adjacency[to].append(frm)
        
        # Print statistics
        isolated = sum(1 for v in self.adjacency.values() if not v)
        connected = len(self.node_ids) - isolated
        print(f"Final grid: {len(self.node_ids)} nodes, {len(self.branch_ids)} branches")
        print(f"Connected nodes: {connected}, Isolated nodes: {isolated}")
        
        # If there are isolated nodes, print a warning
        if isolated > 0:
            print(f"WARNING: {isolated} nodes are isolated! This may affect model performance.")

    def _debug_node_connectivity(self, hidden_nodes):
        """Debug function to analyze connectivity of hidden nodes"""
        print("\n===== DEBUGGING NODE CONNECTIVITY =====")
        
        hidden_count = len(hidden_nodes)
        hidden_with_neighbors = 0
        fathers_with_hidden_neighbors = 0
        known_nodes = set(self.node_ids) - hidden_nodes
        
        hidden_neighbor_counts = []
        hidden_fathers = set()
        
        # Check each hidden node's connectivity
        for node_id in hidden_nodes:
            neighbors = self.adjacency[node_id]
            known_neighbors = [n for n in neighbors if n in known_nodes]
            
            if known_neighbors:
                hidden_with_neighbors += 1
                hidden_neighbor_counts.append(len(known_neighbors))
                # Add these father nodes to our tracking set
                hidden_fathers.update(known_neighbors)
        
        # Print detailed connectivity info
        print(f"Total nodes: {len(self.node_ids)}")
        print(f"Hidden nodes: {hidden_count} ({hidden_count/len(self.node_ids):.4f})")
        print(f"Hidden nodes with known neighbors: {hidden_with_neighbors} ({hidden_with_neighbors/hidden_count:.4f})")
        print(f"Father nodes with hidden neighbors: {len(hidden_fathers)} ({len(hidden_fathers)/len(known_nodes):.4f})")
        
        print("=======================================\n")
        return hidden_fathers

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
            node_id = int(row['Node'])
            
            # Skip if node isn't in our master node list (shouldn't happen since we updated it)
            if node_id not in self.node_ids:
                continue
            
            # Extract phase A voltage (complex components)
            va_mag = row['Va_mag']
            va_ang = row['Va_ang_deg']
            v_real, v_imag = self._parse_complex_from_mag_ang(va_mag, va_ang)
            
            # Store features
            node_features[node_id] = [v_real, v_imag, va_mag, va_ang]
        
        # Process current data for edge features - using current data directly
        edge_features = {}
        
        # Debug edge creation
        print(f"\n--- Edge Features Generation ---")
        print(f"Processing {len(self.branch_ids)} branches for time step {time_step}")
        
        features_count = 0
        
        for _, row in line_df.iterrows():
            branch_id = int(row['Line'])
            
            # Skip if branch isn't in our branch mapping
            if branch_id not in self.branch_ids:
                continue
            
            # Extract current data for all three phases (A, B, C)
            features = [
                row['Ia_mag'], row['Ia_ang_deg'],  # Phase A
                row['Ib_mag'], row['Ib_ang_deg'],  # Phase B
                row['Ic_mag'], row['Ic_ang_deg']   # Phase C
            ]
            
            # Store edge features - ALL edges get features even if nodes aren't in node_features
            edge_features[branch_id] = features
            features_count += 1
        
        print(f"Created {features_count} edge features out of {len(self.branch_ids)} branches")
        
        # If no edge features were created, provide default values
        if features_count == 0 and len(self.branch_ids) > 0:
            print("WARNING: No edge features could be calculated. Using default values.")
            for branch_id in self.branch_ids:
                # Default values for 6 features: mag and angle for 3 phases
                edge_features[branch_id] = [0.01, 0.0, 0.01, -120.0, 0.01, 120.0]
        
        return node_features, edge_features

    def _prepare_pyg_data(self, time_step, node_features, edge_features, hidden_nodes=None):
        """
        Prepare PyTorch Geometric data object for a specific time step
        """
        num_nodes = len(self.node_ids)
        
        # Convert node features to tensor
        x = torch.zeros((num_nodes, 4), dtype=torch.float)
        for node_id, features in node_features.items():
            if node_id in self.node_id_to_idx:
                idx = self.node_id_to_idx[node_id]
                x[idx] = torch.tensor(features, dtype=torch.float)
        
        # Create edge indices and attributes for the original graph
        edge_list = []
        edge_attr_list = []
        
        # Create edges for all branches that have edge features
        for branch_id, (from_node, to_node) in self.branch_to_nodes.items():
            if from_node in self.node_id_to_idx and to_node in self.node_id_to_idx:
                from_idx = self.node_id_to_idx[from_node]
                to_idx = self.node_id_to_idx[to_node]
                
                # Use edge features if available, else default
                if branch_id in edge_features:
                    # For compatibility with the rest of the code, convert 6-feature current data
                    # to 4-feature format expected by the model (VdiffR, VdiffI, S_real, S_imag)
                    current_features = edge_features[branch_id]
                    
                    # Extract phase A current components
                    i_mag = current_features[0]
                    i_ang = current_features[1]
                    i_real, i_imag = self._parse_complex_from_mag_ang(i_mag, i_ang)
                    
                    # Calculate voltage difference if both nodes have features
                    if from_node in node_features and to_node in node_features:
                        from_v_real = node_features[from_node][0]
                        from_v_imag = node_features[from_node][1]
                        to_v_real = node_features[to_node][0]
                        to_v_imag = node_features[to_node][1]
                        
                        vdiff_real = from_v_real - to_v_real
                        vdiff_imag = from_v_imag - to_v_imag
                        
                        # Calculate complex power S = V * I*
                        s_real = from_v_real * i_real + from_v_imag * i_imag
                        s_imag = from_v_imag * i_real - from_v_real * i_imag
                    else:
                        # Default values if voltage data not available
                        vdiff_real = 0.01
                        vdiff_imag = 0.01
                        s_real = 0.1
                        s_imag = 0.05
                    
                    edge_attr = [vdiff_real, vdiff_imag, s_real, s_imag]
                else:
                    # Default edge attributes if no current data
                    edge_attr = [0.01, 0.01, 0.1, 0.05]
                
                # Add forward edge
                edge_list.append((from_idx, to_idx))
                edge_attr_list.append(edge_attr)
                
                # Add backward edge (undirected graph)
                edge_list.append((to_idx, from_idx))
                # Reverse attributes for backward edge
                reverse_attrs = [
                    -edge_attr[0],  # Negate voltage diff
                    -edge_attr[1],  # Negate voltage diff
                    edge_attr[2],   # Keep power flow
                    edge_attr[3]    # Keep power flow
                ]
                edge_attr_list.append(reverse_attrs)
        
        # Convert to tensors for the original grid
        edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float) if edge_attr_list else torch.zeros((0, 4), dtype=torch.float)
        
        # Determine nodes to hide if not provided (lowest degree nodes)
        if hidden_nodes is None:
            # Calculate node degrees
            degrees = {node_id: len(neighbors) for node_id, neighbors in self.adjacency.items()}
            
            # Filter out isolated nodes
            non_isolated_nodes = [node_id for node_id, degree in degrees.items() if degree > 0]
            
            if not non_isolated_nodes:
                print("WARNING: All nodes are isolated! Cannot create meaningful hiding strategy.")
                sorted_nodes = sorted(list(degrees.keys()), key=lambda n: str(n))
            else:
                # Sort by degree (lowest first)
                sorted_nodes = sorted(non_isolated_nodes, key=lambda n: (degrees[n], n))
            
            # Hide the specified ratio of lowest-degree nodes
            num_to_hide = max(int(len(non_isolated_nodes) * self.hide_ratio), 1)
            hidden_nodes = set(sorted_nodes[:num_to_hide])
            
            # Debug connectivity and get known nodes with hidden neighbors
            hidden_fathers = self._debug_node_connectivity(hidden_nodes)
        else:
            # If hidden_nodes are provided, still identify the father nodes
            known_nodes = set(self.node_ids) - hidden_nodes
            hidden_fathers = set()
            for node_id in known_nodes:
                if any(neighbor in hidden_nodes for neighbor in self.adjacency[node_id]):
                    hidden_fathers.add(node_id)
        
        # Create masks for known and hidden nodes
        known_nodes = set(self.node_ids) - hidden_nodes
        
        # Convert node IDs to indices
        known_indices = [self.node_id_to_idx[n] for n in known_nodes if n in self.node_id_to_idx]
        hidden_indices = [self.node_id_to_idx[n] for n in hidden_nodes if n in self.node_id_to_idx]
        
        father_nodes = torch.tensor(known_indices, dtype=torch.long)
        
        # Convert time step to minutes
        time_minutes = int(time_step[:2]) * 60 + int(time_step[2:])
        
        # Create time index for each node
        time_index = torch.full((num_nodes,), time_minutes, dtype=torch.long)
        
        # Generate training/testing data
        if self.is_train:
            # Get features for visible nodes
            father_feats = x[father_nodes]
            Nf = father_nodes.size(0)
            
            # Generate labels for virtual nodes: 1 if parent node connects to any hidden node, 0 otherwise
            real_labels = []
            
            # Debug label generation
            father_with_hidden_count = 0
            
            # Create reference from index to node ID for debugging
            idx_to_node = {idx: node_id for node_id, idx in self.node_id_to_idx.items()}
            
            for i, idx in enumerate(known_indices):
                node_id = idx_to_node[idx]
                
                # Check if this node has any hidden neighbors
                has_hidden_neighbor = False
                for neighbor in self.adjacency[node_id]:
                    if neighbor in hidden_nodes:
                        has_hidden_neighbor = True
                        father_with_hidden_count += 1
                        break
                real_labels.append(1 if has_hidden_neighbor else 0)
            
            real_labels = torch.tensor(real_labels, dtype=torch.long)
            
            pos_count = sum(real_labels).item()
            pos_ratio = pos_count / len(real_labels) if len(real_labels) > 0 else 0
            print(f"Father nodes with hidden neighbors: {father_with_hidden_count}")
            print(f"Positive examples: {pos_count}/{len(real_labels)} ({pos_ratio:.4f})")
            
            # Create features for virtual nodes (initially same as parent nodes)
            child_feats_real = father_feats.clone()
            
            # Combine features for all nodes
            node_features_combined = torch.cat([father_feats, child_feats_real], dim=0)
            total_num_nodes = 2 * Nf
            
            # Create indices for virtual nodes
            candidate_real_nodes = torch.arange(Nf, 2 * Nf)
            candidate_nodes = candidate_real_nodes
            candidate_nodes_label = real_labels
            
            # Create time index for all nodes
            time_index_combined = torch.full((total_num_nodes,), time_minutes, dtype=torch.long)
            
            # Create parent-virtual node edges
            fc_edges_real = []
            fc_attr_real = []

            for i in range(Nf):
                father_idx = i
                child_idx = Nf + i
                fc_edges_real.append((father_idx, child_idx))
                
                if real_labels[i] == 1:
                    # This father has a hidden neighbor
                    father_node_id = idx_to_node[known_indices[i]]
                    
                    # Find a hidden neighbor
                    hidden_neighbor = None
                    for neighbor in self.adjacency[father_node_id]:
                        if neighbor in hidden_nodes:
                            hidden_neighbor = neighbor
                            break
                    
                    if hidden_neighbor is not None:
                        # Find the branch connecting these nodes
                        found_branch = False
                        for branch_id, (frm, to) in self.branch_to_nodes.items():
                            if (frm == father_node_id and to == hidden_neighbor) or \
                               (frm == hidden_neighbor and to == father_node_id):
                                if branch_id in edge_features:
                                    # Convert 6-feature edge features to 4-feature format
                                    current_features = edge_features[branch_id]
                                    
                                    # Extract phase A current components
                                    i_mag = current_features[0]
                                    i_ang = current_features[1]
                                    i_real, i_imag = self._parse_complex_from_mag_ang(i_mag, i_ang)
                                    
                                    # Calculate voltage difference if both nodes have features
                                    if father_node_id in node_features and hidden_neighbor in node_features:
                                        from_v_real = node_features[father_node_id][0]
                                        from_v_imag = node_features[father_node_id][1]
                                        to_v_real = node_features[hidden_neighbor][0]
                                        to_v_imag = node_features[hidden_neighbor][1]
                                        
                                        vdiff_real = from_v_real - to_v_real
                                        vdiff_imag = from_v_imag - to_v_imag
                                        
                                        # Calculate complex power S = V * I*
                                        s_real = from_v_real * i_real + from_v_imag * i_imag
                                        s_imag = from_v_imag * i_real - from_v_real * i_imag
                                    else:
                                        # Default values if voltage data not available
                                        vdiff_real = 0.01
                                        vdiff_imag = 0.01
                                        s_real = 0.1
                                        s_imag = 0.05
                                    
                                    fc_attr_real.append([vdiff_real, vdiff_imag, s_real, s_imag])
                                    found_branch = True
                                    break
                        
                        if not found_branch:
                            # Default features
                            fc_attr_real.append([0.01, 0.01, 0.1, 0.05])
                    else:
                        fc_attr_real.append([0, 0, 0, 0])
                else:
                    fc_attr_real.append([0, 0, 0, 0])
            
            fc_edges_real = torch.tensor(fc_edges_real, dtype=torch.long).T
            fc_attr_real = torch.tensor(fc_attr_real, dtype=torch.float)
            
            # Filter original edges to keep only visible node connections
            father_edges = []
            father_attrs = []
            
            # Create mapping from original indices to new indices
            old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(known_indices)}
            
            if edge_index.size(1) > 0:
                # Find edges between father nodes - only include those between visible nodes
                for i, (s, t) in enumerate(zip(edge_index[0], edge_index[1])):
                    s_item, t_item = s.item(), t.item()
                    if s_item in old2new and t_item in old2new:
                        father_edges.append((old2new[s_item], old2new[t_item]))
                        father_attrs.append(edge_attr[i].tolist())
            
            father_edges = torch.tensor(father_edges, dtype=torch.long).T if father_edges else torch.zeros((2, 0), dtype=torch.long)
            father_attrs = torch.tensor(father_attrs, dtype=torch.float) if father_attrs else torch.zeros((0, 4), dtype=torch.float)
            
            # Combine all edges
            new_edge_index = torch.cat([father_edges, fc_edges_real], dim=1) if father_edges.size(1) > 0 else fc_edges_real
            new_edge_attr = torch.cat([father_attrs, fc_attr_real], dim=0) if father_attrs.size(0) > 0 else fc_attr_real
            
            # Create mask for visible nodes
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
            # Testing data implementation - similar to training
            father_feats = x[father_nodes]
            Nf = father_nodes.size(0)
            
            # Clone father features for child
            child_feats = father_feats.clone()
            
            # Generate labels
            label_list = []
            idx_to_node = {idx: node_id for node_id, idx in self.node_id_to_idx.items()}
            
            for i, idx in enumerate(known_indices):
                node_id = idx_to_node[idx]
                # Direct check for hidden neighbors
                has_hidden_neighbor = False
                for neighbor in self.adjacency[node_id]:
                    if neighbor in hidden_nodes:
                        has_hidden_neighbor = True
                        break
                label_list.append(1 if has_hidden_neighbor else 0)
                
            label_list = torch.tensor(label_list, dtype=torch.long)
            
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
                    # Find a hidden neighbor and its edge attributes
                    father_node_id = idx_to_node[known_indices[i]]
                    
                    found_edge = False
                    for neighbor in self.adjacency[father_node_id]:
                        if neighbor in hidden_nodes:
                            for branch_id, (frm, to) in self.branch_to_nodes.items():
                                if (frm == father_node_id and to == neighbor) or \
                                   (frm == neighbor and to == father_node_id):
                                    if branch_id in edge_features:
                                        # Convert 6-feature format to 4-feature 
                                        current_features = edge_features[branch_id]
                                        
                                        # Extract phase A current components
                                        i_mag = current_features[0]
                                        i_ang = current_features[1]
                                        i_real, i_imag = self._parse_complex_from_mag_ang(i_mag, i_ang)
                                        
                                        # Calculate voltage difference if both nodes have features
                                        if father_node_id in node_features and neighbor in node_features:
                                            from_v_real = node_features[father_node_id][0]
                                            from_v_imag = node_features[father_node_id][1]
                                            to_v_real = node_features[neighbor][0]
                                            to_v_imag = node_features[neighbor][1]
                                            
                                            vdiff_real = from_v_real - to_v_real
                                            vdiff_imag = from_v_imag - to_v_imag
                                            
                                            # Calculate complex power S = V * I*
                                            s_real = from_v_real * i_real + from_v_imag * i_imag
                                            s_imag = from_v_imag * i_real - from_v_real * i_imag
                                        else:
                                            # Default values if voltage data not available
                                            vdiff_real = 0.01
                                            vdiff_imag = 0.01
                                            s_real = 0.1
                                            s_imag = 0.05
                                        
                                        fc_attr.append([vdiff_real, vdiff_imag, s_real, s_imag])
                                        found_edge = True
                                        break
                            
                            if found_edge:
                                break
                    
                    if not found_edge:
                        fc_attr.append([0.001, 0.002, 0.1, 0.05])
                else:
                    fc_attr.append([0, 0, 0, 0])
            
            fc_edges = torch.tensor(fc_edges, dtype=torch.long).T
            fc_attr = torch.tensor(fc_attr, dtype=torch.float)
            
            # Filter original edges to keep only father-father edges
            father_edges = []
            father_attrs = []
            
            # Map from old to new indices for father nodes
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
        """Get a sequence of data starting at the specified index"""
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
            
            # Use the same hidden nodes for consistency across time steps
            data, hidden_nodes = self._prepare_pyg_data(time_step, node_features, edge_features, hidden_nodes)
            sequence.append(data)
        
        # Cache the result
        self.data_cache[idx] = sequence
        return sequence

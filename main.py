import torch
import os
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader

from dataset import TemporalPowerGridDataset
from model import TemporalVirtualNodePredictor
from utils import train_step, visualize_results, visualize_temporal_results

def collate_fn(batch):
    """Custom collate function to handle sequences of PyG data objects"""
    # batch is a list of sequences, where each sequence is a list of PyG Data objects
    # Just return the batch as is, since we handle sequences directly in the training loop
    return batch[0]  # Only handle batch size 1 for simplicity

def train(model, train_dataset, optimizer, device, epochs=3000, sequence_length=3):
    """Train the model on the training dataset"""
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # Process one sequence at a time
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_items = 0
        
        for i, data_sequence in enumerate(train_loader):
            # Move data to device
            for j in range(len(data_sequence)):
                data_sequence[j] = data_sequence[j].to(device)
            
            # Train step
            loss_dict = train_step(
                model, 
                data_sequence, 
                optimizer,
                lambda_edge=1.0,
                lambda_phy=1.0,
                lambda_temporal=1.0
            )
            
            epoch_loss += loss_dict['total_loss']
            epoch_items += 1
            
            # Print progress
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}/{len(train_loader)}, "
                      f"Loss: {loss_dict['total_loss']:.4f}, "
                      f"Node: {loss_dict['node_loss']:.4f}, "
                      f"Edge: {loss_dict['edge_loss']:.4f}, "
                      f"Phy: {loss_dict['phy_loss']:.4f}, "
                      f"Temporal: {loss_dict['temporal_loss']:.4f}")
                
                # Visualize intermediate results
                if i == 0 and epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        node_probs_seq = []
                        node_feats_seq = []
                        for data in data_sequence:
                            node_probs, node_feats_pred, _ = model(
                                data.x,
                                data.edge_index,
                                data.edge_attr,
                                data.candidate_nodes,
                                time_index=data.time_index
                            )
                            node_probs_seq.append(node_probs)
                            node_feats_seq.append(node_feats_pred)
                        
                        visualize_temporal_results(
                            data_sequence,
                            node_probs_seq,
                            node_feats_seq,
                            iteration=f"epoch_{epoch}",
                            threshold=0.5,
                            save_path='./results/train'
                        )
                    model.train()
        
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / epoch_items
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), './results/best_model.pth')
        
        print(f"[Epoch {epoch}] Average Loss: {avg_epoch_loss:.4f}")
    
    return model

def test(model, test_dataset, device, sequence_length=3):
    """Evaluate the model on the test dataset"""
    model.eval()
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # Process one sequence at a time
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Track metrics
    all_results = []
    total_correct = 0
    total_predictions = 0
    
    with torch.no_grad():
        for i, data_sequence in enumerate(test_loader):
            # Move data to device
            for j in range(len(data_sequence)):
                data_sequence[j] = data_sequence[j].to(device)
            
            # Make predictions for each time step
            node_probs_seq = []
            node_feats_seq = []
            for t, data in enumerate(data_sequence):
                node_probs, node_feats_pred, _ = model(
                    data.x,
                    data.edge_index,
                    data.edge_attr,
                    data.candidate_nodes,
                    time_index=data.time_index
                )
                node_probs_seq.append(node_probs)
                node_feats_seq.append(node_feats_pred)
            
            # Calculate metrics
            for t, (data, node_probs, node_feats_pred) in enumerate(zip(data_sequence, node_probs_seq, node_feats_seq)):
                # Extract predictions and ground truth
                y_pred = (node_probs >= 0.5).float()
                y_true = data.candidate_nodes_label.float()
                
                # Update accuracy counts
                batch_correct = (y_pred == y_true).sum().item()
                batch_total = len(y_true)
                total_correct += batch_correct
                total_predictions += batch_total
                
                # Calculate accuracy for this batch
                batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0
                
                # Track results for each candidate node
                for j, (prob, pred, true) in enumerate(zip(node_probs.cpu().numpy(), 
                                                         y_pred.cpu().numpy(), 
                                                         y_true.cpu().numpy())):
                    node_idx = data.candidate_nodes[j].item()
                    v_real_pred = node_feats_pred[node_idx, 0].item()
                    v_imag_pred = node_feats_pred[node_idx, 1].item()
                    
                    result = {
                        'sequence': i,
                        'time_step': t,
                        'node': j,
                        'prob': prob,
                        'pred': pred,
                        'true': true,
                        'v_real_pred': v_real_pred,
                        'v_imag_pred': v_imag_pred,
                        'batch_accuracy': batch_accuracy  # Changed to 'batch_accuracy'
                    }
                    all_results.append(result)
            
            # Visualize results for every few sequences
            if i % 5 == 0:
                visualize_temporal_results(
                    data_sequence,
                    node_probs_seq,
                    node_feats_seq,
                    iteration=f"test_seq_{i}",
                    threshold=0.5,
                    save_path='./results/test'
                )
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('./results/test_predictions.csv', index=False)
    
    # Compute and print overall metrics
    print(f"Test Accuracy: {overall_accuracy:.4f}")
    
    # Add overall accuracy to the results for reference
    # Don't try to use the 'accuracy' column anymore
    
    return results_df

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./results/train', exist_ok=True)
    os.makedirs('./results/test', exist_ok=True)
    
    # 设置最大时间步数，基于是否使用小时级别的数据
    max_time_steps = 24 if args.hourly else 1440
    print(f"Using {'hourly' if args.hourly else 'minute-by-minute'} data with {max_time_steps} time steps")
    
    # 1) 创建数据集，现在包含hourly参数
    print("Loading training dataset...")
    train_dataset = TemporalPowerGridDataset(
        data_dir=args.train_dir,
        hide_ratio=args.hide_ratio,
        is_train=True,
        max_time_steps=max_time_steps,
        sequence_length=args.sequence_length,
        hourly=args.hourly  # 新增参数
    )
    
    print("Loading test dataset...")
    test_dataset = TemporalPowerGridDataset(
        data_dir=args.test_dir,
        hide_ratio=args.hide_ratio,
        is_train=False,
        max_time_steps=max_time_steps,
        sequence_length=args.sequence_length,
        hourly=args.hourly  # 新增参数
    )
    
    # 检查数据集
    sample_seq = train_dataset[0]
    print(f"Sample sequence length: {len(sample_seq)}")
    sample_data = sample_seq[0]
    print(f"Sample data: {sample_data}")
    print(f"Node features shape: {sample_data.x.shape}")
    
    # 2) 创建模型 - 更新模型中的max_time_steps
    node_in_dim = sample_data.x.size(1)
    edge_in_dim = sample_data.edge_attr.size(1) if sample_data.edge_attr is not None else 0
    
    model = TemporalVirtualNodePredictor(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        max_time_steps=max_time_steps,  # 更新时间步数
        dropout=0.1
    ).to(device)
    
    # 3) 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 4) 训练
    if args.mode == 'train' or args.mode == 'both':
        print("Starting training...")
        trained_model = train(
            model, 
            train_dataset, 
            optimizer, 
            device,
            epochs=args.epochs,
            sequence_length=args.sequence_length
        )
    
    # 5) 测试
    if args.mode == 'test' or args.mode == 'both':
        # 如果仅测试，则加载最佳模型
        if args.mode == 'test':
            print("Loading best model...")
            model.load_state_dict(torch.load('./results/best_model.pth'))
        
        print("Starting testing...")
        results = test(
            model, 
            test_dataset, 
            device,
            sequence_length=args.sequence_length
        )
    
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal Power Grid Analysis')
    parser.add_argument('--train_dir', type=str, default='./train_data', 
                        help='Directory containing training data with HHMM CSV files')
    parser.add_argument('--test_dir', type=str, default='./test_data', 
                        help='Directory containing test data with HHMM CSV files')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                        help='Operation mode: train, test, or both')
    parser.add_argument('--epochs', type=int, default=11, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hide_ratio', type=float, default=0.1, help='Ratio of nodes to hide')
    parser.add_argument('--sequence_length', type=int, default=3, 
                        help='Number of consecutive time steps in each sequence')
    parser.add_argument('--hourly', action='store_true', default=True,
                        help='If set, only use data from hourly intervals (HH00)')
    
    args = parser.parse_args()
    main(args)

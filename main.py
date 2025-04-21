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

def calculate_dataset_class_balance(dataset):
    """计算数据集中正负样本的比例，用于设置权重或偏置"""
    total_positive = 0
    total_nodes = 0
    
    # 遍历数据集中的所有序列和数据点
    for i in range(len(dataset)):
        data_sequence = dataset[i]
        for data in data_sequence:
            if hasattr(data, 'candidate_nodes_label'):
                total_positive += data.candidate_nodes_label.sum().item()
                total_nodes += len(data.candidate_nodes_label)
    
    # 计算正样本比例
    pos_ratio = total_positive / total_nodes if total_nodes > 0 else 0
    # 计算推荐的正样本权重
    pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0
    # 计算存在性预测偏置
    # 将偏置从sigmoid反向映射回logit空间
    # 当正样本比例为0.5时，偏置应为0
    exist_bias = np.log(pos_ratio / (1 - pos_ratio)) if 0 < pos_ratio < 1 else 0
    
    print(f"Dataset Statistics:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Positive nodes: {total_positive} ({pos_ratio:.4f})")
    print(f"  Recommended positive weight: {pos_weight:.4f}")
    print(f"  Recommended existence bias: {exist_bias:.4f}")
    
    return pos_ratio, pos_weight, exist_bias

def train(model, train_dataset, optimizer, device, epochs=3000, sequence_length=3, 
          pos_weight=None, use_focal_loss=False, focal_gamma=2.0):
    """Train the model on the training dataset with class balance handling"""
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
            
            # Train step with class balancing
            loss_dict = train_step(
                model, 
                data_sequence, 
                optimizer,
                lambda_edge=1.0,
                lambda_phy=1.0,
                lambda_temporal=1.0,
                pos_weight=pos_weight,  # 加入正样本权重
                focal_gamma=focal_gamma,  # Focal Loss参数
                use_focal_loss=use_focal_loss  # 是否使用Focal Loss
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
                if i == 0 and epoch % 10 == 0:
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

def test(model, test_dataset, device, sequence_length=3, threshold=0.5):
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
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_true_negatives = 0
    
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
                y_pred = (node_probs >= threshold).float()
                y_true = data.candidate_nodes_label.float()
                
                # Update confusion matrix counts
                tp = ((y_pred == 1) & (y_true == 1)).sum().item()
                fp = ((y_pred == 1) & (y_true == 0)).sum().item()
                fn = ((y_pred == 0) & (y_true == 1)).sum().item()
                tn = ((y_pred == 0) & (y_true == 0)).sum().item()
                
                total_true_positives += tp
                total_false_positives += fp
                total_false_negatives += fn
                total_true_negatives += tn
                
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
                        'batch_accuracy': batch_accuracy
                    }
                    all_results.append(result)
            
            # Visualize results for every few sequences
            if i % 5 == 0:
                visualize_temporal_results(
                    data_sequence,
                    node_probs_seq,
                    node_feats_seq,
                    iteration=f"test_seq_{i}",
                    threshold=threshold,
                    save_path='./results/test'
                )
    
    # Calculate overall metrics
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('./results/test_predictions.csv', index=False)
    
    # Compute and print overall metrics
    print(f"Test Results (threshold={threshold:.2f}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {total_true_positives}, FP: {total_false_positives}")
    print(f"    FN: {total_false_negatives}, TN: {total_true_negatives}")
    
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
    
    # 1) 创建数据集
    print("Loading training dataset...")
    train_dataset = TemporalPowerGridDataset(
        data_dir=args.train_dir,
        hide_ratio=args.hide_ratio,
        is_train=True,
        max_time_steps=max_time_steps,
        sequence_length=args.sequence_length,
        hourly=args.hourly
    )
    
    print("Loading test dataset...")
    test_dataset = TemporalPowerGridDataset(
        data_dir=args.test_dir,
        hide_ratio=args.hide_ratio,
        is_train=False,
        max_time_steps=max_time_steps,
        sequence_length=args.sequence_length,
        hourly=args.hourly
    )
    
    # 计算数据集的类别平衡，用于设置权重和偏置
    pos_ratio, pos_weight, exist_bias = calculate_dataset_class_balance(train_dataset)
    
    # 根据命令行参数覆盖计算的权重和偏置
    if args.pos_weight is not None:
        pos_weight = args.pos_weight
        print(f"Using user-specified positive weight: {pos_weight:.4f}")
    
    if args.exist_bias is not None:
        exist_bias = args.exist_bias
        print(f"Using user-specified existence bias: {exist_bias:.4f}")
    
    # 检查数据集
    sample_seq = train_dataset[0]
    print(f"Sample sequence length: {len(sample_seq)}")
    sample_data = sample_seq[0]
    print(f"Sample data: {sample_data}")
    print(f"Node features shape: {sample_data.x.shape}")
    
    # 2) 创建模型 - 添加偏置参数
    node_in_dim = sample_data.x.size(1)
    edge_in_dim = sample_data.edge_attr.size(1) if sample_data.edge_attr is not None else 0
    
    model = TemporalVirtualNodePredictor(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        max_time_steps=max_time_steps,
        dropout=0.1,
        exist_bias=exist_bias  # 加入存在性偏置
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
            sequence_length=args.sequence_length,
            pos_weight=pos_weight,
            use_focal_loss=args.focal_loss,
            focal_gamma=args.focal_gamma
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
            sequence_length=args.sequence_length,
            threshold=args.threshold
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
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hide_ratio', type=float, default=0.2, help='Ratio of nodes to hide')
    parser.add_argument('--sequence_length', type=int, default=3, 
                        help='Number of consecutive time steps in each sequence')
    # 控制是否只使用小时级别的数据
    parser.add_argument('--hourly', action='store_true',  default=True,
                        help='If set, only use data from hourly intervals (HH00)')
    
    # 添加样本不平衡处理的参数
    parser.add_argument('--pos_weight', type=float, default=None, 
                        help='Weight for positive samples in BCE loss (default: auto-calculate)')
    parser.add_argument('--exist_bias', type=float, default=None, 
                        help='Bias term for node existence prediction (default: auto-calculate)')
    parser.add_argument('--focal_loss', action='store_true', 
                        help='Use Focal Loss instead of weighted BCE')
    parser.add_argument('--focal_gamma', type=float, default=4.0, 
                        help='Gamma parameter for Focal Loss (default: 2.0)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Threshold for binary prediction (default: 0.5)')
    
    args = parser.parse_args()
    main(args)



import numpy as np
import pandas as pd

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from mamba_ssm import Mamba
from sklearn.metrics import accuracy_score, f1_score

from model import *
from dataset import *

DEVICE = torch.device("cuda:0")




# Evaluate
def evaluate_model(model, dataloader, device):
    model.eval()
    metrics = {level: {'true': [], 'pred': []} for level in model.classifiers.keys()}
    
    with torch.no_grad():
        for batch in dataloader:
            # Correctly pass the input of dictionary structure
            inputs = {
                'embeds': {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch['embeds'].items()
                }
            }
            outputs = model(inputs)
            
            for level in metrics.keys():
                preds = torch.argmax(outputs[level], dim=1).cpu().numpy()
                trues = batch['labels'][level].numpy()
                metrics[level]['true'].extend(trues)
                metrics[level]['pred'].extend(preds)
    
    # Calculate metrics
    results = {}
    for level in metrics.keys():
        y_true = np.array(metrics[level]['true'])
        y_pred = np.array(metrics[level]['pred'])
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')  # Weighted F1
        results[level] = {'accuracy': acc, 'f1': f1}
    
    return results

# Verify loss calculation
def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            # Correctly pass the input of dictionary structure
            inputs = {
                'embeds': {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch['embeds'].items()
                }
            }
            targets = {
                level: batch['labels'][level].to(device)
                for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
            }
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
    return total_loss / len(dataloader)




def train_coi_only(model, full_data, config, best_model_path):
    # Create a dataset containing only COI
    coi_dataset = MultiModalCOIDataset(
        embeddings_dict = {
            'coi': full_data['coi'],
            'coi_MAE': full_data['coi_MAE']
        }, 
        labels = full_data['labels']
    )
    
    # Divide the training set and validation set
    train_size = int(0.9 * len(coi_dataset))  # train:val=9:1
    val_size = len(coi_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        coi_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"],
        pin_memory=True
    )
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config["learning_rate"]
    )
    criterion = HierarchicalLoss(weights=[1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    
    # Early stop parameters
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience = 30
    #best_model_path = f"./saved_models_run7_MultiBarcodes/coi_stage0_best_{hash(str(config))}.pt"
    
    # Training epoch
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        
        # Training
        for batch in train_loader:
            # Data preparing
            inputs = {
                'embeds': {
                    'coi': batch['embeds']['coi'].to(DEVICE),
                    'coi_MAE': batch['embeds']['coi_MAE'].to(DEVICE)
                }
            }
            
            targets = {
                level: batch['labels'][level].to(DEVICE)
                for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
            }

            #print(f"Input device: {inputs['embeds']['coi'].device}")

            # Verify device consistency
            assert inputs['embeds']['coi'].device == DEVICE, "The input data is not on GPU"
            for level in targets:
                assert targets[level].device == DEVICE, f"The label {level} is not on GPU"
            
            # Forward propagation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Back Propagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        current_acc = np.mean([val_metrics[level]['accuracy'] for level in ['family', 'genus', 'species']])
        
        # Print log
        train_loss = running_loss / len(train_loader)
        print(f"[Stage 0] Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Validation Metrics:")
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            print(f"- {level.capitalize()}: "
                  f"Acc={val_metrics[level]['accuracy']:.4f} | "
                  f"F1={val_metrics[level]['f1']:.4f}")
        print("--------------------------")
        
        # The early stop is based on verifying MEAN accuracy of 'family', 'genus', 'species'
        if current_acc > best_val_acc:
            best_val_acc = current_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    print(f"\n# COI Only Training Completed #"
          f"Best Val Loss: {best_val_loss:.4f}")
    print("\n")
    return model





def train_multimodal(model, full_data, config, best_model_path):
    # Create a complete multimodal dataset
    multimodal_dataset = MultiModalCOIDataset(
        {
            'coi': full_data['coi'],
            'rn16s': full_data['rn16s'],
            'h3': full_data['h3'],
            'rn18s': full_data['rn18s'],
            'its1': full_data['its1'],
            'its2': full_data['its2'],
            # ... Other modal data ...
        },
        full_data['labels']
    )
    
    # Divide the training set and validation set (9:1)
    train_size = int(0.9 * len(multimodal_dataset))
    val_size = len(multimodal_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        multimodal_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create a data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        pin_memory=True
    )
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=0.01
    )
    criterion = HierarchicalLoss(weights=[1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    
    # Training parameters
    #best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience = 30
    #best_model_path = f"./saved_models_run7_MultiBarcodes/multimodal_stage{model.curriculum_stage}_{hash(str(config))}.pt"
    
    # Training epoch
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {
                'embeds': {
                    k: v.to(DEVICE, non_blocking=True)
                    for k, v in batch['embeds'].items()
                }
            }
            targets = {
                level: batch['labels'][level].to(DEVICE, non_blocking=True)
                for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
            }
            
            # Forward propagation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Back Propagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                norm_type=2
            )
            
            optimizer.step()
            running_loss += loss.item()
            
            # Print log
            if (batch_idx + 1) % max(1, int(0.1 * total_batches)) == 0:
                current_loss = running_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{total_batches} | "
                    f"Loss: {current_loss:.4f}")
        
        # Validation
        val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        current_acc = np.mean([val_metrics[level]['accuracy'] for level in ['family', 'genus', 'species']])

        # The early stop is based on verifying MEAN accuracy of 'family', 'genus', 'species'
        if current_acc > best_val_acc:
            best_val_acc = current_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Print log
        avg_train_loss = running_loss / len(train_loader)
        print(f"\n[Stage {model.curriculum_stage}] Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Validation Metrics:")
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            print(f"- {level.capitalize()}: "
                  f"Acc={val_metrics[level]['accuracy']:.4f} | "
                  f"F1={val_metrics[level]['f1']:.4f}")
        print("--------------------------")
    
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    print(f"\n=== Multimodal Training (Stage {model.curriculum_stage}) Completed ==="
          f"Best Val Loss: {best_val_acc:.4f}")
    return model



def test_model(model, test_loader, device):
    model.eval()
    metrics = {level: {'true': [], 'pred': []} for level in model.classifiers.keys()}
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'embeds': {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch['embeds'].items()
                }
            }
            outputs = model(inputs)
            
            for level in metrics.keys():
                preds = torch.argmax(outputs[level], dim=1).cpu().numpy()
                trues = batch['labels'][level].numpy()
                metrics[level]['true'].extend(trues)
                metrics[level]['pred'].extend(preds)
    
    # Calculate metrics
    results = {}
    for level in metrics.keys():
        y_true = np.array(metrics[level]['true'])
        y_pred = np.array(metrics[level]['pred'])
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')  # weighted F1
        results[level] = {'accuracy': acc, 'f1': f1}
    
    return results







def test_model_ValidSample(model, test_loader, config):
    """Evaluate on labeled data in the target domain and return accuracy and F1 score"""
    model.eval()

    metrics = {
        level: {
            'true': [],
            'pred': [],
            'accuracy': 0.0,
            'f1': 0.0,
            'count': 0
        } for level in model.classifiers
    }
    
    with torch.no_grad():
        for batch in test_loader:
            #-------- Only process full-labeled samples --------#
            valid_indices = [i for i, label in enumerate(batch['labels']['species']) 
                            if label != -1]  # Assuming -1 is an invalid label
            
            if not valid_indices:
                continue
                
            # Extract effective samples
            valid_embeds = {
                k: v[valid_indices].to(DEVICE) for k, v in batch['embeds'].items()
            }
            valid_labels = {
                level: batch['labels'][level][valid_indices].to(DEVICE) 
                for level in batch['labels']
            }
            
            # Prediction based solely on valid samples (full-labeled samples)
            outputs = model({'embeds': valid_embeds})
            
            # Print results
            for level in metrics:
                preds = torch.argmax(outputs[level], dim=1).cpu().numpy()
                trues = valid_labels[level].cpu().numpy()
                metrics[level]['true'].extend(trues)
                metrics[level]['pred'].extend(preds)
                metrics[level]['count'] += len(trues)
    
    # Calculate accuracy and F1 score
    results = {}
    for level in metrics:
        if metrics[level]['count'] > 0:
            trues = metrics[level]['true']
            preds = metrics[level]['pred']
            
            # Calculate accuracy based on valid samples
            acc = accuracy_score(trues, preds)
            
            if len(set(trues)) > 2:
                f1 = f1_score(trues, preds, average='weighted')
            else:
                f1 = f1_score(trues, preds, average='binary')
            
            metrics[level]['accuracy_valid'] = acc
            metrics[level]['f1_valid'] = f1

            results[level] = {
                'count_valid': metrics[level]['count'],
                'accuracy_valid': acc,
                'f1_valid': f1,
            }
        else:
            results[level] = {
                'count_valid': 0,
                'accuracy_valid': 0.0,
                'f1_valid': 0.0,
            }
    
    return results








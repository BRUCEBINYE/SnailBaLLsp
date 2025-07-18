

import numpy as np
import pandas as pd

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from mamba_ssm import Mamba
from sklearn.metrics import accuracy_score, f1_score
#from sklearn.model_selection import train_test_split, KFold

from model import *
from dataset import *

DEVICE = torch.device("cuda:3")




# Evaluation
def evaluate_model(model, dataloader, device):
    model.eval()
    metrics = {level: {'true': [], 'pred': []} for level in model.classifiers.keys()}
    
    with torch.no_grad():
        for batch in dataloader:

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
    
    # Calculate matrics
    results = {}
    for level in metrics.keys():
        y_true = np.array(metrics[level]['true'])
        y_pred = np.array(metrics[level]['pred'])
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')  # weighted F1
        results[level] = {'accuracy': acc, 'f1': f1}
    
    return results

# Calculate validation loss
def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:

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
        embeddings_dict = {'coi': full_data['coi']}, 
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
    
    # Early stop parameter
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30  # 与主流程一致
    #best_model_path = f"./saved_models_run7_MultiBarcodes/coi_stage0_best_{hash(str(config))}.pt"
    
    # Training epoch
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0

        for batch in train_loader:

            inputs = {'embeds': {'coi': batch['embeds']['coi'].to(DEVICE)}}

            targets = {
                level: batch['labels'][level].to(DEVICE)
                for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
            }

            #print(f"Input device: {inputs['embeds']['coi'].device}")

            assert inputs['embeds']['coi'].device == DEVICE, "输入数据未在GPU"
            for level in targets:
                assert targets[level].device == DEVICE, f"标签 {level} 未在GPU"
            
            # Forward propagation
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Back propagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        
        # print log
        train_loss = running_loss / len(train_loader)
        print(f"[Stage 0] Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Validation Metrics:")
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            print(f"- {level.capitalize()}: "
                  f"Acc={val_metrics[level]['accuracy']:.4f} | "
                  f"F1={val_metrics[level]['f1']:.4f}")
        print("--------------------------")
        
        # Early stop and save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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



def train_multimodal_with_params(model, full_data, config, optimizer, criterion, best_model_path):
    # Create a multimodal dataset
    multimodal_dataset = MultiModalCOIDataset(
        {modal: full_data[modal] for modal in ['coi', 'rn16s', 'h3', 'rn18s', 'its1', 'its2']},
        full_data['labels']
    )
    
    # Split the training validation set (9:1)
    train_size = int(0.9 * len(multimodal_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        multimodal_dataset, [train_size, len(multimodal_dataset)-train_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create a data loader (using the current batch size)
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
    
    # Early stop parameter
    best_val_acc = 0.0
    patience_counter = 0
    patience = 30
    #best_model_path = f"./saved_models_run7_MultiBarcodes/tmp_best_model_{hash(str(config))}.pt"
    
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
            
            # Back propagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()

            # Progress printing (print every 10% batch)
            if (batch_idx + 1) % max(1, int(0.1 * total_batches)) == 0:
                current_loss = running_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{total_batches} | "
                    f"Loss: {current_loss:.4f}")
        
        # Validation
        val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        current_acc = val_metrics['genus']['accuracy']

        # Early stop and save model
        if current_acc > best_val_acc:
            best_val_acc = current_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        # 日PRint log
        avg_train_loss = running_loss / len(train_loader)
        print(f"\n[Stage {model.curriculum_stage}] Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Validation Metrics:")
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            print(f"- {level.capitalize()}: "
                  f"Acc={val_metrics[level]['accuracy']:.4f} | "
                  f"F1={val_metrics[level]['f1']:.4f}")
        print("--------------------------")
    
        
    return best_val_acc



# -------------------- Grid search training function --------------------
def grid_search_train(full_data, base_config):
    """Perform grid search and return the best parameters"""
    best_metrics = {"genus_accuracy": 0.0, "params": {}}
    
    # Generate parameter combinations
    keys, values = zip(*hyperparam_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for idx, params in enumerate(param_combinations):
        print(f"\n=== Training Combination {idx+1}/{len(param_combinations)} ===")
        print(f"Params: {params}")
        
        # Update Configuration
        tuned_config = base_config.copy()
        tuned_config.update({
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "loss_weights": params["loss_weights"],
            "batch_size": params["batch_size"],
            "attention_heads": params["attention_heads"]
        })

        # Initial model
        coi_model = CurriculumDMGHAN(tuned_config, curriculum_stage=0).to(DEVICE)
        
        try:
            # Stage 0: Train COI only
            trained_coi = train_coi_only(
                model=coi_model,
                full_data=full_data,
                config=tuned_config,
                best_model_path=f"./saved_models_run7_MultiBarcodes/Combination{idx+1}_coi_stage0.pt"
            )
            
            # Stage 1: Multimodal fusion 
            print("\n# Mutilemodal Training #")
            multimodal_model = CurriculumDMGHAN(tuned_config, curriculum_stage=1).to(DEVICE)
            multimodal_model.load_state_dict(trained_coi.state_dict(), strict=False)
            
            # Freeze COI parameters
            for name, param in multimodal_model.named_parameters():
                if 'coi' in name:
                    param.requires_grad_(False)
            
            
            # Custom optimizer and loss
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, multimodal_model.parameters()),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"]
            )
            criterion = HierarchicalLoss(weights=params["loss_weights"])
            
            # Train and achieve optimal validation accuracy
            val_acc = train_multimodal_with_params(
                model=multimodal_model,
                full_data=full_data,
                config=tuned_config,
                optimizer=optimizer,
                criterion=criterion,
                best_model_path=f"./saved_models_run7_MultiBarcodes/Combination{idx+1}_multimodal_stage1.pt"
            )
            
            # Update Best Results
            if val_acc > best_metrics["genus_accuracy"]:
                best_metrics["genus_accuracy"] = val_acc
                best_metrics["params"] = params.copy()
                print(f"New best accuracy (Genus): {val_acc:.4f}")
                
        except RuntimeError as e:
            print(f"Training failed with params {params}: {str(e)}")
            continue
            
    return best_metrics



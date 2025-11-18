

import numpy as np
import pandas as pd

import itertools
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
from addMAEtrain import *

DEVICE = torch.device("cuda:0")




def train_coi_only_withacc(model, full_data, config, best_model_path):
    # Create dataset of COI only
    coi_dataset = MultiModalCOIDataset(
        embeddings_dict = {
            'coi': full_data['coi'],
            'coi_MAE': full_data['coi_MAE']
        }, 
        labels = full_data['labels']
    )
    
    # Training and validation
    train_size = int(0.9 * len(coi_dataset))  # train:val=9:1
    val_size = len(coi_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        coi_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloder
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
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config["learning_rate"]
    )
    criterion = HierarchicalLoss(weights=[1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    
    # Early stop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    patience = 30

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
            
            assert inputs['embeds']['coi'].device == DEVICE, "Input data is not on GPU"
            for level in targets:
                assert targets[level].device == DEVICE, f"Label {level} is not on GPU"
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)
        val_metrics = evaluate_model(model, val_loader, DEVICE)
        current_acc = np.mean([val_metrics[level]['accuracy'] for level in ['family', 'genus', 'species']])
        
        # Print
        train_loss = running_loss / len(train_loader)
        print(f"[Stage 0] Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print("Validation Metrics:")
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            print(f"- {level.capitalize()}: "
                  f"Acc={val_metrics[level]['accuracy']:.4f} | "
                  f"F1={val_metrics[level]['f1']:.4f}")
        print("--------------------------")
        
        # Early stop based on the mean val acc of 'family', 'genus', and 'species'
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
    return best_val_acc




# -------------------- Grid search --------------------

def grid_search_train(full_data, base_config, hyperparam_grid):
    best_metrics = {"mean_accuracy": 0.0, "params": {}}
    
    keys, values = zip(*hyperparam_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for idx, params in enumerate(param_combinations):
        print(f"\n=== Training Combination {idx+1}/{len(param_combinations)} ===")
        print(f"Params: {params}")
        
        tuned_config = base_config.copy()
        tuned_config.update({
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "loss_weights": params["loss_weights"],
            "batch_size": params["batch_size"],
            "attention_heads": params["attention_heads"]
        })
        
        coi_model = CurriculumDMGHANmae(tuned_config, curriculum_stage=0).to(DEVICE)
        
        try:
            val_acc = train_coi_only_withacc(
                model=coi_model,
                full_data=full_data,
                config=tuned_config,
                best_model_path=f"./saved_models_MAE_paramstuning/Combination{idx+1}_coi_stage0.pt"
            )
            
            if val_acc > best_metrics["mean_accuracy"]:
                best_metrics["mean_accuracy"] = val_acc
                best_metrics["params"] = params.copy()
                print(f"New best accuracy (Genus): {val_acc:.4f}")
                
        except RuntimeError as e:
            print(f"Training failed with params {params}: {str(e)}")
            continue
            
    return best_metrics



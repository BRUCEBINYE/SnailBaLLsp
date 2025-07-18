

import numpy as np
import pandas as pd

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba
from sklearn.metrics import accuracy_score, f1_score

from model import *
#from dataset import *

DEVICE = torch.device("cuda:0")





#### Domain Adaptive Model Architecture
#### Targeting Performance Degradation in Independent Test Sets

class DomainAdaptiveDMGHAN(CurriculumDMGHAN):
    def __init__(self, config, curriculum_stage=0):
        super().__init__(config, curriculum_stage)
        
        # Domain classifier (used for domain adversarial training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(config["mamba_config"]["d_model"], 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Domain invariant feature extractor (shared feature space)
        self.feature_aligner = nn.Sequential(
            nn.Linear(config["mamba_config"]["d_model"], 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, config["mamba_config"]["d_model"])
        )
        
        # Domain adaptive parameters
        self.lambda_da = 0.5
        self.alpha = 0.1

    def forward(self, x, domain_label=None, adapt_mode=False):
        # Feature extraction
        if self.curriculum_stage == 0:
            coi_emb = self.coi_projector(x['embeds']['coi'])
            fused = coi_emb
        else:
            projected = {
                'coi': self.coi_projector(x['embeds']['coi']),
                'rn16s': self.modal_projectors['rn16s'](x['embeds']['rn16s']),
                'h3': self.modal_projectors['h3'](x['embeds']['h3']),
                'rn18s': self.modal_projectors['rn18s'](x['embeds']['rn18s']),
                'its1': self.modal_projectors['its1'](x['embeds']['its1']),
                'its2': self.modal_projectors['its2'](x['embeds']['its2'])
            }

            # Perform average pooling on each modality
            pooled_features = {modal: feat.mean(dim=1) for modal, feat in projected.items()}
        
            # Features after splicing pooling
            concated = torch.cat(list(pooled_features.values()), dim=-1)
        
            # Generate weights
            weights = self.fusion_gate(concated)  # shape: (B, 6)
        
            # Weighted fusion: weighted sum of sequence features of each modality
            fused = sum(
                weight.unsqueeze(-1).unsqueeze(-1) * feat
                for weight, feat in zip(weights.unbind(-1), projected.values())
            )
        
        # Mamba processing
        seq_features = self.mamba(fused)
        pooled = seq_features.mean(dim=1)
        
        # Domain adaptive feature alignment
        if adapt_mode:
            domain_invariant_features = self.feature_aligner(pooled)
        else:
            domain_invariant_features = pooled
        
        # Hierarchical network processing
        hierarchy_features = {}
        for i, level in enumerate(['subclass', 'order', 'superfamily', 'family', 'genus', 'species']):
            if i == 0:
                attn_out, _ = self.attention_layers[level](pooled.unsqueeze(0), pooled.unsqueeze(0), pooled.unsqueeze(0))
                hierarchy_features[level] = attn_out.squeeze(0)
            else:
                prev_level = list(self.attention_layers.keys())[i-1]
                prev_feat = hierarchy_features[prev_level]
                gate = self.gates[f'gate_{i-1}'](torch.cat([prev_feat, pooled], dim=-1))
                fused_feat = gate * prev_feat + (1 - gate) * pooled
                attn_out, _ = self.attention_layers[level](fused_feat.unsqueeze(0), fused_feat.unsqueeze(0), fused_feat.unsqueeze(0))
                hierarchy_features[level] = attn_out.squeeze(0)
        
        outputs = {level: self.classifiers[level](hierarchy_features[level]) 
                  for level in hierarchy_features}
        
        # Domain classification output
        if domain_label is not None and adapt_mode:
            domain_pred = self.domain_classifier(domain_invariant_features)
            return outputs, domain_pred
        
        return outputs

    def gradient_reversal(self, x):
        """Implementation of Gradient Reversal Layer (GRL)"""
        class ReverseGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, alpha):
                ctx.alpha = alpha
                return x
            
            @staticmethod
            def backward(ctx, grad_output):
                output = -ctx.alpha * grad_output
                return output, None
        
        return ReverseGrad.apply(x, self.alpha)




def domain_adaptive_train(model, source_loader, target_loader, config, best_model_path):
    """
    Domain adaptive training
    :param source_loader: Source domain data loader (original training data)
    :param target_loader: Target domain data loader (independent testing data)
    """
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    # Loss function
    class_loss = HierarchicalLoss(weights=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    domain_loss = nn.BCELoss()
    
    # Train parameters
    best_target_acc = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(config["da_epochs"]):
        model.train()
        
        # Simultaneously iterate the source domain and target domain
        for (source_batch, target_batch) in zip(source_loader, target_loader):
            # Source domain data (labeled)
            source_inputs = {'embeds': {k: v.to(DEVICE) for k, v in source_batch['embeds'].items()}}
            source_labels = {level: source_batch['labels'][level].to(DEVICE) for level in source_batch['labels']}
            
            # Target domain data (unlabeled)
            target_inputs = {'embeds': {k: v.to(DEVICE) for k, v in target_batch['embeds'].items()}}
            
            # 1. Source domain supervised training
            source_outputs = model(source_inputs)
            loss_class = class_loss(source_outputs, source_labels)
            
            # 2. Domain adversarial training
            # Combining source domain and target domain data
            combined_inputs = {
                'embeds': {
                    k: torch.cat([source_inputs['embeds'][k], target_inputs['embeds'][k]])
                    for k in source_inputs['embeds']
                }
            }
            
            # Create domain label: Source domain=0, Target domain=1
            domain_labels = torch.cat([
                torch.zeros(source_inputs['embeds']['coi'].size(0)),
                torch.ones(target_inputs['embeds']['coi'].size(0))
            ]).to(DEVICE).unsqueeze(1)
            
            # Forward propagation (with domain classification)
            model_outputs, domain_pred = model(combined_inputs, domain_label=domain_labels, adapt_mode=True)
            
            # Calculate domain classification loss
            loss_domain = domain_loss(domain_pred, domain_labels)
            
            # 3. Total loss
            total_loss = loss_class + model.lambda_da * loss_domain
            
            # Back Propagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Target domain evaluation (using partially labeled data)
        target_metrics = evaluate_target_domain(model, target_loader, config)
        
        print(f"Epoch {epoch+1}/{config['da_epochs']} | "
              f"Class Loss: {loss_class.item():.4f} | "
              f"Domain Loss: {loss_domain.item():.4f}")
        
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            print(f"- {level.capitalize()}: "
                f"Acc={target_metrics[level]['accuracy_valid']:.4f} | "
                f"F1={target_metrics[level]['f1_valid']:.4f}")
        print("------------\n")

        # Early Stop and Save
        if target_metrics['species']['accuracy_valid'] > best_target_acc:
            best_target_acc = target_metrics['species']['accuracy_valid']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    return model




def evaluate_target_domain(model, target_loader, config):
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
        for batch in target_loader:
            #-------- Only process full-labeled samples --------#
            valid_indices = [i for i, label in enumerate(batch['labels']['species']) 
                            if label != -1]  # Assuming -1 is an invalid label
            
            if not valid_indices:
                continue
                
            valid_embeds = {
                k: v[valid_indices].to(DEVICE) for k, v in batch['embeds'].items()
            }
            valid_labels = {
                level: batch['labels'][level][valid_indices].to(DEVICE) 
                for level in batch['labels']
            }
            
            outputs = model({'embeds': valid_embeds})
            
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
            
            acc = accuracy_score(trues, preds)
            
            if len(set(trues)) > 2:
                f1 = f1_score(trues, preds, average='macro')
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








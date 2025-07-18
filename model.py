

import numpy as np
import pandas as pd

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba




# Dynamic Multi Granular Hierarchical Network
class DMGHAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mamba = Mamba(
            d_model=config["mamba_config"]["d_model"],
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.input_proj = nn.Linear(config["embed_dim"], config["mamba_config"]["d_model"])
        
        self.attention_layers = nn.ModuleDict({
            level: nn.MultiheadAttention(
                embed_dim=256,
                num_heads=config["attention_heads"],
                dropout=0.1
            ) for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
        })
        
        self.gates = nn.ModuleDict({
            f'gate_{i}': nn.Sequential(
                nn.Linear(256*2, 256),
                nn.Sigmoid()
            ) for i in range(5)
        })
        
        self.classifiers = nn.ModuleDict({
            level: nn.Linear(256, config["n_classes"][i])
            for i, level in enumerate(['subclass', 'order', 'superfamily', 'family', 'genus', 'species'])
        })

    def forward(self, x):
        x = self.input_proj(x)
        seq_features = self.mamba(x)
        pooled = seq_features.mean(dim=1)
        
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
        
        return {level: self.classifiers[level](hierarchy_features[level]) for level in hierarchy_features}


# Hierarchical loss function
class HierarchicalLoss(nn.Module):
    def __init__(self, weights=[1.2, 1.0, 0.8, 0.6, 0.4, 0.2]):
        super().__init__()
        self.weights = weights
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        return sum(self.weights[i] * self.ce(outputs[level], targets[level]) 
                   for i, level in enumerate(['subclass', 'order', 'superfamily', 'family', 'genus', 'species']))



class MultimodalFusionLayer(nn.Module):
    def __init__(self, input_dim=768*6, hidden_dim=256):
        super().__init__()
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, modal_embeddings):
        # modal_embeddings: dict {modal_name: tensor}
        concatenated = torch.cat([emb for emb in modal_embeddings.values()], dim=-1)
        return self.fusion_net(concatenated)



class CurriculumDMGHAN(DMGHAN):
    def __init__(self, config, curriculum_stage=0):
        super().__init__(config)
        self.curriculum_stage = curriculum_stage
        del self.input_proj
        
        # COI Projection Layer: 768 -> 256
        self.coi_projector = nn.Linear(768, config["mamba_config"]["d_model"])
        
        if curriculum_stage > 0:
            # Other modal projection layers: 768 -> 256
            self.modal_projectors = nn.ModuleDict({
                modal: nn.Linear(768, config["mamba_config"]["d_model"])
                for modal in ['rn16s', 'h3', 'rn18s', 'its1', 'its2']
            })  
            # Fusion Gate Network: 256*6 -> 6
            self.fusion_gate = nn.Sequential(
                nn.Linear(config["mamba_config"]["d_model"] * 6, 6),
                nn.Softmax(dim=-1)
            )
        
    
    def forward(self, x):
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
            weights = self.fusion_gate(concated)
        
            # Weighted fusion: weighted sum of sequence features of each modality
            fused = sum(
                weight.unsqueeze(-1).unsqueeze(-1) * feat
                for weight, feat in zip(weights.unbind(-1), projected.values())
            )
    
        # Input features into Mamba module
        seq_features = self.mamba(fused)  # The input dimension should be (B, seq_len, d_model)
        pooled = seq_features.mean(dim=1)
        

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
        
        return {level: self.classifiers[level](hierarchy_features[level]) for level in hierarchy_features}






class CurriculumDMGHANmae(DMGHAN):
    def __init__(self, config, curriculum_stage=0):
        super().__init__(config)
        self.curriculum_stage = curriculum_stage
        del self.input_proj
        
        # Add COI fusion module
        self.coi_fusion = nn.Sequential(
            nn.Linear(config["mamba_config"]["d_model"] * 2,  # The input is a combination of primary and additional COI features
                     config["mamba_config"]["d_model"]),
            nn.ReLU(),
            nn.LayerNorm(config["mamba_config"]["d_model"])
        )
        
        # The original COI projection layer remains unchanged
        self.coi_projector = nn.Linear(768, config["mamba_config"]["d_model"])
        self.coi_MAE_projector = nn.Linear(768, config["mamba_config"]["d_model"])
        
        if curriculum_stage > 0:
            # Other modal projection layers: 768 -> 256
            self.modal_projectors = nn.ModuleDict({
                modal: nn.Linear(768, config["mamba_config"]["d_model"])
                for modal in ['rn16s', 'h3', 'rn18s', 'its1', 'its2']
            })  
            # Fusion gate network: 256*6 -> 6
            self.fusion_gate = nn.Sequential(
                nn.Linear(config["mamba_config"]["d_model"] * 6, 6),
                nn.Softmax(dim=-1)
            )
        
    
    def forward(self, x):
        if self.curriculum_stage == 0:
            # Stage 0: RNA+DNA COI modal fusion
            coi_main = self.coi_projector(x['embeds']['coi'])  # [B, 12, 256]
            coi_MAE = self.coi_MAE_projector(x['embeds']['coi_MAE'])  # [B, 768] -> [B, 256]
        
            # Expand auxiliary features to match sequence length
            coi_MAE_expanded = coi_MAE.unsqueeze(1).expand(-1, 12, -1)  # [B, 12, 256]
        
            # Splicing and fusing features
            fused_coi = torch.cat([coi_main, coi_MAE_expanded], dim=-1)  # [B, 12, 512]
            fused = self.coi_fusion(fused_coi)  # [B, 12, 256]
        else:
            projected = {
                'coi': self.coi_projector(x['embeds']['coi']),
                'rn16s': self.modal_projectors['rn16s'](x['embeds']['rn16s']),
                'h3': self.modal_projectors['h3'](x['embeds']['h3']),
                'rn18s': self.modal_projectors['rn18s'](x['embeds']['rn18s']),
                'its1': self.modal_projectors['its1'](x['embeds']['its1']),
                'its2': self.modal_projectors['its2'](x['embeds']['its2'])
            }

            pooled_features = {modal: feat.mean(dim=1) for modal, feat in projected.items()}
        
            concated = torch.cat(list(pooled_features.values()), dim=-1)
        
            weights = self.fusion_gate(concated)
        
            fused = sum(
                weight.unsqueeze(-1).unsqueeze(-1) * feat
                for weight, feat in zip(weights.unbind(-1), projected.values())
            )

        # Input features into Mamba module
        seq_features = self.mamba(fused)
        pooled = seq_features.mean(dim=1)
        

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
        
        return {level: self.classifiers[level](hierarchy_features[level]) for level in hierarchy_features}



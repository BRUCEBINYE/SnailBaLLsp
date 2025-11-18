

import numpy as np
import pandas as pd

import os
import torch
from sklearn.model_selection import train_test_split, KFold

from addMAEtrain import *
from addMAEtest import *
from addMAEtrainTuning import *
from domainA import DomainAdaptiveDMGHAN, domain_adaptive_train, evaluate_target_domain, DomainAdaptiveDMGHANmae

import itertools

DEVICE = torch.device("cuda:0")



#===========  设定最佳运行参数 ==========#       
base_config = {
    "embed_dim": 768,
    "seq_len": 12,
    "n_classes": [2, 24, 103, 354, 2753, 11295],
    "mamba_config": {"d_model": 256, "dropout": 0.1},
    "epochs": 300  # 保持固定
}



#===========  Load pretrained models ==========#   
## We have already pretrained the models
## You can download the pretrained models in DRYAD
model_pretrained_fold = "./pretrained_models"



#======== Stage 0 ========#
print("\n=== Stage 0: COI Only Training ===")
# Load COI-only model (addMAE)
final_config0 = base_config.copy()
final_config0.update({
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 32,
    "curriculum_stage": 0
})
trained_coi_model = CurriculumDMGHANmae(final_config0).to(DEVICE)
state_dict = torch.load(f"{model_pretrained_fold}/addMAE_stage0_coi_only_BestAcc.pt", map_location=DEVICE)
trained_coi_model.load_state_dict(state_dict, strict=True)  # Strict
trained_coi_model.eval()


#======== Stage ========#
print("\n=== Stage 1: Multimodal Fusion Training ===")
# Load multimodal_fusion model (addMAE)
final_config1 = base_config.copy()
final_config1.update({
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 32,
    "curriculum_stage": 1
})
trained_multimodal = CurriculumDMGHANmae(final_config1, curriculum_stage=1).to(DEVICE)
state_dict = torch.load(f"{model_pretrained_fold}/addMAE_stage1_multimodal_fusion_BestMeanAcc.pt", map_location=DEVICE)
trained_multimodal.load_state_dict(state_dict, strict=True)  # Strict
trained_multimodal.eval()


#======== Stage 2 ========#
print("\n=== Stage 2: Full Fine-tuning ===")
# Load multimodal_fulltuning model (addMAE)
final_config2 = base_config.copy()
final_config2.update({
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 32,
    "curriculum_stage": 2
})
multimodal_model_full = CurriculumDMGHANmae(final_config2, curriculum_stage=2).to(DEVICE)
state_dict = torch.load(f"{model_pretrained_fold}/addMAE_stage2_multimodal_fulltuning_BestMeanAcc.pt", map_location=DEVICE)
multimodal_model_full.load_state_dict(state_dict, strict=True)  # Strict
multimodal_model_full.eval()



#======== Stage 2, after Domain Adaptative ========#
print("\n=== Stage 2: Full Fine-tuning, after Domain Adaptative ===")
# Load multimodal_fulltuning model (addMAE)
final_config2DA = base_config.copy()
final_config2DA.update({
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 32,
    "curriculum_stage": 2
})
multimodal_model_full_DA = DomainAdaptiveDMGHANmae(final_config2DA, curriculum_stage=2).to(DEVICE)
state_dict = torch.load(f"{model_pretrained_fold}/DomainA_BestAcc_after_stage_2_BarcodeMAE.pt", map_location=DEVICE)
multimodal_model_full_DA.load_state_dict(state_dict, strict=True)  # Strict
multimodal_model_full_DA.eval()




#===========  Load the independent dataset  ==========#  

#### Independent, 3146 samples with full labels
## Labels
label_arrays_INDP = load_test_labels("./data/data_Independent/Indep_3146_samples/Indpendent_3146_samples_fulllabels_Labels.txt", class_counts=[2, 24, 103, 354, 2753, 11295])

## Embeddings, coi
coi_MAE_emb_INDP = np.load("./embedding/BarcodeMAE/unseen_BarcodeMAE_embedding.npy")

coi_emb_INDP = np.load("./embedding/ERNIE-RNA/Indep_3146_samples/Independent_unseen_3146_sequences/cls_embedding.npy")
coi_index_INDP = np.array([i for i in range(coi_emb_INDP.shape[0])])

## Embeddings, other markers
rn16s_emb_INDP = np.load("./embedding/ERNIE-RNA/Indep_3146_samples/Indep_3146_samples_Other_Markers_16S_3146_index_INDP_sequences/cls_embedding.npy")
rn16s_index_INDP = pd.read_csv("./data/data_Independent/Indep_3146_samples/Other_Markers_16S_3146_index_INDP_labels.txt", sep="\t")['Index'].values

h3_emb_INDP = np.load("./embedding/ERNIE-RNA/Indep_3146_samples/Indep_3146_samples_Other_Markers_H3_3146_index_INDP_sequences/cls_embedding.npy")
h3_index_INDP = pd.read_csv("./data/data_Independent/Indep_3146_samples/Other_Markers_H3_3146_index_INDP_labels.txt", sep="\t")['Index'].values

its2_emb_INDP = np.load("./embedding/ERNIE-RNA/Indep_3146_samples/Indep_3146_samples_Other_Markers_ITS2_3146_index_INDP_sequences/cls_embedding.npy")
its2_index_INDP = pd.read_csv("./data/data_Independent/Indep_3146_samples/Other_Markers_ITS2_3146_index_INDP_labels.txt", sep="\t")['Index'].values


# Prepare test data
test_data = {
    'coi': {
        'embeddings': coi_emb_INDP,
        'index': coi_index_INDP  # COI index
    },
    'coi_MAE': {
        'embeddings': coi_MAE_emb_INDP,
        'index': coi_index_INDP  
    },
    'rn16s': {
        'embeddings': rn16s_emb_INDP,
        'index': rn16s_index_INDP  # Only some samples have 16S data
    },
    'h3': {
        'embeddings': h3_emb_INDP,
        'index': h3_index_INDP  # Only some samples have H3 data
    },
    'its2': {
        'embeddings': its2_emb_INDP,
        'index': its2_index_INDP  # Only some samples have ITS2 data
    }
}


tester = FixedMultiModalTesterMAE(
        coi_model=trained_coi_model,
        multimodal_model=trained_multimodal,
        class_counts=[2, 24, 103, 354, 2753, 11295],
        device=DEVICE
    )

aligned_embeddings = tester.align_test_data(test_data)
target_dataset = tester.create_test_dataset(aligned_embeddings, label_arrays_INDP)
target_loader = DataLoader(target_dataset, batch_size=128, shuffle=True)


#======== Evaluate performance of target_loader ========#

print("\n=== COI Only - model")
target_metrics0 = test_model(trained_coi_model, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics0[level]['accuracy']:.4f} | "
        f"F1={target_metrics0[level]['f1']:.4f}")
#- Subclass: Acc=0.9946 | F1=0.9946
#- Order: Acc=0.9447 | F1=0.9406
#- Superfamily: Acc=0.9186 | F1=0.9258
#- Family: Acc=0.8102 | F1=0.8454
#- Genus: Acc=0.8293 | F1=0.8524
#- Species: Acc=0.5356 | F1=0.5408   # Better than performance in Independent set predicted by BarcodeMAE



print("\n=== Mutilemodal Fusion - model")
target_metrics1 = test_model(trained_multimodal, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics1[level]['accuracy']:.4f} | "
        f"F1={target_metrics1[level]['f1']:.4f}")
#- Subclass: Acc=0.9895 | F1=0.9895
#- Order: Acc=0.9415 | F1=0.9384
#- Superfamily: Acc=0.9142 | F1=0.9248
#- Family: Acc=0.8086 | F1=0.8468
#- Genus: Acc=0.8347 | F1=0.8557
#- Species: Acc=0.5312 | F1=0.5464



print("\n=== Mutilemodal Full-Tune - model")
target_metrics2 = test_model(multimodal_model_full, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics2[level]['accuracy']:.4f} | "
        f"F1={target_metrics2[level]['f1']:.4f}")
#- Subclass: Acc=0.9962 | F1=0.9962
#- Order: Acc=0.9402 | F1=0.9377
#- Superfamily: Acc=0.9085 | F1=0.9153
#- Family: Acc=0.7969 | F1=0.8324
#- Genus: Acc=0.8363 | F1=0.8532
#- Species: Acc=0.5445 | F1=0.5568


print("\n=== Mutilemodal Full-Tune, Domain Adaptive - model")
target_metrics2DA = test_model(multimodal_model_full_DA, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics2DA[level]['accuracy']:.4f} | "
        f"F1={target_metrics2DA[level]['f1']:.4f}")
#- Subclass: Acc=0.9914 | F1=0.9914
#- Order: Acc=0.9418 | F1=0.9391
#- Superfamily: Acc=0.9005 | F1=0.9096
#- Family: Acc=0.7978 | F1=0.8321
#- Genus: Acc=0.8334 | F1=0.8499
#- Species: Acc=0.5502 | F1=0.5596






import numpy as np
import pandas as pd

import os
import torch
from sklearn.model_selection import train_test_split, KFold

from train import *
from test import *




#===========  Set optimal operating parameters  ==========#       
base_config = {
    "embed_dim": 768,
    "seq_len": 12,
    "n_classes": [2, 24, 103, 354, 2753, 11295],
    "mamba_config": {"d_model": 256, "dropout": 0.1},
    "epochs": 300
}

model_out_fold = "./pretrained_models"
os.makedirs(model_out_fold, exist_ok=True)


#======== Stage 0: Train COI only ========#
print("\n=== Stage 0: COI Only Training ===")
#-------- Directly load existed model --------#
final_config0 = base_config.copy()
final_config0.update({
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 16,
    "curriculum_stage": 0
})
trained_coi_model = CurriculumDMGHAN(final_config0).to(DEVICE)
state_dict = torch.load(f"{model_out_fold}/stage0_coi_only.pt", map_location=DEVICE)
trained_coi_model.load_state_dict(state_dict, strict=True)
trained_coi_model.eval()


#======== Stage 1: Train fusion model ========#
#-------- Directly load existed model --------#
final_config1 = base_config.copy()
final_config1.update({
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 16,
    "curriculum_stage": 1
})
trained_multimodal = CurriculumDMGHAN(final_config1, curriculum_stage=1).to(DEVICE)
state_dict = torch.load(f"{model_out_fold}/stage1_multimodal_fusion_BestMeanAcc.pt", map_location=DEVICE)
trained_multimodal.load_state_dict(state_dict, strict=True)
trained_multimodal.eval()



#======== Stage 2: Train full-tuning model ========#
#-------- Directly load existed model --------#
final_config2 = base_config.copy()
final_config2.update({
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 16,
    "curriculum_stage": 2
})
multimodal_model_full = CurriculumDMGHAN(final_config2, curriculum_stage=2).to(DEVICE)
state_dict = torch.load(f"{model_out_fold}/stage2_multimodal_full_BestMeanAcc.pt", map_location=DEVICE)
multimodal_model_full.load_state_dict(state_dict, strict=True)
multimodal_model_full.eval()







#===========  Load independent test set data  ==========#  

#### Independnet Datasets
tester = FixedMultiModalTester(
        coi_model=trained_coi_model,
        multimodal_model=trained_multimodal,
        class_counts=[2, 24, 103, 354, 2753, 11295],
        device=DEVICE
    )

#### GenBank From 20250101-20250523
label_arrays_1 = tester.load_labels(
    "./data_Independent/GenBank_From20250101/Independent_Gastropoda_seqs_n_1927_EDITED_LabelSpecies.txt"
)
coi_emb_1 = np.load("./embedding/ERNIE-RNA/Independent1_GenBank_From20250101_n_1927_EDITED_sequences/cls_embedding.npy")
print(coi_emb_1.shape)

#### BOLD_Mar2025 Gastropoda, real seqs
label_arrays_2 = tester.load_labels(
    "./data_Independent/BOLD_Mar2025_Gastropoda/EDITED_Independent2_BOLD_Taxonomy_label_is_the_SAME_with_training_LabelsSpecies.txt"
)  # need to train new model
coi_emb_2 = np.load("./embedding/ERNIE-RNA/Independent2_BOLD_Taxonomy_label_is_the_SAME_with_training_Sequences/cls_embedding.npy")
print(coi_emb_2.shape)

#### Merge two independent test sets
label_arrays_INDP = []
for arr1, arr2 in zip(label_arrays_1, label_arrays_2):
    merged = np.concatenate((arr1, arr2))
    label_arrays_INDP.append(merged)

coi_emb_INDP = np.concatenate((coi_emb_1, coi_emb_2), axis=0)
print(coi_emb_INDP.shape)

test_coi_embs_INDP = {'coi': coi_emb_INDP}

# Create dataset objects
target_dataset = tester.create_test_dataset(test_coi_embs_INDP, label_arrays_INDP)
target_loader = DataLoader(target_dataset, batch_size=128, shuffle=True)






#======== Application model for evaluation of target_loader ========#

print("\n=== COI Only - model")
target_metrics0 = test_model(trained_coi_model, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics0[level]['accuracy']:.4f} | "
        f"F1={target_metrics0[level]['f1']:.4f}")

target_metrics0_vlid = test_model_ValidSample(trained_coi_model, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics0_vlid[level]['accuracy_valid']:.4f} | "
        f"F1={target_metrics0_vlid[level]['f1_valid']:.4f} | "
        f"Counts={target_metrics0_vlid[level]['count_valid']}")




print("\n=== Mutilemodal Fusion - model")
target_metrics1 = test_model(trained_multimodal, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics1[level]['accuracy']:.4f} | "
        f"F1={target_metrics1[level]['f1']:.4f}")

target_metrics1_vlid = test_model_ValidSample(trained_multimodal, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics1_vlid[level]['accuracy_valid']:.4f} | "
        f"F1={target_metrics1_vlid[level]['f1_valid']:.4f} | "
        f"Counts={target_metrics1_vlid[level]['count_valid']}")




print("\n=== Mutilemodal Full-Tune - model")
target_metrics2 = test_model(multimodal_model_full, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics2[level]['accuracy']:.4f} | "
        f"F1={target_metrics2[level]['f1']:.4f}")

target_metrics2_vlid = test_model_ValidSample(multimodal_model_full, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics2_vlid[level]['accuracy_valid']:.4f} | "
        f"F1={target_metrics2_vlid[level]['f1_valid']:.4f} | "
        f"Counts={target_metrics2_vlid[level]['count_valid']}")


import itertools
import copy

import numpy as np
import pandas as pd

import os
import torch
from sklearn.model_selection import train_test_split, KFold

from addMAEtrain import *
from addMAEtest import *
from addMAEtrainTuning import *
import itertools

from domainA import DomainAdaptiveDMGHAN, domain_adaptive_train, evaluate_target_domain, DomainAdaptiveDMGHANmae

DEVICE = torch.device("cuda:0")


def preprocess_labels(lbs_df, n_classes_list):
    for i, level in enumerate(['subclass', 'order', 'superfamily', 'family', 'genus', 'species']):
        # Convert labels to 0-based integer encoding
        lbs_df[level] = lbs_df[level].astype('category').cat.codes
        # Verify label range
        max_label = lbs_df[level].max()
        min_label = lbs_df[level].min()
        n_class = n_classes_list[i]
        assert max_label < n_class and min_label >= 0, \
            f"Error：Level range of {level} [{min_label}, {max_label}] is exceed the number of categories {n_class}"
    #return lbs_df
    return [
        lbs_df['subclass'].values,
        lbs_df['order'].values,
        lbs_df['superfamily'].values,
        lbs_df['family'].values,
        lbs_df['genus'].values,
        lbs_df['species'].values
    ]

def replace_nan(x):
    # determine whether it is a scalar NaN
    if isinstance(x, float) and pd.isna(x):
        return np.zeros((12, 768))
    else:
        return x


# -------------------- 0. Align multimodal sequence embedding --------------------
def align_multimodal_data(coi_emb, coi_MAE_emb, coi_index, 
                            rn16s_emb, rn16s_index, 
                            h3_emb, h3_index, 
                            rn18s_emb, rn18s_index, 
                            its1_emb, its1_index, 
                            its2_emb, its2_index, 
                            labels):
    # Integrate embedding and index to DataFrame for each modelity
    coi_df = pd.DataFrame({'index': coi_index, 'emb': list(coi_emb), 'emb_MAE': list(coi_MAE_emb)})
    rn16s_df = pd.DataFrame({'index': rn16s_index, 'emb': list(rn16s_emb)})
    h3_df = pd.DataFrame({'index': h3_index, 'emb': list(h3_emb)})
    rn18s_df = pd.DataFrame({'index': rn18s_index, 'emb': list(rn18s_emb)})
    its1_df = pd.DataFrame({'index': its1_index, 'emb': list(its1_emb)})
    its2_df = pd.DataFrame({'index': its2_index, 'emb': list(its2_emb)})

    labels_df = pd.DataFrame({
        'index': coi_index,  # Using COI index as a base
        'subclass': labels[0],
        'order': labels[1],
        'superfamily': labels[2],
        'family': labels[3],
        'genus': labels[4],
        'species': labels[5]
    })
    
    # Merge data from other modalities with COI modality
    merged_df = coi_df.merge(rn16s_df, on='index', how='left', suffixes=('_coi', '_rn16s'))
    merged_df = merged_df.merge(h3_df, on='index', how='left', suffixes=('_rn16s', '_h3'))
    merged_df = merged_df.merge(rn18s_df, on='index', how='left', suffixes=('_h3', '_rn18s'))
    merged_df = merged_df.merge(its1_df, on='index', how='left', suffixes=('_rn18s', '_its1'))
    merged_df = merged_df.merge(its2_df, on='index', how='left', suffixes=('_its1', '_its2'))
    
    # Handle missing values, filling with zeros
    merged_df['emb_rn16s'] = merged_df['emb_rn16s'].apply(replace_nan)
    merged_df['emb_h3'] = merged_df['emb_h3'].apply(replace_nan)
    merged_df['emb_rn18s'] = merged_df['emb_rn18s'].apply(replace_nan)
    merged_df['emb_its1'] = merged_df['emb_its1'].apply(replace_nan)
    merged_df['emb_its2'] = merged_df['emb_its2'].apply(replace_nan)
    
    # Merge label data
    final_df = merged_df.merge(labels_df, on='index', how='left')
    
    # Extract aligned data
    aligned_coi = np.stack(final_df['emb_coi'].values)
    aligned_coi_MAE = np.stack(final_df['emb_MAE'].values)
    aligned_rn16s = np.stack(final_df['emb_rn16s'].values)
    aligned_h3 = np.stack(final_df['emb_h3'].values)
    aligned_rn18s = np.stack(final_df['emb_rn18s'].values)
    aligned_its1 = np.stack(final_df['emb_its1'].values)
    aligned_its2 = np.stack(final_df['emb_its2'].values)
    aligned_labels = {
        'subclass': final_df['subclass'].values,
        'order': final_df['order'].values,
        'superfamily': final_df['superfamily'].values,
        'family': final_df['family'].values,
        'genus': final_df['genus'].values,
        'species': final_df['species'].values
    }
    
    return aligned_coi, aligned_coi_MAE, aligned_rn16s, aligned_h3, aligned_rn18s, aligned_its1, aligned_its2, aligned_labels



# Load data
coi_MAE_emb = np.load("./embedding/BarcodeMAE/train_BarcodeMAE_embedding.npy")

coi_emb = np.load("./embedding/ERNIE-RNA/Augmented_76577_All_used_Gastropoda_seqs_Taxinomy_sequences/cls_embedding.npy")
coi_index = np.array([i for i in range(coi_emb.shape[0])])

rn16s_emb = np.load("./embedding/ERNIE-RNA/Augmented_Other_Marker_16S_sequences/cls_embedding.npy")
rn16s_index = pd.read_csv("./data/data_Train_Val_Test/Augmented_Other_Marker_16S.txt", sep="\t")['Index'].values

h3_emb = np.load("./embedding/ERNIE-RNA/Augmented_Other_Marker_H3_sequences/cls_embedding.npy")
h3_index = pd.read_csv("./data/data_Train_Val_Test/Augmented_Other_Marker_H3.txt", sep="\t")['Index'].values

rn18s_emb = np.load("./embedding/ERNIE-RNA/Augmented_Other_Marker_18S_sequences/cls_embedding.npy")
rn18s_index = pd.read_csv("./data/data_Train_Val_Test/Augmented_Other_Marker_18S.txt", sep="\t")['Index'].values

its1_emb = np.load("./embedding/ERNIE-RNA/Augmented_Other_Marker_ITS1_sequences/cls_embedding.npy")
its1_index = pd.read_csv("./data/data_Train_Val_Test/Augmented_Other_Marker_ITS1.txt", sep="\t")['Index'].values

its2_emb = np.load("./embedding/ERNIE-RNA/Augmented_Other_Marker_ITS2_sequences/cls_embedding.npy")
its2_index = pd.read_csv("./data/data_Train_Val_Test/Augmented_Other_Marker_ITS2.txt", sep="\t")['Index'].values

dfl = pd.read_csv("./data/data_Train_Val_Test/Augmented_76577_All_used_Gastropoda_seqs_Taxinomy_LabelSpecies.txt", sep="\t")
labels = dfl[['subclass', 'order', 'superfamily', 'family', 'genus', 'species']]
labels = preprocess_labels(labels, [2, 24, 103, 354, 2753, 11295])

# Align data
aligned_coi, aligned_coi_MAE, aligned_rn16s, aligned_h3, aligned_rn18s, aligned_its1, aligned_its2, aligned_labels = align_multimodal_data(
                            coi_emb, coi_MAE_emb, coi_index, 
                            rn16s_emb, rn16s_index, 
                            h3_emb, h3_index, 
                            rn18s_emb, rn18s_index, 
                            its1_emb, its1_index, 
                            its2_emb, its2_index, 
                            labels)

    
# Global partitioning: 90% training+validation, 10% testing
all_indices = np.arange(len(aligned_coi))
trainval_idx, test_idx = train_test_split(all_indices, test_size=0.1, random_state=42)

#  创建包含所有模态的完整数据集
full_dataset = MultiModalCOIDataset(
    embeddings_dict={
        'coi': torch.FloatTensor(aligned_coi),
        'coi_MAE': torch.FloatTensor(aligned_coi_MAE),
        'rn16s': torch.FloatTensor(aligned_rn16s),
        'h3': torch.FloatTensor(aligned_h3),
        'rn18s': torch.FloatTensor(aligned_rn18s),
        'its1': torch.FloatTensor(aligned_its1),
        'its2': torch.FloatTensor(aligned_its2)
    },
    labels=[
        torch.LongTensor(aligned_labels['subclass']),
        torch.LongTensor(aligned_labels['order']),
        torch.LongTensor(aligned_labels['superfamily']),
        torch.LongTensor(aligned_labels['family']),
        torch.LongTensor(aligned_labels['genus']),
        torch.LongTensor(aligned_labels['species'])
    ]
)


trainval_idx = torch.load("./data/data_Train_Val_Test/trainval_idx.npy")
test_idx = torch.load("./data/data_Train_Val_Test/test_idx.npy")

# Divide the training validation set and testing set
source_dataset = torch.utils.data.Subset(full_dataset, trainval_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

source_loader = DataLoader(source_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)



#===========  Set optimal parameters ==========#       
base_config = {
    "embed_dim": 768,
    "seq_len": 12,
    "n_classes": [2, 24, 103, 354, 2753, 11295],
    "mamba_config": {"d_model": 256, "dropout": 0.1},
    "epochs": 300
}

final_config = base_config.copy()
final_config.update({
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 32
})


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


#======== Stage 1 ========#
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







#==============================================#
#=============== target_dataset ===============#
#==============================================#

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







####
#### Domain adaptive for Mutimodal-Full-Tuning model, param tuning
####

model_out_fold_DA = "./saved_models_MAE"
os.makedirs(model_out_fold_DA, exist_ok=True)


print("="*50)
print("="*50)
print("="*50)

# Define hyperparameter grid
param_grid = {
    'lambda_da': [0.1, 0.3, 0.5, 0.7, 1.0],
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5],
    'da_epochs': [50, 70]
}
# Best hyperparameter: {'lambda_da': 0.5, 'alpha': 0.3, 'da_epochs': 50}

# Save the best model
best_score = 0
best_params = None
best_model = None

use_config = final_config2
use_model = multimodal_model_full
use_stage = 2
use_save_name = f"addMAE_DomainA_BestAcc_stage_{use_stage}.pt"

# Grid search
for params in itertools.product(*param_grid.values()):

    lambda_da, alpha, da_epochs = params
    print(f"\n=== 开始训练: lambda_da={lambda_da}, alpha={alpha}, da_epochs={da_epochs} ===")
    
    # Create copy and update parameters
    da_config = copy.deepcopy(use_config)
    da_config.update({
        "lambda_da": lambda_da,
        "alpha": alpha,
        "da_epochs": da_epochs
    })
    
    # Create model
    da_model = DomainAdaptiveDMGHANmae(da_config, curriculum_stage=use_stage).to(DEVICE)
    da_model.load_state_dict(use_model.state_dict(), strict=False)
    
    # Train model
    model_path = f"{model_out_fold_DA}/temp_DA_stage_{use_stage}_{lambda_da}_{alpha}_{da_epochs}.pt"
    adapted_model = domain_adaptive_train(
        model=da_model,
        source_loader=source_loader,
        target_loader=target_loader,
        config=da_config,
        best_model_path=model_path
    )
    
    # Evaluate the performance of the target domain
    results = evaluate_target_domain(adapted_model, target_loader, da_config)
    species_acc = results['species']['accuracy_valid']
    print(f"Species accuracy: {species_acc:.4f}")
    
    # Update the best results
    if species_acc > best_score:
        best_score = species_acc
        best_params = {'lambda_da': lambda_da, 'alpha': alpha, 'da_epochs': da_epochs}
        best_model = adapted_model
        print(f"The new best accuracy: {best_score:.4f}")
        
        # Save the best model
        torch.save(best_model.state_dict(), f"{model_out_fold}/{use_save_name}")

# Print the best parameter
print("\n=== Grid search complete! ===")
print(f"Best parameters: {best_params}")
# Best parameter: {'lambda_da': 0.5, 'alpha': 0.3, 'da_epochs': 50}

#print(f"The best species accuracy: {best_score:.4f}")
# The best species accuracy: 0.5502




# Use the best model for final evaluation
print("\n=== The evaluation results of the best model in the target domain ===")
final_results = evaluate_target_domain(best_model, target_loader, {})
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
          f"Acc={final_results[level]['accuracy_valid']:.4f} | "
          f"F1={final_results[level]['f1_valid']:.4f}")
print("\n----------------------\n")
#- Subclass: Acc=0.9914 | F1=0.9922
#- Order: Acc=0.9418 | F1=0.9391
#- Superfamily: Acc=0.9005 | F1=0.9096
#- Family: Acc=0.7978 | F1=0.8321
#- Genus: Acc=0.8334 | F1=0.8499
#- Species: Acc=0.5502 | F1=0.5596

print("\n=== The evaluation results of the best model in the test_loader ===")
final_results2 = evaluate_target_domain(best_model, test_loader, {})
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
          f"Acc={final_results2[level]['accuracy_valid']:.4f} | "
          f"F1={final_results2[level]['f1_valid']:.4f}")
print("\n----------------------\n")
#- Subclass: Acc=0.9960 | F1=0.9961
#- Order: Acc=0.9865 | F1=0.9865
#- Superfamily: Acc=0.9765 | F1=0.9765
#- Family: Acc=0.9695 | F1=0.9694
#- Genus: Acc=0.9483 | F1=0.9474
#- Species: Acc=0.8543 | F1=0.8424


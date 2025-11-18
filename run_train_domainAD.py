

import numpy as np
import pandas as pd

import csv
import os
import torch
from sklearn.model_selection import train_test_split, KFold

from train import *
from test import *
from domainA import DomainAdaptiveDMGHAN, domain_adaptive_train, evaluate_target_domain

DEVICE = torch.device("cuda:0")


def preprocess_labels(lbs_df, n_classes_list):
    for i, level in enumerate(['subclass', 'order', 'superfamily', 'family', 'genus', 'species']):
        # Convert tags to 0-based integer encoding
        lbs_df[level] = lbs_df[level].astype('category').cat.codes
        # Verify label range
        max_label = lbs_df[level].max()
        min_label = lbs_df[level].min()
        n_class = n_classes_list[i]
        assert max_label < n_class and min_label >= 0, \
            f"Error: {level}level range[{min_label}, {max_label}]exceeds the categoriy number{n_class}"

    return [
        lbs_df['subclass'].values,
        lbs_df['order'].values,
        lbs_df['superfamily'].values,
        lbs_df['family'].values,
        lbs_df['genus'].values,
        lbs_df['species'].values
    ]

def replace_nan(x):
    # Determine whether it is a scalar NaN (non array)
    if isinstance(x, float) and pd.isna(x):
        return np.zeros((12, 768))
    else:
        return x


# -------------------- 0. Align multimodal sequence embedding --------------------
def align_multimodal_data(coi_emb, coi_index, 
                            rn16s_emb, rn16s_index, 
                            h3_emb, h3_index, 
                            rn18s_emb, rn18s_index, 
                            its1_emb, its1_index, 
                            its2_emb, its2_index, 
                            labels):
    # Merge the embedded data and index of each modality into a DataFrame
    coi_df = pd.DataFrame({'index': coi_index, 'emb': list(coi_emb)})
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
    
    # Merge data from other modalities with COI data
    merged_df = coi_df.merge(rn16s_df, on='index', how='left', suffixes=('_coi', '_rn16s'))
    merged_df = merged_df.merge(h3_df, on='index', how='left', suffixes=('_rn16s', '_h3'))
    merged_df = merged_df.merge(rn18s_df, on='index', how='left', suffixes=('_h3', '_rn18s'))
    merged_df = merged_df.merge(its1_df, on='index', how='left', suffixes=('_rn18s', '_its1'))
    merged_df = merged_df.merge(its2_df, on='index', how='left', suffixes=('_its1', '_its2'))
    
    # Handle missing values, such as filling with zeros
    merged_df['emb_rn16s'] = merged_df['emb_rn16s'].apply(replace_nan)
    merged_df['emb_h3'] = merged_df['emb_h3'].apply(replace_nan)
    merged_df['emb_rn18s'] = merged_df['emb_rn18s'].apply(replace_nan)
    merged_df['emb_its1'] = merged_df['emb_its1'].apply(replace_nan)
    merged_df['emb_its2'] = merged_df['emb_its2'].apply(replace_nan)
    
    # Merge labels
    final_df = merged_df.merge(labels_df, on='index', how='left')
    
    # Extract aligned data
    aligned_coi = np.stack(final_df['emb_coi'].values)
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
    
    return aligned_coi, aligned_rn16s, aligned_h3, aligned_rn18s, aligned_its1, aligned_its2, aligned_labels



#==============================================#
#=============== source_dataset ===============#
#==============================================#


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


# Align the data
aligned_coi, aligned_rn16s, aligned_h3, aligned_rn18s, aligned_its1, aligned_its2, aligned_labels = align_multimodal_data(
                            coi_emb, coi_index, 
                            rn16s_emb, rn16s_index, 
                            h3_emb, h3_index, 
                            rn18s_emb, rn18s_index, 
                            its1_emb, its1_index, 
                            its2_emb, its2_index, 
                            labels)

# Create a complete dataset that includes all modalities
full_dataset = MultiModalCOIDataset(
    embeddings_dict={
        'coi': torch.FloatTensor(aligned_coi),
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

#with open('./data/test_idx_output.csv', 'w', newline='') as file:
#    writer = csv.writer(file)
#    writer.writerow(['Sample_idx'])
#    for value in test_idx:
#        writer.writerow([value])


# Training-validation set and testing set
source_dataset = torch.utils.data.Subset(full_dataset, trainval_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

source_loader = DataLoader(source_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)





#===========  Set optimal operating parameters ==========#       
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
    "attention_heads": 16
})



## We have already pretrained the models
## You can download the pretrained models in DRYAD
model_pretrained_fold = "./pretrained_models"


#======== Stage 0 Training ========#
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
state_dict = torch.load(f"{model_pretrained_fold}/stage0_coi_only.pt", map_location=DEVICE)
trained_coi_model.load_state_dict(state_dict, strict=True)
trained_coi_model.eval()


#======== Stage 1 Training ========#
print("\n=== Stage 1: Multimodal Fusion Training ===")
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
state_dict = torch.load(f"{model_pretrained_fold}/stage1_multimodal_fusion_BestMeanAcc.pt", map_location=DEVICE)
trained_multimodal.load_state_dict(state_dict, strict=True)
trained_multimodal.eval()


#======== Stage 2 Training ========#
print("\n=== Stage 2: Full Fine-tuning ===")
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
state_dict = torch.load(f"{model_pretrained_fold}/stage2_multimodal_fulltuning_BestMeanAcc.pt", map_location=DEVICE)
multimodal_model_full.load_state_dict(state_dict, strict=True)
multimodal_model_full.eval()






#==============================================#
#=============== target_dataset ===============#
#==============================================#

#### Independnet Datasets
tester = FixedMultiModalTester(
        coi_model=trained_coi_model,
        multimodal_model=trained_multimodal,
        class_counts=[2, 24, 103, 354, 2753, 11295],
        device=DEVICE
    )

#### GenBank From 20250101-20250523
label_arrays_1 = tester.load_labels(
    "./data/data_Independent/GenBank_From20250101/Independent_Gastropoda_seqs_n_1927_EDITED_LabelSpecies.txt"
)
coi_emb_1 = np.load("./embedding/ERNIE-RNA/Independent1_GenBank_From20250101_n_1927_EDITED_sequences/cls_embedding.npy")
print(coi_emb_1.shape)

#### BOLD_Mar2025 Gastropoda, real seqs
label_arrays_2 = tester.load_labels(
    "./data/data_Independent/BOLD_Mar2025_Gastropoda/EDITED_Independent2_BOLD_Taxonomy_label_is_the_SAME_with_training_LabelsSpecies.txt"
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






#=========================================================#
#=== Application domain adaptation for target_dataset ====#
#=========================================================#

model_out_fold = "./saved_models"
os.makedirs(model_out_fold, exist_ok=True)


#-------  COI only + domain adaptative -------#

# Domain adaptive training configuration
da_config0 = {
    **final_config0,
    "da_epochs": 70,
    "lambda_da": 1.0,
    "alpha": 0.1
}
    
# Create domain adaptive models
da_model0 = DomainAdaptiveDMGHAN(da_config0, curriculum_stage=0).to(DEVICE)
state_dict = torch.load(f"{model_out_fold}/DomainA_BestAcc_stage_0.pt", map_location=DEVICE)
da_model0.load_state_dict(state_dict, strict=True)
da_model0.eval()

final_results = test_model(da_model0, target_loader, DEVICE)
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
          f"Acc={final_results[level]['accuracy']:.4f} | "
          f"F1={final_results[level]['f1']:.4f}")


target_metrics2_vlid = test_model_ValidSample(da_model0, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics2_vlid[level]['accuracy_valid']:.4f} | "
        f"F1={target_metrics2_vlid[level]['f1_valid']:.4f} | "
        f"Counts={target_metrics2_vlid[level]['count_valid']}")


print("\n=== The evaluation results of the best model for test-loader ===")
final_results2 = evaluate_target_domain(da_model0, test_loader, da_config0)
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
          f"Acc={final_results2[level]['accuracy_valid']:.4f} | "
          f"F1={final_results2[level]['f1_valid']:.4f}")




#-------  Mutilmodal Fusion + domain adaptative -------#

da_config1 = {
    **final_config1,
    "da_epochs": 50,
    "lambda_da": 1.0,
    "alpha": 0.5
}
    
da_model1 = DomainAdaptiveDMGHAN(da_config1, curriculum_stage=1).to(DEVICE)
state_dict = torch.load(f"{model_out_fold}/DomainA_BestAcc_stage_1.pt", map_location=DEVICE)
da_model1.load_state_dict(state_dict, strict=True)
da_model1.eval()

final_results = test_model(da_model1, target_loader, DEVICE)
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
          f"Acc={final_results[level]['accuracy']:.4f} | "
          f"F1={final_results[level]['f1']:.4f}")


target_metrics2_vlid = test_model_ValidSample(da_model1, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics2_vlid[level]['accuracy_valid']:.4f} | "
        f"F1={target_metrics2_vlid[level]['f1_valid']:.4f} | "
        f"Counts={target_metrics2_vlid[level]['count_valid']}")


print("\n=== The evaluation results of the best model for test-loader ===")
final_results2 = evaluate_target_domain(da_model1, test_loader, da_config1)
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
          f"Acc={final_results2[level]['accuracy_valid']:.4f} | "
          f"F1={final_results2[level]['f1_valid']:.4f}")




#-------  Mutilmodal Full-FineTuning + domain adaptative -------#

da_config2 = {
    **final_config2,
    "da_epochs": 50,
    "lambda_da": 1.0,
    "alpha": 0.1
}
    
da_model2 = DomainAdaptiveDMGHAN(da_config2, curriculum_stage=2).to(DEVICE)
state_dict = torch.load(f"{model_out_fold}/DomainA_BestAcc_stage_2.pt", map_location=DEVICE)
da_model2.load_state_dict(state_dict, strict=True)
da_model2.eval()

final_results = test_model(da_model2, target_loader, DEVICE)
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
          f"Acc={final_results[level]['accuracy']:.4f} | "
          f"F1={final_results[level]['f1']:.4f}")


target_metrics2_vlid = test_model_ValidSample(da_model2, target_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={target_metrics2_vlid[level]['accuracy_valid']:.4f} | "
        f"F1={target_metrics2_vlid[level]['f1_valid']:.4f} | "
        f"Counts={target_metrics2_vlid[level]['count_valid']}")


print("\n=== The evaluation results of the best model for test-loader ===")
final_results2 = evaluate_target_domain(da_model2, test_loader, da_config2)
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
          f"Acc={final_results2[level]['accuracy_valid']:.4f} | "
          f"F1={final_results2[level]['f1_valid']:.4f}")








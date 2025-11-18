
import numpy as np
import pandas as pd

import os
import torch
from sklearn.model_selection import train_test_split, KFold

from addMAEtrain import *
from addMAEparam import *
import itertools

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
    # Determine whether it is a scalar NaN (non array)
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
    # Merge the embedded data and index of each modality into a DataFrame
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

# Align the data
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

# Create a complete dataset that includes all modalities
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

# Divide the training-validation set and testing set
trainval_dataset = torch.utils.data.Subset(full_dataset, trainval_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

#torch.save(trainval_idx, "data/trainval_idx.npy")
#torch.save(test_idx, "data/test_idx.npy")

# Organize data format
full_data = {
    'coi': torch.stack([trainval_dataset[i]['embeds']['coi'] for i in range(len(trainval_dataset))]),
    'coi_MAE': torch.stack([trainval_dataset[i]['embeds']['coi_MAE'] for i in range(len(trainval_dataset))]),
    'rn16s': torch.stack([trainval_dataset[i]['embeds']['rn16s'] for i in range(len(trainval_dataset))]),
    'h3': torch.stack([trainval_dataset[i]['embeds']['h3'] for i in range(len(trainval_dataset))]),
    'rn18s': torch.stack([trainval_dataset[i]['embeds']['rn18s'] for i in range(len(trainval_dataset))]),
    'its1': torch.stack([trainval_dataset[i]['embeds']['its1'] for i in range(len(trainval_dataset))]),
    'its2': torch.stack([trainval_dataset[i]['embeds']['its2'] for i in range(len(trainval_dataset))]),
    'labels': [
        torch.stack([trainval_dataset[i]['labels']['subclass'] for i in range(len(trainval_dataset))]),
        torch.stack([trainval_dataset[i]['labels']['order'] for i in range(len(trainval_dataset))]),
        torch.stack([trainval_dataset[i]['labels']['superfamily'] for i in range(len(trainval_dataset))]),
        torch.stack([trainval_dataset[i]['labels']['family'] for i in range(len(trainval_dataset))]),
        torch.stack([trainval_dataset[i]['labels']['genus'] for i in range(len(trainval_dataset))]),
        torch.stack([trainval_dataset[i]['labels']['species'] for i in range(len(trainval_dataset))])
    ]
}




#===========  Set optimal operating parameters ==========#       
base_config = {
    "embed_dim": 768,
    "seq_len": 12,
    "n_classes": [2, 24, 103, 354, 2753, 11295],
    "mamba_config": {"d_model": 256, "dropout": 0.1},
    "epochs": 300
}

# -------------------- Hyperparameter grid configuration --------------------
#hyperparam_grid = {
#    "learning_rate": [1e-4],
#    "weight_decay": [0.01, 0.005],
#    "loss_weights": [
#        [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
#        [1.2, 1.0, 0.8, 0.6, 0.4, 0.2],
#        [1.8, 1.5, 1.2, 0.9, 0.6, 0.3]
#    ],
#    "batch_size": [64],
#    "attention_heads": [32, 16]
#}
#best_met = grid_search_train(full_data, base_config, hyperparam_grid)

final_config = base_config.copy()
final_config.update({
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    #"loss_weights": [1.2, 1.0, 0.8, 0.6, 0.4, 0.2],
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 16
})


model_out_fold = "./saved_models_MAE"
os.makedirs(model_out_fold, exist_ok=True)

## We have already pretrained the models
## You can download the pretrained models in DRYAD
model_pretrained_fold = "./pretrained_models"

#======== Stage 0 training ========#
print("\n=== Stage 0: COI Only Training ===")
# Initialize and train
coi_model = CurriculumDMGHANmae(final_config, curriculum_stage=0).to(DEVICE)
# Train
trained_coi_model = train_coi_only(
    model=coi_model,
    full_data=full_data,
    config=final_config,
    best_model_path=f"{model_out_fold}/stage0_coi_only_BestAcc.pt"
)



#-------- If there is already a trained model, you can directly load it with the following code --------#
# Load coi model
#final_config0 = base_config.copy()
#final_config0.update({
#    "learning_rate": 0.0001,
#    "weight_decay": 0.01,
#    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
#    "batch_size": 64,
#    "attention_heads": 16,
#    "curriculum_stage": 0
#})
#trained_coi_model = CurriculumDMGHANmae(final_config0).to(DEVICE)
#state_dict = torch.load(f"{model_pretrained_fold}/stage0_coi_only_BestAcc.pt", map_location=DEVICE)
#trained_coi_model.load_state_dict(state_dict, strict=True)
#trained_coi_model.eval()



#======== Stage 1 training ========#
#print("\n=== Stage 1: Multimodal Fusion Training ===")
multimodal_model = CurriculumDMGHANmae(final_config, curriculum_stage=1).to(DEVICE)
multimodal_model.load_state_dict(trained_coi_model.state_dict(), strict=False)
# Freeze COI related parameters
for name, param in multimodal_model.named_parameters():
    if 'coi' in name or 'coi_MAE' in name:
        param.requires_grad_(False)
# Perform multimodal training
trained_multimodal = train_multimodal(
    model=multimodal_model,
    full_data=full_data,
    config=final_config,
    best_model_path=f"{model_out_fold}/stage1_multimodal_fusion_BestMeanAcc.pt"
)

#-------- If there is already a trained model, you can directly load it with the following code --------#
# Load fusion model
#final_config1 = base_config.copy()
#final_config1.update({
#    "learning_rate": 0.0001,
#    "weight_decay": 0.01,
#    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
#    "batch_size": 64,
#    "attention_heads": 16,
#    "curriculum_stage": 1
#})
#trained_multimodal = CurriculumDMGHAN(final_config1, curriculum_stage=1).to(DEVICE)
#state_dict = torch.load(f"{model_pretrained_fold}/stage1_multimodal_fusion_BestMeanAcc.pt", map_location=DEVICE)
#trained_multimodal.load_state_dict(state_dict, strict=True)
#trained_multimodal.eval()



#======== Stage 2 training ========#
print("\n=== Stage 2: Full Fine-tuning ===")
for param in trained_multimodal.parameters():
    param.requires_grad_(True)
    
multimodal_model_full = train_multimodal(
    model=trained_multimodal, 
    full_data=full_data, 
    config=final_config,
    best_model_path=f"{model_out_fold}/stage2_multimodal_full_BestMeanAcc.pt"
    )

#-------- If there is already a trained model, you can directly load it with the following code --------#
# Load full-tuned model
#final_config2 = base_config.copy()
#final_config2.update({
#    "learning_rate": 0.0001,
#    "weight_decay": 0.01,
#    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
#    "batch_size": 64,
#    "attention_heads": 16,
#    "curriculum_stage": 2
#})
#multimodal_model_full = CurriculumDMGHAN(final_config2, curriculum_stage=2).to(DEVICE)
#state_dict = torch.load(f"{model_pretrained_fold}/stage2_multimodal_full_BestMeanAcc.pt", map_location=DEVICE)
#multimodal_model_full.load_state_dict(state_dict, strict=True)
#multimodal_model_full.eval()





#======== Final test set evaluation ========#
print("\n=== Final Testing ===")
test_loader = DataLoader(
    test_dataset,
    batch_size=final_config["batch_size"],
    pin_memory=True
)


print("\n=== COI Only - model")
test_metrics1 = test_model(trained_coi_model, test_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={test_metrics1[level]['accuracy']:.4f} | "
        f"F1={test_metrics1[level]['f1']:.4f}")


print("\n=== Mutilemodal Fusion - model")
test_metrics = test_model(trained_multimodal, test_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={test_metrics[level]['accuracy']:.4f} | "
        f"F1={test_metrics[level]['f1']:.4f}")
              

print("\n=== Mutilemodal Full-tuning - model")
test_metrics = test_model(multimodal_model_full, test_loader, DEVICE)
print("Test Metrics:")
for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
    print(f"- {level.capitalize()}: "
        f"Acc={test_metrics[level]['accuracy']:.4f} | "
        f"F1={test_metrics[level]['f1']:.4f}")









    

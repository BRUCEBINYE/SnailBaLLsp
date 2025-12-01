# fine_tune_casedata.py
import numpy as np
import pandas as pd
import os
import torch
import argparse
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from model import *
from dataset import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_labels_casedata(lbs_df, n_classes_list):
    """Taxonomic labels preprocessing"""
    for i, level in enumerate(['subclass', 'order', 'superfamily', 'family', 'genus', 'species']):
        # Convert labels to 0-based integer encoding
        lbs_df[level] = lbs_df[level].astype('category').cat.codes
        # Validate label range
        max_label = lbs_df[level].max()
        min_label = lbs_df[level].min()
        n_class = n_classes_list[i]
        assert max_label < n_class and min_label >= 0, \
            f"Error: {level} label range [{min_label}, {max_label}] exceeds number of classes {n_class}"
    
    return [
        lbs_df['subclass'].values,
        lbs_df['order'].values,
        lbs_df['superfamily'].values,
        lbs_df['family'].values,
        lbs_df['genus'].values,
        lbs_df['species'].values
    ]

def replace_nan(x):
    """Handle NaN values"""
    if isinstance(x, float) and pd.isna(x):
        return np.zeros((12, 768))
    else:
        return x

def align_multimodal_data_casedata(coi_emb, coi_MAE_emb, coi_index, 
                                  rn16s_emb, rn16s_index, 
                                  h3_emb, h3_index, 
                                  rn18s_emb, rn18s_index, 
                                  its1_emb, its1_index, 
                                  its2_emb, its2_index, 
                                  labels):
    """Multimodal data alignment"""
    # Merge each modality's embedding data with index into DataFrame
    coi_df = pd.DataFrame({'index': coi_index, 'emb': list(coi_emb), 'emb_MAE': list(coi_MAE_emb)})
    rn16s_df = pd.DataFrame({'index': rn16s_index, 'emb': list(rn16s_emb)})
    h3_df = pd.DataFrame({'index': h3_index, 'emb': list(h3_emb)})
    rn18s_df = pd.DataFrame({'index': rn18s_index, 'emb': list(rn18s_emb)})
    its1_df = pd.DataFrame({'index': its1_index, 'emb': list(its1_emb)})
    its2_df = pd.DataFrame({'index': its2_index, 'emb': list(its2_emb)})

    labels_df = pd.DataFrame({
        'index': coi_index,
        'subclass': labels[0],
        'order': labels[1],
        'superfamily': labels[2],
        'family': labels[3],
        'genus': labels[4],
        'species': labels[5]
    })
    
    # Merge data
    merged_df = coi_df.merge(rn16s_df, on='index', how='left', suffixes=('_coi', '_rn16s'))
    merged_df = merged_df.merge(h3_df, on='index', how='left', suffixes=('_rn16s', '_h3'))
    merged_df = merged_df.merge(rn18s_df, on='index', how='left', suffixes=('_h3', '_rn18s'))
    merged_df = merged_df.merge(its1_df, on='index', how='left', suffixes=('_rn18s', '_its1'))
    merged_df = merged_df.merge(its2_df, on='index', how='left', suffixes=('_its1', '_its2'))
    
    # Handle missing values
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

def load_casedata_data(data_config, data_task):
    """Load taxonomic data"""
    print(f"=== Loading {data_task} Data ===")
    
    # Load COI data
    coi_emb = np.load(data_config['coi_emb'])
    coi_MAE_emb = np.load(data_config['coi_MAE_emb'])
    coi_index_df = pd.read_csv(data_config['coi_index'], sep=',')
    coi_index = coi_index_df['Index'].values
    
    # Load other modality data
    rn16s_emb = np.load(data_config['rn16s_emb']) if os.path.exists(data_config['rn16s_emb']) else None
    rn16s_index = pd.read_csv(data_config['rn16s_index'], sep=',')['Index'].values if os.path.exists(data_config['rn16s_index']) else np.array([])
    
    h3_emb = np.load(data_config['h3_emb']) if os.path.exists(data_config['h3_emb']) else None
    h3_index = pd.read_csv(data_config['h3_index'], sep=',')['Index'].values if os.path.exists(data_config['h3_index']) else np.array([])
    
    rn18s_emb = np.load(data_config['rn18s_emb']) if os.path.exists(data_config['rn18s_emb']) else None
    rn18s_index = pd.read_csv(data_config['rn18s_index'], sep=',')['Index'].values if os.path.exists(data_config['rn18s_index']) else np.array([])
    
    its1_emb = np.load(data_config['its1_emb']) if os.path.exists(data_config['its1_emb']) else None
    its1_index = pd.read_csv(data_config['its1_index'], sep=',')['Index'].values if os.path.exists(data_config['its1_index']) else np.array([])
    
    its2_emb = np.load(data_config['its2_emb']) if os.path.exists(data_config['its2_emb']) else None
    its2_index = pd.read_csv(data_config['its2_index'], sep=',')['Index'].values if os.path.exists(data_config['its2_index']) else np.array([])
    
    # Load labels
    dfl = pd.read_csv(data_config['labels'], sep=',')
    labels = dfl[['subclass', 'order', 'superfamily', 'family', 'genus', 'species']]
    
    # Taxonomic class counts - please modify according to actual situation
    casedata_n_classes = data_config.get('n_classes', [3, 17, 33, 71, 617, 3060])  # Example values
    
    labels = preprocess_labels_casedata(labels, casedata_n_classes)
    
    print(f"{data_task} Data Statistics:")
    print(f"- COI samples: {len(coi_emb)}")
    print(f"- 16S samples: {len(rn16s_emb) if rn16s_emb is not None else 0}")
    print(f"- H3 samples: {len(h3_emb) if h3_emb is not None else 0}")
    print(f"- 18S samples: {len(rn18s_emb) if rn18s_emb is not None else 0}")
    print(f"- ITS1 samples: {len(its1_emb) if its1_emb is not None else 0}")
    print(f"- ITS2 samples: {len(its2_emb) if its2_emb is not None else 0}")
    
    return coi_emb, coi_MAE_emb, coi_index, rn16s_emb, rn16s_index, h3_emb, h3_index, rn18s_emb, rn18s_index, its1_emb, its1_index, its2_emb, its2_index, labels, casedata_n_classes

def fine_tune_model_on_casedata(model_path, model_name, casedata_data, output_dir, fine_tune_config, data_task):
    """Fine-tune model on taxonomic data"""
    print(f"\n{'='*60}")
    print(f"Fine-tuning model on {data_task} data: {model_name}")
    print(f"{'='*60}")
    
    # Unpack data
    (coi_emb, coi_MAE_emb, coi_index, rn16s_emb, rn16s_index, 
     h3_emb, h3_index, rn18s_emb, rn18s_index, its1_emb, its1_index, 
     its2_emb, its2_index, labels, casedata_n_classes) = casedata_data
    
    # Align data
    aligned_coi, aligned_coi_MAE, aligned_rn16s, aligned_h3, aligned_rn18s, aligned_its1, aligned_its2, aligned_labels = align_multimodal_data_casedata(
        coi_emb, coi_MAE_emb, coi_index, 
        rn16s_emb, rn16s_index, 
        h3_emb, h3_index, 
        rn18s_emb, rn18s_index, 
        its1_emb, its1_index, 
        its2_emb, its2_index, 
        labels
    )
    
    # Global split: 80% train+validation, 20% test
    all_indices = np.arange(len(aligned_coi))
    trainval_idx, test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    
    # Create full dataset
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
    
    # Split train-validation set and test set
    trainval_dataset = torch.utils.data.Subset(full_dataset, trainval_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    
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
    
    # Update fine_tune_config with taxonomic class counts
    fine_tune_config["n_classes"] = casedata_n_classes  # Use taxonomic class counts
    
    # Create output directory
    model_output_dir = output_dir
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Load pre-trained model
    print(f"Loading pre-trained model: {model_path}")
    
    if "BarcodeMAE" in model_path:
        from domainA import DomainAdaptiveDMGHANmae
        model = DomainAdaptiveDMGHANmae(fine_tune_config, curriculum_stage=2).to(DEVICE)
    else:
        from domainA import DomainAdaptiveDMGHAN
        model = DomainAdaptiveDMGHAN(fine_tune_config, curriculum_stage=2).to(DEVICE)
    
    # Load pre-trained weights
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # Handle class count mismatch - only load matching weights
    model_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() 
                      if k in model_state_dict and model_state_dict[k].shape == v.shape}
    
    # Load matching weights
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict, strict=False)
    
    print(f"Successfully loaded {len(pretrained_dict)}/{len(state_dict)} parameters")
    
    # Fine-tuning training (direct multimodal training)
    print("Starting fine-tuning training...")
    best_model_path = os.path.join(model_output_dir, f"fine_tuned_{model_name}.pt")
    
    # Unfreeze all parameters for fine-tuning
    for param in model.parameters():
        param.requires_grad_(True)
    
    if "BarcodeMAE" in model_path:
        from addMAEtrain import train_coi_only, train_multimodal, evaluate_model, evaluate_loss#, MultiModalCOIDataset
        # Use multimodal training function for fine-tuning
        fine_tuned_model = train_multimodal(
            model=model,
            full_data=full_data,
            config=fine_tune_config,
            best_model_path=best_model_path
        )

        # Evaluate on test set
        print(f"Evaluating fine-tuned model on {data_task} test set...")
        test_loader = DataLoader(test_dataset, batch_size=fine_tune_config["batch_size"], pin_memory=True)
        test_metrics = evaluate_model(fine_tuned_model, test_loader, DEVICE)
    
        print(f"\n{model_name} performance on {data_task} test set:")
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            print(f"- {level.capitalize()}: "
                  f"Acc={test_metrics[level]['accuracy']:.4f} | "
                  f"F1={test_metrics[level]['f1']:.4f}")
    
        # Save evaluation results
        results_file = os.path.join(model_output_dir, f"fine_tune_results_{model_name}.txt")
        with open(results_file, 'w') as f:
            f.write(f"Fine-tuned model: {model_name}\n")
            f.write(f"Pre-trained model: {model_path}\n")
            f.write(f"{data_task} data volume: {len(aligned_coi)} samples\n")
            f.write(f"Test set performance:\n")
            for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
                f.write(f"{level.capitalize()}: Acc={test_metrics[level]['accuracy']:.4f} | F1={test_metrics[level]['f1']:.4f}\n")

    else:
        from train import train_coi_only, train_multimodal, evaluate_model, evaluate_loss#, MultiModalCOIDataset
        # Use multimodal training function for fine-tuning
        fine_tuned_model = train_multimodal(
            model=model,
            full_data=full_data,
            config=fine_tune_config,
            best_model_path=best_model_path
        )
        # Evaluate on test set
        print(f"Evaluating fine-tuned model on {data_task} test set...")
        test_loader = DataLoader(test_dataset, batch_size=fine_tune_config["batch_size"], pin_memory=True)
        test_metrics = evaluate_model(fine_tuned_model, test_loader, DEVICE)
    
        print(f"\n{model_name} performance on {data_task} test set:")
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            print(f"- {level.capitalize()}: "
                  f"Acc={test_metrics[level]['accuracy']:.4f} | "
                  f"F1={test_metrics[level]['f1']:.4f}")
    
        # Save evaluation results
        results_file = os.path.join(model_output_dir, f"fine_tune_results_{model_name}.txt")
        with open(results_file, 'w') as f:
            f.write(f"Fine-tuned model: {model_name}\n")
            f.write(f"Pre-trained model: {model_path}\n")
            f.write(f"{data_task} data volume: {len(aligned_coi)} samples\n")
            f.write(f"Test set performance:\n")
            for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
                f.write(f"{level.capitalize()}: Acc={test_metrics[level]['accuracy']:.4f} | F1={test_metrics[level]['f1']:.4f}\n")

    return test_metrics, fine_tuned_model


def main():
    """Main function"""
    print("=== Taxonomic Data Fine-tuning Experiment ===")
    print("Purpose: Validate model framework's transferability to new taxa")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune models on taxonomic data')
    
    # Data configuration
    parser.add_argument('--data_config', type=str, required=True, 
                       help='Path to JSON file containing data configuration')
    
    # Fine-tuning configuration
    parser.add_argument('--fine_tune_config', type=str, required=True,
                       help='Path to JSON file containing fine-tuning configuration')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_models',
                       help='Output directory for fine-tuned models')
    
    # Pre-trained model (only one required)
    parser.add_argument('--pretrained_model', type=str, required=True,
                       help='Path to pre-trained model for fine-tuning')
    
    # Data task name
    parser.add_argument('--data_task', type=str, default='Bivalve',
                       help='Name of the taxonomic data task (e.g., Bivalve, Gastropod, etc.)')
    
    # Model name (optional)
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name for the fine-tuned model (default: derived from pretrained_model path)')
    
    args = parser.parse_args()
    
    # Load configurations from JSON files
    with open(args.data_config, 'r') as f:
        data_config = json.load(f)
    
    with open(args.fine_tune_config, 'r') as f:
        fine_tune_config = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model name
    if args.model_name is None:
        # Extract model name from pretrained_model path
        model_name = os.path.basename(args.pretrained_model)
        if model_name.endswith('.pt'):
            model_name = model_name[:-3]
    else:
        model_name = args.model_name
    
    # Load taxonomic data
    casedata_data = load_casedata_data(data_config, args.data_task)
    
    # Fine-tune the pre-trained model
    if not os.path.exists(args.pretrained_model):
        print(f"Error: Pre-trained model does not exist: {args.pretrained_model}")
        return
    
    try:
        test_metrics, fine_tuned_model = fine_tune_model_on_casedata(
            args.pretrained_model, model_name, casedata_data, args.output_dir, fine_tune_config, args.data_task
        )
        
        # Generate test set report
        print(f"\n{'='*60}")
        print("Fine-tuning Experiment Summary Report")
        print(f"{'='*60}")
        
        print(f"Data Task: {args.data_task}")
        print(f"Model: {model_name}")
        print(f"Pre-trained model: {args.pretrained_model}")
        print(f"{args.data_task} data volume: {casedata_data[0].shape[0]} samples")
        print(f"Test set performance:")
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            print(f"- {level.capitalize()}: Acc={test_metrics[level]['accuracy']:.4f} | F1={test_metrics[level]['f1']:.4f}")
        
        # Save summary report
        summary_file = os.path.join(args.output_dir, f"fine_tune_summary_{model_name}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"{args.data_task} Fine-tuning Experiment Summary Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Data Task: {args.data_task}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Pre-trained model: {args.pretrained_model}\n")
            f.write(f"{args.data_task} data volume: {casedata_data[0].shape[0]} samples\n")
            f.write(f"Test set performance:\n")
            for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
                f.write(f"{level.capitalize()}: Acc={test_metrics[level]['accuracy']:.4f} | F1={test_metrics[level]['f1']:.4f}\n")
        
        print(f"Summary report saved to: {summary_file}")
            
    except Exception as e:
        print(f"Model fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
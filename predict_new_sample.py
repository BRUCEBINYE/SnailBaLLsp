
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from test import FixedMultiModalTester

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model configuration
base_config = {
    "embed_dim": 768,
    "seq_len": 12,
    "n_classes": [2, 24, 103, 354, 2753, 11295],
    "mamba_config": {"d_model": 256, "dropout": 0.1},
    "learning_rate": 0.0001,
    "weight_decay": 0.01,
    "loss_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
    "batch_size": 64,
    "attention_heads": 32,
}

class LabelMapper:
    """Mapper from classification indices to names"""
    def __init__(self, label_file_path):
        """
        Initialize label mapper
        :param label_file_path: Label file path
        """
        self.label_file_path = label_file_path
        self.label_maps = self._load_label_mappings()
    
    def _load_label_mappings(self):
        """Load label mapping relationships"""
        try:
            # Read label file
            df = pd.read_csv(self.label_file_path, sep='\t')
            
            # Get unique labels for each level and create mappings
            label_maps = {}
            levels = ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
            text_levels = ['Subclass', 'Order', 'Superfamily', 'Family', 'Genus', 'Species']
            
            for level, text_level in zip(levels, text_levels):
                if level in df.columns and text_level in df.columns:
                    # Get unique numeric encodings and corresponding text names
                    unique_pairs = df[[level, text_level]].drop_duplicates()
                    
                    # Create mapping from numeric encoding to text name
                    level_map = {}
                    for _, row in unique_pairs.iterrows():
                        num_value = int(row[level]) if not pd.isna(row[level]) else -1
                        text_value = row[text_level] if not pd.isna(row[text_level]) else "Unknown"
                        level_map[num_value] = text_value
                    
                    label_maps[level] = level_map
                    print(f"Loaded {level} level label mapping: {len(level_map)} categories")
                else:
                    print(f"Warning: {level} or {text_level} column not found in label file")
                    label_maps[level] = {}
            
            return label_maps
            
        except Exception as e:
            print(f"Failed to load label mapping: {e}")
            # Return empty mapping dictionary
            return {level: {} for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']}
    
    def map_predictions(self, predictions):
        """
        Map predicted indices back to classification names
        :param predictions: Prediction results dictionary
        :return: Prediction results containing classification names
        """
        mapped_predictions = {}
        
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            if level in predictions and level in self.label_maps:
                level_predictions = predictions[level]['predictions']
                level_map = self.label_maps[level]
                
                # Map prediction results
                mapped_names = []
                for pred_idx in level_predictions:
                    if pred_idx in level_map:
                        mapped_names.append(level_map[pred_idx])
                    else:
                        mapped_names.append(f"Unknown_category_{pred_idx}")
                
                mapped_predictions[level] = {
                    'predictions': predictions[level]['predictions'],
                    'names': np.array(mapped_names)
                }
                
                # If probability information exists, also keep it
                if 'probabilities' in predictions[level]:
                    mapped_predictions[level]['probabilities'] = predictions[level]['probabilities']
                
                # Map top-k prediction results
                if 'top_k_predictions' in predictions[level]:
                    top_k_preds = predictions[level]['top_k_predictions']
                    top_k_names = []
                    
                    # Map top-k predictions for each sample
                    for sample_preds in top_k_preds:
                        sample_names = []
                        for pred_idx in sample_preds:
                            if pred_idx in level_map:
                                sample_names.append(level_map[pred_idx])
                            else:
                                sample_names.append(f"Unknown_category_{pred_idx}")
                        top_k_names.append(sample_names)
                    
                    mapped_predictions[level]['top_k_predictions'] = predictions[level]['top_k_predictions']
                    mapped_predictions[level]['top_k_names'] = np.array(top_k_names)
                
                if 'top_k_probabilities' in predictions[level]:
                    mapped_predictions[level]['top_k_probabilities'] = predictions[level]['top_k_probabilities']
            else:
                print(f"Warning: Unable to map {level} level prediction results")
                mapped_predictions[level] = predictions[level]
        
        return mapped_predictions

class FlexibleSamplePredictor:
    def __init__(self, model_path, config, curriculum_stage=0, device=DEVICE, label_mapper=None):
        self.device = device
        self.config = config
        self.curriculum_stage = curriculum_stage
        self.label_mapper = label_mapper
        
        # Load different model classes based on model name
        if "DomainA" in model_path:
            from domainA import DomainAdaptiveDMGHAN
            self.model_class = DomainAdaptiveDMGHAN
        else:
            from model import CurriculumDMGHAN
            self.model_class = CurriculumDMGHAN
        
        # Load model
        self.model = self.model_class(config, curriculum_stage=curriculum_stage).to(device)
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        
        # Define all supported modalities and their shapes
        self.supported_modals = {
            'coi': (12, 768),           # COI (ERNIE-RNA)
            'rn16s': (12, 768),         # 16S rRNA
            'h3': (12, 768),            # H3
            'rn18s': (12, 768),         # 18S rRNA
            'its1': (12, 768),          # ITS1
            'its2': (12, 768)           # ITS2
        }
        
        print(f"Model loaded successfully: {model_path}")
        print(f"Model type: {self.model_class.__name__}, stage: {curriculum_stage}")
        print(f"Supported modalities: {list(self.supported_modals.keys())}")

    def validate_modal_data(self, modal_name, data):
        """Validate modal data shape"""
        expected_shape = self.supported_modals[modal_name]
        if len(data.shape) != len(expected_shape) + 1:
            raise ValueError(f"{modal_name} data shape error: expected {(-1,) + expected_shape}, got {data.shape}")
        
        for i, (actual, expected) in enumerate(zip(data.shape[1:], expected_shape)):
            if actual != expected:
                raise ValueError(f"{modal_name} data shape error: dimension {i+1} expected {expected}, got {actual}")
        
        return True

    def align_samples_by_index(self, embeddings_dict, index_dict, base_modal='coi'):
        """
        Align samples from different modalities based on index
        embeddings_dict: Dictionary containing embedding data for each modality
        index_dict: Dictionary containing index data for each modality
        base_modal: Base modality, other modalities will be aligned with this one
        """
        if base_modal not in embeddings_dict or base_modal not in index_dict:
            raise ValueError(f"Base modality {base_modal} does not exist")
        
        # Get base modality indices and embeddings
        base_indices = index_dict[base_modal]
        base_embeddings = embeddings_dict[base_modal]
        n_base_samples = len(base_indices)
        
        print(f"Base modality {base_modal} has {n_base_samples} samples")
        
        # Create index mapping
        base_index_map = {idx: i for i, idx in enumerate(base_indices)}
        
        # Aligned embeddings dictionary
        aligned_embeddings = {base_modal: base_embeddings}
        
        # Align other modalities
        for modal in embeddings_dict.keys():
            if modal == base_modal:
                continue
                
            if modal not in index_dict:
                print(f"Warning: {modal} has no index data, skipping")
                continue
                
            modal_indices = index_dict[modal]
            modal_embeddings = embeddings_dict[modal]
            
            # Create aligned embedding array
            aligned_modal_embeddings = np.zeros((n_base_samples, *self.supported_modals[modal]))
            
            # Match samples
            matched_count = 0
            for i, base_idx in enumerate(base_indices):
                if base_idx in modal_indices:
                    # Find matching sample
                    modal_idx_pos = np.where(modal_indices == base_idx)[0]
                    if len(modal_idx_pos) > 0:
                        aligned_modal_embeddings[i] = modal_embeddings[modal_idx_pos[0]]
                        matched_count += 1
                # If no match, keep as zero (automatically filled)
            
            aligned_embeddings[modal] = aligned_modal_embeddings
            print(f"{modal} modality: {matched_count}/{n_base_samples} samples matched")
        
        return aligned_embeddings

    def prepare_samples(self, modal_data_dict):
        """
        Prepare sample data, support any combination of modalities
        modal_data_dict: Dictionary, keys are modality names, values are corresponding embedding data
        """
        # Check required modalities
        if 'coi' not in modal_data_dict:
            raise ValueError("COI modality data must be provided")
        
        # Determine which modalities are needed based on model stage
        required_modals = ['coi']
        
        # For multimodal models, add other modalities (if provided)
        if self.curriculum_stage > 0:
            for modal in ['rn16s', 'h3', 'rn18s', 'its1', 'its2']:
                if modal in modal_data_dict:
                    required_modals.append(modal)
                else:
                    print(f"Warning: Multimodal model may require {modal}, but not provided")
        
        # Filter out needed modalities
        filtered_modal_data = {modal: modal_data_dict[modal] for modal in required_modals 
                              if modal in modal_data_dict}
        
        # Get number of samples
        n_samples = len(filtered_modal_data['coi'])
        
        # Verify sample count consistency for all modal data
        for modal, data in filtered_modal_data.items():
            if modal not in self.supported_modals:
                raise ValueError(f"Unsupported modality: {modal}")
            if len(data) != n_samples:
                raise ValueError(f"{modal} sample count mismatch: expected {n_samples}, got {len(data)}")
            self.validate_modal_data(modal, data)
        
        print(f"Successfully prepared {n_samples} samples, used modalities: {list(filtered_modal_data.keys())}")
        return filtered_modal_data

    def predict(self, embeddings_dict, batch_size=64, return_probabilities=False, top_k=3):
        """
        Predict on new samples
        """
        # Create dummy COI model
        from model import CurriculumDMGHAN
        dummy_coi_model = CurriculumDMGHAN(self.config, curriculum_stage=0).to(self.device)
        
        # Use FixedMultiModalTester for prediction
        tester = FixedMultiModalTester(
            coi_model=dummy_coi_model,
            multimodal_model=self.model,
            class_counts=self.config["n_classes"],
            device=self.device
        )
        
        # Create dummy labels (all set to -1, meaning unknown)
        n_samples = len(embeddings_dict['coi'])
        dummy_labels = [
            np.full(n_samples, -1),  # subclass
            np.full(n_samples, -1),  # order
            np.full(n_samples, -1),  # superfamily
            np.full(n_samples, -1),  # family
            np.full(n_samples, -1),  # genus
            np.full(n_samples, -1)   # species
        ]
        
        # Create test dataset
        test_dataset = tester.create_test_dataset(embeddings_dict, dummy_labels)
        
        # Create DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=tester._collate_fn,
            pin_memory=True
        )
        
        # Perform prediction
        predictions = self._predict_model(self.model, test_loader, return_probabilities, top_k)
        
        # If label mapper is provided, map prediction results
        if self.label_mapper:
            predictions = self.label_mapper.map_predictions(predictions)
        
        return predictions

    def _predict_model(self, model, test_loader, return_probabilities=False, top_k=3):
        """
        Use model for prediction
        """
        model.eval()
        all_predictions = {
            level: {
                'pred': [], 
                'prob': [],
                'top_k_pred': [],
                'top_k_prob': []
            } 
            for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
        }
        
        with torch.no_grad():
            for batch in test_loader:
                # Prepare input data
                inputs = {
                    'embeds': {
                        modal: tensor.to(self.device, non_blocking=True)
                        for modal, tensor in batch['embeds'].items()
                    }
                }
                
                # Forward propagation
                outputs = model(inputs)
                
                # Process predictions for each level
                for level in all_predictions:
                    level_outputs = outputs[level]
                    
                    # Get predicted classes
                    preds = torch.argmax(level_outputs, dim=1).cpu().numpy()
                    all_predictions[level]['pred'].extend(preds)
                    
                    # Get probabilities
                    probs = torch.softmax(level_outputs, dim=1).cpu().numpy()
                    all_predictions[level]['prob'].extend(probs)
                    
                    # Get top-k predictions
                    top_k_probs, top_k_preds = torch.topk(level_outputs, min(top_k, level_outputs.size(1)), dim=1)
                    top_k_probs = torch.softmax(top_k_probs, dim=1).cpu().numpy()
                    top_k_preds = top_k_preds.cpu().numpy()
                    
                    all_predictions[level]['top_k_pred'].extend(top_k_preds)
                    all_predictions[level]['top_k_prob'].extend(top_k_probs)
        
        # Organize results
        results = {}
        for level in all_predictions:
            results[level] = {
                'predictions': np.array(all_predictions[level]['pred'])
            }
            if return_probabilities:
                results[level]['probabilities'] = np.array(all_predictions[level]['prob'])
                results[level]['top_k_predictions'] = np.array(all_predictions[level]['top_k_pred'])
                results[level]['top_k_probabilities'] = np.array(all_predictions[level]['top_k_prob'])
        
        return results

def load_embedding_files(embedding_paths_dict):
    """
    Load embedding files from specified file paths
    embedding_paths_dict: Dictionary, keys are modality names, values are corresponding embedding file paths
    """
    embeddings = {}
    for modal, file_path in embedding_paths_dict.items():
        if os.path.exists(file_path):
            try:
                embeddings[modal] = np.load(file_path)
                print(f"Successfully loaded {modal} embedding data: {embeddings[modal].shape} (from {file_path})")
            except Exception as e:
                print(f"Failed to load {modal} embedding data: {e}")
        else:
            print(f"Warning: {modal} embedding file does not exist: {file_path}")
    
    return embeddings

def load_index_files(index_paths_dict):
    """
    Load index files from specified file paths
    index_paths_dict: Dictionary, keys are modality names, values are corresponding index file paths
    """
    indices = {}
    for modal, file_path in index_paths_dict.items():
        if os.path.exists(file_path):
            try:
                # Assume index file is in txt format, containing 'Index' column
                df = pd.read_csv(file_path, sep=',')
                if 'Index' in df.columns:
                    indices[modal] = df['Index'].values
                    print(f"Successfully loaded {modal} index data: {len(indices[modal])} indices (from {file_path})")
                else:
                    print(f"Warning: No 'Index' column in {file_path}")
            except Exception as e:
                print(f"Failed to load {modal} index data: {e}")
        else:
            print(f"Warning: {modal} index file does not exist: {file_path}")
    
    return indices

def save_predictions(predictions, output_path, sample_names=None, include_top_k=False, model_name=""):
    """
    Save prediction results to CSV file
    """
    n_samples = len(predictions['species']['predictions'])
    
    # Create result DataFrame
    results_df = pd.DataFrame()
    
    # Add model information
    results_df['model'] = [model_name] * n_samples
    
    # Add sample names (if any)
    if sample_names is not None:
        results_df['sample_name'] = sample_names
    else:
        results_df['sample_id'] = range(n_samples)
    
    # Add prediction results for each level
    for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
        # Add prediction indices
        results_df[f'{level}_prediction'] = predictions[level]['predictions']
        
        # If classification names are included, add name column
        if 'names' in predictions[level]:
            results_df[f'{level}_name'] = predictions[level]['names']
        
        # If probabilities are included, add confidence
        if 'probabilities' in predictions[level]:
            # Maximum probability as confidence
            max_probs = np.max(predictions[level]['probabilities'], axis=1)
            results_df[f'{level}_confidence'] = max_probs
        
        # Add top-k prediction results
        if include_top_k and 'top_k_predictions' in predictions[level]:
            top_k_preds = predictions[level]['top_k_predictions']
            top_k_probs = predictions[level]['top_k_probabilities']
            
            for k in range(top_k_preds.shape[1]):
                results_df[f'{level}_top{k+1}_pred'] = top_k_preds[:, k]
                results_df[f'{level}_top{k+1}_prob'] = top_k_probs[:, k]
                
                # Add top-k name columns
                if 'top_k_names' in predictions[level]:
                    top_k_names = predictions[level]['top_k_names']
                    results_df[f'{level}_top{k+1}_name'] = top_k_names[:, k]
    
    # Save to file
    results_df.to_csv(output_path, index=False)
    print(f"Prediction results saved to: {output_path}")
    
    return results_df

def predict_with_model(model_info, embedding_paths_dict, index_paths_dict, output_dir, sample_names=None, label_mapper=None):
    """
    Predict using specified model
    model_info: Dictionary containing model path, stage, etc.
    label_mapper: Label mapper instance
    """
    model_path = model_info['path']
    curriculum_stage = model_info.get('stage', 0)
    model_name = model_info.get('name', os.path.basename(model_path))
    
    print(f"\n{'='*50}")
    print(f"Using model: {model_name}")
    print(f"{'='*50}")
    
    # Initialize predictor
    predictor = FlexibleSamplePredictor(
        model_path, 
        base_config, 
        curriculum_stage=curriculum_stage,
        label_mapper=label_mapper
    )
    
    # Load embedding data
    embeddings = load_embedding_files(embedding_paths_dict)
    
    # Load index data
    indices = load_index_files(index_paths_dict)
    
    # Check required data
    if 'coi' not in embeddings:
        print("Error: COI embedding data must be provided")
        return None
    
    if 'coi' not in indices:
        print("Error: COI index data must be provided")
        return None
    
    # Align samples by index
    print("Aligning samples by index...")
    aligned_embeddings = predictor.align_samples_by_index(embeddings, indices, base_modal='coi')
    
    # Prepare test data
    test_embeddings = predictor.prepare_samples(aligned_embeddings)
    
    # Perform prediction
    print("Starting prediction...")
    predictions = predictor.predict(test_embeddings, return_probabilities=True, top_k=3)
    
    # Save results
    output_path = os.path.join(output_dir, f"predictions_{model_name}.csv")
    results_df = save_predictions(predictions, output_path, sample_names, include_top_k=True, model_name=model_name)
    
    # Print partial results
    print(f"\n{model_name} prediction completed! Top 5 sample prediction results:")
    print(results_df.head())
    
    # Statistics of prediction distribution for each level
    print(f"\n{model_name} prediction category distribution for each level:")
    for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
        if 'names' in predictions[level]:
            unique_names, counts = np.unique(predictions[level]['names'], return_counts=True)
            print(f"{level}: {len(unique_names)} different categories")
            # Print top 5 most common categories
            for name, count in zip(unique_names[:5], counts[:5]):
                print(f"  - {name}: {count} samples")
        else:
            unique, counts = np.unique(predictions[level]['predictions'], return_counts=True)
            print(f"{level}: {len(unique)} different categories")
    
    return results_df, predictions


def generate_summary_report(all_results, output_dir):
    """
    Generate summary report for all models
    """
    if not all_results:
        print("No prediction results available")
        return
    
    # Create summary DataFrame
    summary_data = []
    
    for model_name, result in all_results.items():
        predictions = result['predictions']
        
        # Calculate accuracy for each level (if true labels exist)
        # Here we only count prediction category distribution
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            if 'names' in predictions[level]:
                level_names = predictions[level]['names']
                unique, counts = np.unique(level_names, return_counts=True)
                
                summary_data.append({
                    'model': model_name,
                    'level': level,
                    'unique_classes': len(unique),
                    'most_common_class': unique[np.argmax(counts)] if len(unique) > 0 else "Unknown",
                    'most_common_count': np.max(counts) if len(counts) > 0 else 0
                })
            else:
                level_preds = predictions[level]['predictions']
                unique, counts = np.unique(level_preds, return_counts=True)
                
                summary_data.append({
                    'model': model_name,
                    'level': level,
                    'unique_classes': len(unique),
                    'most_common_class': f"Index_{unique[np.argmax(counts)]}" if len(unique) > 0 else -1,
                    'most_common_count': np.max(counts) if len(counts) > 0 else 0
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "prediction_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary report saved to: {summary_path}")
    
    # Print summary information
    print("\nPrediction Summary:")
    for model_name in all_results.keys():
        print(f"\n{model_name}:")
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            level_data = summary_df[(summary_df['model'] == model_name) & (summary_df['level'] == level)]
            if not level_data.empty:
                row = level_data.iloc[0]
                print(f"  {level}: {row['unique_classes']} different categories, most common: {row['most_common_class']} ({row['most_common_count']} samples)")


def main():
    # Model configuration
    MODEL_DIR = "./pretrained_models"
    
    # Define list of models to use
    models_to_predict = [
        {
            'name': 'stage0_coi_only',
            'path': f"{MODEL_DIR}/stage0_coi_only.pt",
            'stage': 0
        },
        {
            'name': 'stage1_multimodal_fusion',
            'path': f"{MODEL_DIR}/stage1_multimodal_fusion_BestMeanAcc.pt",
            'stage': 1
        },
        {
            'name': 'stage2_multimodal_fulltuning',
            'path': f"{MODEL_DIR}/stage2_multimodal_fulltuning_BestMeanAcc.pt",
            'stage': 2
        },
        {
            'name': 'DomainA_after_stage_2',
            'path': f"{MODEL_DIR}/DomainA_BestAcc_after_stage_2.pt",
            'stage': 2
        }
    ]

    # Label file path
    LABEL_FILE_PATH = "./data/data_Train_Val_Test/Augmented_76577_All_used_Gastropoda_seqs_Taxinomy_LabelSpecies.txt"

    
    # User specified embedding file paths
    EMBEDDING_PATHS = {
        'coi': "./data/Case_Study_Gastropoda/Seq_Labels_COI_df_SEQUENCE/cls_embedding.npy",           # COI embedding (ERNIE-RNA)
        'rn16s': "./data/Case_Study_Gastropoda/Seq_Labels_16S_df_SEQUENCE/cls_embedding.npy",         # 16S embedding
        'h3': "./data/Case_Study_Gastropoda/Seq_Labels_H3_df_SEQUENCE/cls_embedding.npy",            # 18S embedding
        'rn18s': "./data/Case_Study_Gastropoda/Seq_Labels_18S_df_SEQUENCE/cls_embedding.npy"         # H3 embedding
    }
    
    # User specified index file paths (remove coi_MAE)
    INDEX_PATHS = {
        'coi': "./data/Case_Study_Gastropoda/Seq_Labels_COI_df.txt",           # COI index file
        'rn16s': "./data/Case_Study_Gastropoda/Seq_Labels_16S_df.txt",         # 16S index file
        'h3': "./data/Case_Study_Gastropoda/Seq_Labels_H3_df.txt",          # 18S index file
        'rn18s': "./data/Case_Study_Gastropoda/Seq_Labels_18S_df.txt"       # H3 index file
    }
    
    OUTPUT_DIR = "./data/Case_Study_Gastropoda/predictions_Gastropoda"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Sample names (optional)
    smp_df = pd.read_csv("./data/Case_Study_Gastropoda/Seq_Labels_COI_df.txt", sep=",")
    SAMPLE_NAMES = smp_df['accession'].values  # Can be set to sample name list

    
    # Initialize label mapper
    print("Loading label mapping...")
    label_mapper = LabelMapper(LABEL_FILE_PATH)
    
    # Predict for all models
    all_results = {}
    for model_info in models_to_predict:
        try:
            results_df, predictions = predict_with_model(
                model_info, 
                EMBEDDING_PATHS, 
                INDEX_PATHS, 
                OUTPUT_DIR, 
                SAMPLE_NAMES,
                label_mapper=label_mapper
            )
            all_results[model_info['name']] = {
                'dataframe': results_df,
                'predictions': predictions
            }
        except Exception as e:
            print(f"Model {model_info['name']} prediction failed: {e}")
            continue
    
    # Generate summary report
    generate_summary_report(all_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()

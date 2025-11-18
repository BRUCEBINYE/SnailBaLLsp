

import numpy as np
import pandas as pd

import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device("cuda:0")


def load_test_labels(label_path, class_counts):
        """
        Load label data and handle NaN values
        """
        dfl = pd.read_csv(label_path, sep="\t", dtype='str')
        labels = dfl[['subclass', 'order', 'superfamily', 'family', 'genus', 'species']]
        
        invalid_label = -1

        # Handle NaN values
        labels = labels.fillna(str(invalid_label))
        class_counts = {
            'subclass': class_counts[0],
            'order': class_counts[1],
            'superfamily': class_counts[2],
            'family': class_counts[3],
            'genus': class_counts[4],
            'species': class_counts[5]
        }
        
        # Change the lable to integer
        label_arrays = []
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            try:
                level_labels = labels[level].astype(int).values
            except ValueError:
                # 处理可能的转换错误
                valid_labels = []
                for val in labels[level]:
                    try:
                        valid_labels.append(int(val))
                    except ValueError:
                        valid_labels.append(invalid_label)
                level_labels = np.array(valid_labels)
            
            # Replace labels that are out of range to the invalid values -1
            max_label = class_counts[level] - 1
            level_labels[(level_labels < 0) | (level_labels > max_label)] = invalid_label
            label_arrays.append(level_labels)
        
        return label_arrays





class FixedMultiModalTester:
    def __init__(self, coi_model, multimodal_model, class_counts, device=DEVICE):
        """
        :param class_counts: Number of categories at each taxonomy level [subclass, order, superfamily, family, genus, species]
        """
        self.coi_model = coi_model.to(device)
        self.coi_model.eval()
        
        self.multimodal_model = multimodal_model.to(device)
        self.multimodal_model.eval()
        
        self.device = device
        self.zero_embed = torch.zeros(12, 768).to(device)
        self.zero_aux = torch.zeros(768).to(device)
        self.all_modals = ['coi', 'coi_MAE', 'rn16s', 'h3', 'rn18s', 'its1', 'its2']  # Adding coi_MAE
        self.invalid_label = -1
        
        # Number of categories at each taxonomy level
        self.class_counts = {
            'subclass': class_counts[0],
            'order': class_counts[1],
            'superfamily': class_counts[2],
            'family': class_counts[3],
            'genus': class_counts[4],
            'species': class_counts[5]
        }

    def load_labels(self, label_path):
        """
        Load label data and handle NaN values
        """
        dfl = pd.read_csv(label_path, sep="\t", dtype='str')
        labels = dfl[['subclass', 'order', 'superfamily', 'family', 'genus', 'species']]
        
        # Handle NaN values
        labels = labels.fillna(str(self.invalid_label))
        
        label_arrays = []
        for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']:
            try:
                level_labels = labels[level].astype(int).values
            except ValueError:
                valid_labels = []
                for val in labels[level]:
                    try:
                        valid_labels.append(int(val))
                    except ValueError:
                        valid_labels.append(self.invalid_label)
                level_labels = np.array(valid_labels)
            
            max_label = self.class_counts[level] - 1
            level_labels[(level_labels < 0) | (level_labels > max_label)] = self.invalid_label
            label_arrays.append(level_labels)
        
        return label_arrays

    def create_test_dataset(self, embeddings_dict, labels):
        """
        Create a test dataset and handle invalid labels
        """
        # Ensure the existence of COI
        if 'coi' not in embeddings_dict:
            raise ValueError("The test set must include COI modal data")
            
        # Sample size
        num_samples = len(embeddings_dict['coi'])
        
        # Verify label length
        for level_labels in labels:
            if len(level_labels) != num_samples:
                raise ValueError(f"Number of labels ({len(level_labels)})is mismatch with the number of embeddings({num_samples})")
        
        # Fill in missing modes (including coi_maE)
        for modal in self.all_modals:
            if modal not in embeddings_dict:
                if modal == 'coi_MAE':
                    # coi_MAE is a vector, not a matrix
                    embeddings_dict[modal] = [np.zeros(768) for _ in range(num_samples)]
                else:
                    embeddings_dict[modal] = [np.zeros((12, 768)) for _ in range(num_samples)]
        
        # Create dataset
        dataset = []
        for i in range(num_samples):
            sample = {
                'embeds': {
                    'coi': torch.FloatTensor(embeddings_dict['coi'][i]),
                    'coi_MAE': torch.FloatTensor(embeddings_dict['coi_MAE'][i]),
                    'rn16s': torch.FloatTensor(embeddings_dict['rn16s'][i]),
                    'h3': torch.FloatTensor(embeddings_dict['h3'][i]),
                    'rn18s': torch.FloatTensor(embeddings_dict['rn18s'][i]),
                    'its1': torch.FloatTensor(embeddings_dict['its1'][i]),
                    'its2': torch.FloatTensor(embeddings_dict['its2'][i])
                },
                'labels': {
                    'subclass': labels[0][i],
                    'order': labels[1][i],
                    'superfamily': labels[2][i],
                    'family': labels[3][i],
                    'genus': labels[4][i],
                    'species': labels[5][i]
                }
            }
            dataset.append(sample)
        return dataset

    def evaluate_test_set(self, test_dataset, batch_size=64, k=5):
        """
        Evaluate the performance of the test set
        """
        # Automatically select model - judge based on the first sample
        first_sample = test_dataset[0]
        has_other_modals = any(modal in first_sample['embeds'] for modal in ['rn16s', 'h3', 'rn18s', 'its1', 'its2'])
        model = self.multimodal_model if has_other_modals else self.coi_model
        
        # Create Dataset class
        class TestDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Create DataLoader
        test_loader = DataLoader(
            TestDataset(test_dataset),
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
        
        return self._evaluate_model(model, test_loader, k)

    def _collate_fn(self, batch):
        """
        Custom batch processing function
        """
        collated = {
            'embeds': {},
            'labels': {
                'subclass': [],
                'order': [],
                'superfamily': [],
                'family': [],
                'genus': [],
                'species': []
            }
        }
        
        # Processing embeddings (distinguishing between matrices and vectors)
        for item in batch:
            for modal in self.all_modals:
                if modal not in collated['embeds']:
                    # Determine dimensions based on the first sample
                    sample_shape = item['embeds'][modal].shape
                    if len(sample_shape) == 1:  # Vector, such as coi_MAE
                        collated['embeds'][modal] = []
                    else:  # Matrix, such as other modalities
                        collated['embeds'][modal] = []
        
        # Fill embedding
        for modal in collated['embeds']:
            for item in batch:
                collated['embeds'][modal].append(item['embeds'][modal])
            # Convert to tensor
            collated['embeds'][modal] = torch.stack(collated['embeds'][modal])
        
        # Handle the labels
        for level in collated['labels']:
            collated['labels'][level] = torch.LongTensor([item['labels'][level] for item in batch])
            
        return collated

    def _safe_topk(self, tensor, k, level):
        """Top-K operations"""
        num_classes = self.class_counts[level]
        valid_k = min(k, num_classes)
        if valid_k < k:
            print(f" Warning: Level {level} only has {num_classes} categories,"
                  f"Automatically adjusted k value from {k} to {valid_k}")
        
        return torch.topk(tensor, valid_k, dim=1)

    def _evaluate_model(self, model, test_loader, k=5):
        """
        Evaluate model performance (with secure Top-K and invalid label handling)
        """
        model.eval()
        metrics = {
            level: {
                'true': [],
                'pred': [],
                'topk_correct': 0,
                'valid_count': 0,  # Effective sample count
                'actual_k': min(k, self.class_counts[level])  # Record the actual value of k used
            } for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
        }
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Input, ensure to include coi_MAE
                inputs = {
                    'embeds': {
                        modal: tensor.to(self.device, non_blocking=True)
                        for modal, tensor in batch['embeds'].items()
                    }
                }
                
                # Targets
                targets = {
                    level: batch['labels'][level].to(self.device, non_blocking=True)
                    for level in metrics
                }
                
                # Prediction
                outputs = model(inputs)
                
                # Results
                batch_size = len(batch['labels']['subclass'])
                total_samples += batch_size
                
                for level in metrics:
                    # Obtain predictions and targets for the current level
                    level_outputs = outputs[level]
                    level_targets = targets[level]
                    
                    # Identify effective samples
                    valid_mask = (level_targets != self.invalid_label)
                    valid_indices = torch.where(valid_mask)[0]
                    valid_targets = level_targets[valid_mask]
                    
                    # Only process effective samples
                    if len(valid_targets) > 0:
                        # Top-1 accuracy
                        valid_preds = torch.argmax(level_outputs[valid_mask], dim=1).cpu().numpy()
                        metrics[level]['true'].extend(valid_targets.cpu().numpy())
                        metrics[level]['pred'].extend(valid_preds)
                        
                        # Top-K accuracy
                        _, topk_preds = self._safe_topk(level_outputs[valid_mask], k, level)
                        for i in range(len(valid_targets)):
                            if valid_targets[i] in topk_preds[i]:
                                metrics[level]['topk_correct'] += 1
                        
                        # Update effective sample count
                        metrics[level]['valid_count'] += len(valid_targets)
        
        # Metrics
        results = {}
        for level in metrics:
            if metrics[level]['valid_count'] > 0:
                y_true = np.array(metrics[level]['true'])
                y_pred = np.array(metrics[level]['pred'])
                
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted')
                actual_k = metrics[level]['actual_k']
                topk_acc = metrics[level]['topk_correct'] / metrics[level]['valid_count']
                
                results[level] = {
                    'top1_accuracy': acc,
                    'f1_score': f1,
                    f'top{actual_k}_accuracy': topk_acc,
                    'actual_k_used': actual_k,
                    'valid_samples': metrics[level]['valid_count'],
                    'total_samples': total_samples
                }
            else:
                results[level] = {
                    'error': f"No effective samples (all the labels are {self.invalid_label})",
                    'valid_samples': 0,
                    'total_samples': total_samples
                }
        
        return results












class FixedMultiModalTesterMAE:
    def __init__(self, coi_model, multimodal_model, class_counts, device=DEVICE):
        """
        Number of categories at each taxonomy level [subclass, order, superfamily, family, genus, species]
        """
        self.coi_model = coi_model.to(device)
        self.coi_model.eval()
        
        self.multimodal_model = multimodal_model.to(device)
        self.multimodal_model.eval()
        
        self.device = device
        self.zero_embed = torch.zeros(12, 768).to(device)
        self.zero_aux = torch.zeros(768).to(device)
        self.all_modals = ['coi', 'coi_MAE', 'rn16s', 'h3', 'rn18s', 'its1', 'its2']  # Adding coi_MAE
        self.invalid_label = -1  # Novalid Label
        
        #  Number of categories at each taxonomy level
        self.class_counts = {
            'subclass': class_counts[0],
            'order': class_counts[1],
            'superfamily': class_counts[2],
            'family': class_counts[3],
            'genus': class_counts[4],
            'species': class_counts[5]
        }

        # Store modal information
        self.modal_shapes = {
            'coi': (12, 768),
            'coi_MAE': (768,),
            'rn16s': (12, 768),
            'h3': (12, 768),
            'rn18s': (12, 768),
            'its1': (12, 768),
            'its2': (12, 768)
        }

    
    def align_test_data(self, data_dict):
        """
        Align multimodal test data (based on COI)
        :param data_dict: Dictionary format, containing data and indexes for each modality
            {
                'coi': {'embeddings': np.array, 'index': np.array},
                'coi_MAE': {'embeddings': np.array, 'index': np.array},
                'rn16s': {'embeddings': np.array, 'index': np.array},
                ... # Other modalities
            }
        :return: Aligned embedded dictionary and label array
        """
        # COI (ERNIE-RNA embedding) modality must exist
        if 'coi' not in data_dict:
            raise ValueError("The test data must include COI modality")
        
        # Based on COI index
        base_index = data_dict['coi']['index']
        num_samples = len(base_index)
        
        # Create result dict
        aligned_embeddings = {
            'coi': data_dict['coi']['embeddings'],
            'coi_MAE': np.zeros((num_samples, 768)),  # Default filling with 0
            'rn16s': np.zeros((num_samples, 12, 768)),
            'h3': np.zeros((num_samples, 12, 768)),
            'rn18s': np.zeros((num_samples, 12, 768)),
            'its1': np.zeros((num_samples, 12, 768)),
            'its2': np.zeros((num_samples, 12, 768))
        }
        
        # Align every modality
        for modal in ['coi_MAE', 'rn16s', 'h3', 'rn18s', 'its1', 'its2']:
            if modal in data_dict:
                modal_data = data_dict[modal]
                modal_emb = modal_data['embeddings']
                modal_idx = modal_data['index']
                
                # Create index mapping
                index_map = {idx: i for i, idx in enumerate(modal_idx)}
                
                # Align modal data to COI index
                for i, coi_idx in enumerate(base_index):
                    if coi_idx in index_map:
                        pos = index_map[coi_idx]
                        aligned_embeddings[modal][i] = modal_emb[pos]
        
        return aligned_embeddings


    def create_test_dataset(self, embeddings_dict, labels):
        """
        Create a test dataset and handle invalid labels
        """
        # COI (ERNIE-RNA embedding) modality must exist
        if 'coi' not in embeddings_dict:
            raise ValueError("The test data must include COI modality")
            
        # Sample size
        num_samples = len(embeddings_dict['coi'])
        
        # Label length
        for level_labels in labels:
            if len(level_labels) != num_samples:
                raise ValueError(f"Number of labels ({len(level_labels)})is mismatch with the number of embeddings({num_samples})")
        
        # Create dataset
        dataset = []
        for i in range(num_samples):
            sample = {
                'embeds': {
                    'coi': torch.FloatTensor(embeddings_dict['coi'][i]),
                    'coi_MAE': torch.FloatTensor(embeddings_dict['coi_MAE'][i]),
                    'rn16s': torch.FloatTensor(embeddings_dict['rn16s'][i]),
                    'h3': torch.FloatTensor(embeddings_dict['h3'][i]),
                    'rn18s': torch.FloatTensor(embeddings_dict['rn18s'][i]),
                    'its1': torch.FloatTensor(embeddings_dict['its1'][i]),
                    'its2': torch.FloatTensor(embeddings_dict['its2'][i])
                },
                'labels': {
                    'subclass': labels[0][i],
                    'order': labels[1][i],
                    'superfamily': labels[2][i],
                    'family': labels[3][i],
                    'genus': labels[4][i],
                    'species': labels[5][i]
                }
            }
            dataset.append(sample)
        return dataset

    def evaluate_test_set(self, test_dataset, batch_size=64, k=5):
        """
        Evaluate the performance of the test set
        """
        # Automatically select model - judge based on the first sample
        first_sample = test_dataset[0]
        has_other_modals = any(modal in first_sample['embeds'] for modal in ['rn16s', 'h3', 'rn18s', 'its1', 'its2'])
        model = self.multimodal_model if has_other_modals else self.coi_model
        
        # Create Dataset class
        class TestDataset(Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Create DataLoader
        test_loader = DataLoader(
            TestDataset(test_dataset),
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            pin_memory=True
        )
        
        return self._evaluate_model(model, test_loader, k)

    def _collate_fn(self, batch):
        """
        Custom batch processing function, supporting coi_MAE
        """
        collated = {
            'embeds': {},
            'labels': {
                'subclass': [],
                'order': [],
                'superfamily': [],
                'family': [],
                'genus': [],
                'species': []
            }
        }
        
        # Processing embeddings (distinguishing between matrices and vectors)
        for item in batch:
            for modal in self.all_modals:
                if modal not in collated['embeds']:
                    # Determine dimensions based on the first sample
                    sample_shape = item['embeds'][modal].shape
                    if len(sample_shape) == 1:  # Vector, such as coi_MAE
                        collated['embeds'][modal] = []
                    else:  # Matrix, such as other modalities
                        collated['embeds'][modal] = []
        
        # Fill embedding
        for modal in collated['embeds']:
            for item in batch:
                collated['embeds'][modal].append(item['embeds'][modal])
            # Convert to tensor
            collated['embeds'][modal] = torch.stack(collated['embeds'][modal])
        
        # Handle labels
        for level in collated['labels']:
            collated['labels'][level] = torch.LongTensor([item['labels'][level] for item in batch])
            
        return collated

    def _safe_topk(self, tensor, k, level):
        """Top-K operations"""
        num_classes = self.class_counts[level]
        valid_k = min(k, num_classes)
        if valid_k < k:
            print(f" Warning: Level {level} only has {num_classes} categories,"
                  f"Automatically adjusted k value from {k} to {valid_k}")
        return torch.topk(tensor, valid_k, dim=1)

    def _evaluate_model(self, model, test_loader, k=5):
        """
        Evaluate model performance (with secure Top-K and invalid label handling)
        """
        model.eval()
        metrics = {
            level: {
                'true': [],
                'pred': [],
                'topk_correct': 0,
                'valid_count': 0,  # Effective sample count
                'actual_k': min(k, self.class_counts[level])  # Record the actual value of k used
            } for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
        }
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Input, ensure to include coi_MAE
                inputs = {
                    'embeds': {
                        modal: tensor.to(self.device, non_blocking=True)
                        for modal, tensor in batch['embeds'].items()
                    }
                }
                
                # Targets
                targets = {
                    level: batch['labels'][level].to(self.device, non_blocking=True)
                    for level in metrics
                }
                
                # Predict
                outputs = model(inputs)
                
                # Results
                batch_size = len(batch['labels']['subclass'])
                total_samples += batch_size
                
                for level in metrics:
                    # Obtain predictions and targets for the current level
                    level_outputs = outputs[level]
                    level_targets = targets[level]
                    
                    # Identify effective samples
                    valid_mask = (level_targets != self.invalid_label)
                    valid_indices = torch.where(valid_mask)[0]
                    valid_targets = level_targets[valid_mask]
                    
                    # Only process effective samples
                    if len(valid_targets) > 0:
                        # Top-1 accuracy
                        valid_preds = torch.argmax(level_outputs[valid_mask], dim=1).cpu().numpy()
                        metrics[level]['true'].extend(valid_targets.cpu().numpy())
                        metrics[level]['pred'].extend(valid_preds)
                        
                        # Top-K accuracy
                        _, topk_preds = self._safe_topk(level_outputs[valid_mask], k, level)
                        for i in range(len(valid_targets)):
                            if valid_targets[i] in topk_preds[i]:
                                metrics[level]['topk_correct'] += 1
                        
                        # Update effective sample count
                        metrics[level]['valid_count'] += len(valid_targets)
        
        #  Metrics
        results = {}
        for level in metrics:
            if metrics[level]['valid_count'] > 0:
                y_true = np.array(metrics[level]['true'])
                y_pred = np.array(metrics[level]['pred'])
                
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='weighted')
                actual_k = metrics[level]['actual_k']
                topk_acc = metrics[level]['topk_correct'] / metrics[level]['valid_count']
                
                results[level] = {
                    'top1_accuracy': acc,
                    'f1_score': f1,
                    f'top{actual_k}_accuracy': topk_acc,
                    'actual_k_used': actual_k,
                    'valid_samples': metrics[level]['valid_count'],
                    'total_samples': total_samples
                }
            else:
                results[level] = {
                    'error': f"没有有效样本（所有标签均为{self.invalid_label}）",
                    'valid_samples': 0,
                    'total_samples': total_samples
                }
        
        return results



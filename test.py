
import numpy as np
import pandas as pd

import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device("cuda:0")


class FixedMultiModalTester:
    def __init__(self, coi_model, multimodal_model, class_counts, device=DEVICE):
        """
        :param class_counts: Number of categories at each level [subclass, order, superfamily, family, genus, species]
        """
        self.coi_model = coi_model.to(device)
        self.coi_model.eval()
        
        self.multimodal_model = multimodal_model.to(device)
        self.multimodal_model.eval()
        
        self.device = device
        self.zero_embed = torch.zeros(12, 768).to(device)
        self.all_modals = ['coi', 'rn16s', 'h3', 'rn18s', 'its1', 'its2']
        self.invalid_label = -1  # Invalid label
        
        # Store the number of categories for each level
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
        Load label data and process NaN values
        """
        dfl = pd.read_csv(label_path, sep="\t", dtype='str')
        labels = dfl[['subclass', 'order', 'superfamily', 'family', 'genus', 'species']]
        
        # Handling NaN values
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
            
            # Replace labels that are out of range with invalid values
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
            
        num_samples = len(embeddings_dict['coi'])
        
        # Verify label length
        for level_labels in labels:
            if len(level_labels) != num_samples:
                raise ValueError(f"标签数量({len(level_labels)})与嵌入数量({num_samples})不匹配")
        
        # Fill in missing modes
        for modal in self.all_modals:
            if modal not in embeddings_dict:
                embeddings_dict[modal] = [np.zeros((12, 768)) for _ in range(num_samples)]
        
        # Create dataset
        dataset = []
        for i in range(num_samples):
            sample = {
                'embeds': {modal: torch.FloatTensor(embeddings_dict[modal][i]) 
                          for modal in self.all_modals},
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
        """Evaluate the performance of the test set"""
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
        """batch processing function"""
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
        
        # Processing embeddings
        for modal in self.all_modals:
            collated['embeds'][modal] = torch.stack([item['embeds'][modal] for item in batch])
        
        # Processing labels
        for level in collated['labels']:
            collated['labels'][level] = torch.LongTensor([item['labels'][level] for item in batch])
            
        return collated

    def _safe_topk(self, tensor, k, level):
        """Safely execute Top-K operations"""
        num_classes = self.class_counts[level]
        valid_k = min(k, num_classes)
        if valid_k < k:
            print(f" Warning: {level}-level only has{num_classes}categies,"
                  f"Automatically adjusted k value from{k} to {valid_k}")
        
        return torch.topk(tensor, valid_k, dim=1)

    def _evaluate_model(self, model, test_loader, k=5):
        """Evaluate model performance"""
        model.eval()
        metrics = {
            level: {
                'true': [],
                'pred': [],
                'topk_correct': 0,
                'valid_count': 0,
                'actual_k': min(k, self.class_counts[level])  # Record the actual value of k used
            } for level in ['subclass', 'order', 'superfamily', 'family', 'genus', 'species']
        }
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:

                inputs = {
                    'embeds': {
                        k: v.to(self.device, non_blocking=True)
                        for k, v in batch['embeds'].items()
                    }
                }
                
                targets = {
                    level: batch['labels'][level].to(self.device, non_blocking=True)
                    for level in metrics
                }
                
                outputs = model(inputs)
                
                batch_size = len(batch['labels']['subclass'])
                total_samples += batch_size
                
                for level in metrics:
                    # Obtain predictions and targets for the current level
                    level_outputs = outputs[level]
                    level_targets = targets[level]
                    
                    # Find valid samples (label not -1)
                    valid_mask = (level_targets != self.invalid_label)
                    valid_indices = torch.where(valid_mask)[0]
                    valid_targets = level_targets[valid_mask]
                    
                    # Only process valid samples
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
        
        # Calculate matrics
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




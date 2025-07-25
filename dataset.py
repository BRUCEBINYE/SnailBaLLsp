
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -------------------- 1. Support multimodal dataset classes --------------------
class MultiModalCOIDataset(Dataset):
    def __init__(self, embeddings_dict, labels):
        """
        embeddings_dict:
        {
            'coi': tensor(coi_emb), 
            'rn16s': tensor(rn16s_emb),
            ...
        }
        """
        self.embeddings = embeddings_dict
        self.labels = [torch.LongTensor(l).cpu() for l in labels]
        
    def __len__(self):
        return len(self.embeddings['coi'])
    
    def __getitem__(self, idx):
        return {
            'embeds': {
                modal: emb[idx]  # The dimensions of emb should be (N, 12, 768)
                for modal, emb in self.embeddings.items()
            },
            'labels': {
                level: self.labels[i][idx]
                for i, level in enumerate(['subclass', 'order', 'superfamily', 'family', 'genus', 'species'])
            }
        }





class MultiModalCOIDatasetMAE(Dataset):
    def __init__(self, embeddings_dict, labels):
        """
        embeddings_dict: A dictionary containing embeddings of various modalities:
        {
            'coi': tensor(coi_emb), 
            'rn16s': tensor(rn16s_emb),
            ...
        }
        """
        self.embeddings = embeddings_dict
        self.labels = [torch.LongTensor(l).cpu() for l in labels]
        
    def __len__(self):
        return len(self.embeddings['coi'])
    
    def __getitem__(self, idx):
        return {
            'embeds': {
                'coi': embeddings['coi'][idx],          # COI embedding of RNA secondary structure
                'coi_MAE': embeddings['coi_MAE'][idx],  # Additional COI embedding of DNA barcoding
                'rn16s': embeddings['rn16s'][idx],
                'h3': embeddings['h3'][idx],
                'rn18s': embeddings['rn18s'][idx],
                'its1': embeddings['its1'][idx],
                'its2': embeddings['its2'][idx]
            },
            'labels': {
                level: self.labels[i][idx]
                for i, level in enumerate(['subclass', 'order', 'superfamily', 'family', 'genus', 'species'])
            }
        }


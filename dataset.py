"""
dataset.py
==========
Loads the CLIP features cached by extract_clip_features.py.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset


class MFNDFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = sorted(Path(feature_dir).glob("*.pt"))
        if len(self.files) == 0:
            raise RuntimeError(f"No cached feature files found in {feature_dir}. "
                                f"Run extract_clip_features.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rec = torch.load(self.files[idx])
        return rec["text_feat"], rec["image_feat"], rec["label"]


def collate_fn(batch):
    text_feats = torch.stack([b[0] for b in batch], dim=0)
    image_feats = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return text_feats, image_feats, labels

"""
config.py
=========
All hyperparameters below are taken verbatim from the paper
"MFND-DCL: Multimodal fake news detection based on dual contrastive learning"
(Qian, Guo, Chen, Wang, Ye — Information Processing & Management, 2026).

IMPORTANT REPRODUCIBILITY NOTE (read this before running anything):
---------------------------------------------------------------------
Section 4.4 of the paper states the contrastive temperature tau = 0.1 in the
"standard" training recipe. However, Table 5 (main results, Weibo, Ours ->
F1 = 0.9565) is numerically IDENTICAL to the tau = 0.2 row of Table 6 (the
temperature sensitivity sweep) and the lr = 1e-4 row of Table 7 (the learning
rate sweep). This means the headline numbers in Table 3/4/5 were actually
produced with tau = 0.2, not tau = 0.1 as literally written in Section 4.4.
To reproduce the paper's REPORTED numbers, use tau = 0.2 (set below).
If you want to follow Section 4.4 literally instead, change TAU to 0.1 -
expect ~1 point lower accuracy/F1, consistent with the tau=0.1 row in Table 6
(Acc 0.9447 vs 0.9549 at tau=0.2).
"""

import random
import numpy as np
import torch

SEED = 129  # paper: "fixing the global random seed to 129"


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Paper does not mention deterministic cudnn flags explicitly; setting
    # them is good practice for exact reproducibility across runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaseConfig:
    # ---- CLIP backbone ----
    clip_model_name = "openai/clip-vit-base-patch16"  # CLIP ViT-B/16, per Sec 3.3
    clip_text_dim = 512     # CLIP ViT-B/16 joint embedding dim
    clip_image_dim = 512
    freeze_clip = True      # "frozen CLIP encoder used only for feature extraction" (Sec 5.1.2)

    # ---- Gated Cross-Attention (GCA) ----
    d_model = 512            # feature dim d used for Q/K/V projections (Sec 3.4)
    d_k = 512                # projection dim d_k (paper uses same d for simplicity)
    gca_heads = 1             # paper's equations describe single-head cross-attention

    # ---- Dual Contrastive Learning (DCL) ----
    proj_dim = 128            # dp, projection-head output dim (Sec 3.5.1: dp = 128)
    lambda_sample = 0.5       # weight on sample-level (supervised contrastive) loss, Eq. 12
    lambda_modal = 0.3        # weight on modal-level (cross-modal InfoNCE) loss, Eq. 12
    tau = 0.2                 # contrastive temperature -- see note above (paper text says
                               # 0.1 but the reported main results correspond to 0.2)
    hard_negative_k = 5
    hard_negative_weight = 1.5
    min_temperature = 0.05
    max_temperature = 1.0

    # ---- Training (shared) ----
    max_epochs = 100          # Sec 4.4
    early_stopping_metric = "val_f1"
    early_stopping_patience = 15  # Matched from author's GitHub
                                   # "training terminated when val F1 ceases to improve";
    optimizer = "adam"
    grad_clip_norm = 1.0      # "gradient clipping applied to stabilize training" (Sec 4.4)

    device = "cuda" if torch.cuda.is_available() else "cpu"


class WeiboConfig(BaseConfig):
    dataset_name = "weibo"
    language = "zh"
    
    # Override dimensions for the 1026-dim ViT-H-14 + Time feature architecture
    clip_text_dim = 1026
    clip_image_dim = 1026
    d_model = 1026
    d_k = 1026
    
    batch_size = 128
    lr = 1e-4
    classifier_dropout = 0.3
    class_weights = None       # Weibo classes are roughly balanced (3748/3783, 1000/996)
    train_ratio, test_ratio = 0.8, 0.2   # Sec 4.1.1
    num_train_fake, num_train_real = 3748, 3783
    num_test_fake, num_test_real = 1000, 996


class PhemeConfig(BaseConfig):
    dataset_name = "pheme"
    language = "en"
    batch_size = 256
    lr = 6e-5
    classifier_dropout = 0.3   # not separately specified for PHEME; paper only
                                # differentiates batch size / lr / class weights explicitly,
                                # so the Weibo dropout value is reused here
    class_weights = [1.0, 2.4]  # "class weights of [1.0, 2.4] introduced into cross-entropy"
    train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2  # Sec 4.1.2
    events = [
        "charliehebdo", "ferguson", "germanwings-crash",
        "ottawashooting", "sydneysiege",
    ]


def get_config(dataset: str):
    dataset = dataset.lower()
    if dataset == "weibo":
        return WeiboConfig()
    elif dataset == "pheme":
        return PhemeConfig()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'weibo' or 'pheme'.")

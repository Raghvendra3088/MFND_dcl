import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RoViTModel(nn.Module):
    def __init__(self, mode='baseline', d_in=768, d_proj=256, dropout=0.3):
        super().__init__()
        self.mode = mode
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_in * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        if mode in ['supcon', 'hard_negative']:
            self.proj_t = nn.Linear(d_in, d_proj)
            self.proj_v = nn.Linear(d_in, d_proj)
            self.temp = nn.Parameter(torch.tensor(0.07))
            
    def forward(self, text_cls, img_cls):
        fused = torch.cat([text_cls, img_cls], dim=-1)
        logits = self.classifier(fused)
        if self.mode in ['supcon', 'hard_negative']:
            pt = F.normalize(self.proj_t(text_cls), dim=-1)
            pv = F.normalize(self.proj_v(img_cls), dim=-1)
            return logits, pt, pv
        return logits

def compute_supcon_loss(pt, pv, labels, temp, hard_neg=False):
    B = pt.size(0)
    sim = torch.matmul(pt, pv.T) / torch.clamp(temp, min=0.01, max=1.0)
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    neg_mask = 1.0 - pos_mask
    
    if hard_neg:
        weights = torch.where(neg_mask.bool(), torch.exp(torch.clamp(sim.detach(), max=2.0) * 0.5), torch.ones_like(sim))
        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True)[0]) * weights
    else:
        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True)[0])
        
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8
    log_prob = sim - torch.log(denom)
    
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss

def train_and_eval(args):
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    set_seed(args.seed)
    
    data_dir = args.data_dir
    tr_t = np.load(f'{data_dir}/train_text_cls.npy')
    tr_i = np.load(f'{data_dir}/train_img_cls.npy')
    tr_y = np.load(f'{data_dir}/train_labels.npy')
    
    te_t = np.load(f'{data_dir}/test_text_cls.npy')
    te_i = np.load(f'{data_dir}/test_img_cls.npy')
    te_y = np.load(f'{data_dir}/test_labels.npy')
    
    N = len(tr_y)
    indices = np.arange(N)
    np.random.shuffle(indices)
    val_sz = int(0.1 * N)
    va_idx, tr_idx = indices[:val_sz], indices[val_sz:]
    
    train_ds = TensorDataset(torch.from_numpy(tr_t[tr_idx]).float(), torch.from_numpy(tr_i[tr_idx]).float(), torch.from_numpy(tr_y[tr_idx]).long())
    val_ds = TensorDataset(torch.from_numpy(tr_t[va_idx]).float(), torch.from_numpy(tr_i[va_idx]).float(), torch.from_numpy(tr_y[va_idx]).long())
    test_ds = TensorDataset(torch.from_numpy(te_t).float(), torch.from_numpy(te_i).float(), torch.from_numpy(te_y).long())
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    model = RoViTModel(mode=args.mode, d_in=768, d_proj=256, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    best_val_f1 = -1.0
    best_state = None
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            bt_t, bt_i, by = [x.to(device) for x in batch]
            optimizer.zero_grad()
            if args.mode == 'baseline':
                logits = model(bt_t, bt_i)
                loss = F.cross_entropy(logits, by)
            else:
                logits, pt, pv = model(bt_t, bt_i)
                loss_cls = F.cross_entropy(logits, by)
                loss_con = compute_supcon_loss(pt, pv, by, model.temp, hard_neg=(args.mode == 'hard_negative'))
                loss = loss_cls + args.lambda_con * loss_con
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # Validation
        model.eval()
        v_preds, v_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                bt_t, bt_i, by = [x.to(device) for x in batch]
                out = model(bt_t, bt_i)
                logits = out[0] if isinstance(out, tuple) else out
                preds = logits.argmax(dim=-1).cpu().numpy()
                v_preds.extend(preds)
                v_trues.extend(by.cpu().numpy())
        val_f1 = f1_score(v_trues, v_preds, average='macro')
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Evaluation on Test Set using best validation checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    t_preds, t_trues, t_probs = [], [] , []
    with torch.no_grad():
        for batch in test_loader:
            bt_t, bt_i, by = [x.to(device) for x in batch]
            out = model(bt_t, bt_i)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            t_preds.extend(preds)
            t_trues.extend(by.cpu().numpy())
            t_probs.extend(probs)
            
    t_preds = np.array(t_preds)
    t_trues = np.array(t_trues)
    t_probs = np.array(t_probs)
    
    acc = accuracy_score(t_trues, t_preds)
    f1 = f1_score(t_trues, t_preds, average='macro')
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'preds_rovit_{args.mode}_seed{args.seed}.npy', t_preds)
    np.save(out_dir / f'probs_rovit_{args.mode}_seed{args.seed}.npy', t_probs)
    
    result = {'mode': args.mode, 'seed': args.seed, 'accuracy': acc, 'macro_f1': f1, 'best_val_f1': best_val_f1}
    print(f'[{args.mode} | seed {args.seed}] Test Acc: {acc*100:.2f}% | Test F1: {f1*100:.2f}% | Val F1: {best_val_f1*100:.2f}%')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'supcon', 'hard_negative'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_con', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str, default='/DATA/raghvendra3088/MFND_project/data/weibo/rovit')
    parser.add_argument('--out_dir', type=str, default='/DATA/raghvendra3088/MFND_project/files/predictions')
    args = parser.parse_args()
    res = train_and_eval(args)

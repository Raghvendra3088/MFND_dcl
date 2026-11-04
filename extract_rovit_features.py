import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, ViTImageProcessor, ViTModel

def extract_features(csv_path, out_dir, split_name, gpu_id=4, bs=64):
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu'
    print(f'=== Extracting {split_name} features on {device} ===')
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Models
    print('Loading RoBERTa and ViT...')
    model_path = '/DATA/raghvendra3088/models/chinese-roberta-wwm-ext'
    tok = AutoTokenizer.from_pretrained(model_path)
    text_enc = AutoModel.from_pretrained(model_path).to(device).eval()
    
    proc = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    img_enc = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device).eval()
    
    df = pd.read_csv(csv_path)
    N = len(df)
    D = 768
    
    text_cls_arr = np.lib.format.open_memmap(f'{out_dir}/{split_name}_text_cls.npy', mode='w+', dtype=np.float32, shape=(N, D))
    img_cls_arr = np.lib.format.open_memmap(f'{out_dir}/{split_name}_img_cls.npy', mode='w+', dtype=np.float32, shape=(N, D))
    labels = df['label'].values.astype(np.int64)
    np.save(f'{out_dir}/{split_name}_labels.npy', labels)
    
    base_dir = '/DATA/raghvendra3088/MFND_project/files'
    fallback_img = Image.new('RGB', (224, 224), (0, 0, 0))
    
    t0 = time.time()
    for s in tqdm(range(0, N, bs), desc=f'{split_name}'):
        e = min(s + bs, N)
        rows = df.iloc[s:e]
        
        # 1. Text processing
        texts = rows['text'].fillna('').tolist()
        t_inp = tok(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            t_out = text_enc(**t_inp)
            t_cls = t_out.last_hidden_state[:, 0, :].cpu().numpy()
        text_cls_arr[s:e] = t_cls
        
        # 2. Image processing
        imgs = []
        for p in rows['image_path']:
            if pd.isna(p):
                imgs.append(fallback_img)
            else:
                full_p = os.path.normpath(os.path.join(base_dir, p))
                if os.path.exists(full_p):
                    try:
                        with Image.open(full_p) as im:
                            imgs.append(im.convert('RGB'))
                    except Exception:
                        imgs.append(fallback_img)
                else:
                    imgs.append(fallback_img)
                    
        i_inp = proc(images=imgs, return_tensors='pt').to(device)
        with torch.no_grad():
            i_out = img_enc(**i_inp)
            i_cls = i_out.last_hidden_state[:, 0, :].cpu().numpy()
        img_cls_arr[s:e] = i_cls

    text_cls_arr.flush()
    img_cls_arr.flush()
    print(f'Done {split_name} ({N} samples) in {round(time.time() - t0, 2)}s')

if __name__ == '__main__':
    out = '/DATA/raghvendra3088/MFND_project/data/weibo/rovit'
    extract_features('/DATA/raghvendra3088/MFND_project/data/weibo/weibo_train.csv', out, 'train', gpu_id=4, bs=64)
    extract_features('/DATA/raghvendra3088/MFND_project/data/weibo/weibo_test.csv', out, 'test', gpu_id=4, bs=64)

import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from collections import Counter

class UnifiedCFGDataset(Dataset):
    """
    修复版 Dataset: 移除强制过采样，防止过拟合。
    依靠 Class Embeddings 来区分不同类别，而不是物理复制数据。
    """
    def __init__(self, data_sources, tokenizer, esm_model, device, 
                 augment_prob=0.2, mutation_rate=0.1): 
        self.tokenizer = tokenizer
        self.esm_model = esm_model.to(device)
        self.device = device
        self.augment_prob = augment_prob
        self.mutation_rate = mutation_rate
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        self.sequences = []
        self.labels = []

        print("Loading data sources (No Oversampling)...")
        for name, (df, label) in data_sources.items():
            if 'sequence' in df.columns:
                df = df.rename(columns={'sequence': 'SEQUENCE'})
            
            if label is not None:
                # 手动指定标签模式
                temp_df = df[['SEQUENCE']].dropna()
                current_seqs = temp_df['SEQUENCE'].astype(str).tolist()
                current_labels = [int(label)] * len(current_seqs)
                print(f"-> Loaded {len(current_seqs)} sequences from '{name}' with manual label {label}")
            else:
                # 自动读取标签模式
                if 'label' not in df.columns:
                    raise KeyError(f"Dataset '{name}' missing 'label' column.")
                temp_df = df[['SEQUENCE', 'label']].dropna()
                current_seqs = temp_df['SEQUENCE'].astype(str).tolist()
                current_labels = temp_df['label'].astype(int).tolist()
                print(f"-> Loaded {len(current_seqs)} sequences from '{name}' using its 'label' column")
            
            self.sequences.extend(current_seqs)
            self.labels.extend(current_labels)

        # 打印分布情况，不再进行强制平衡
        counts = Counter(self.labels)
        print(f"Final Training Distribution: {dict(counts)}")
        print(f"Total sequences: {len(self.sequences)}")
        
        # 混合数据
        combined = list(zip(self.sequences, self.labels))
        random.shuffle(combined)
        self.sequences, self.labels = zip(*combined)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # 仅在训练时开启突变增强，防止过拟合
        if random.random() < self.augment_prob:
            seq = self._mutate_sequence(seq)

        with torch.no_grad():
            inputs = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs.input_ids.squeeze(0)
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
            # 获取 ESM embedding
            outputs = self.esm_model(**inputs_on_device)
            # 去掉 CLS 和 EOS
            embedding = outputs.last_hidden_state[0, 1:-1]
            target_ids = input_ids[1:-1] 

        return {
            "embedding": embedding.cpu(), 
            "input_ids": target_ids.cpu(),
            "label": torch.tensor(label, dtype=torch.long)
        }
    
    def _mutate_sequence(self, seq):
        seq_list = list(seq)
        # 稍微增加一点突变率以增加数据多样性
        num_mutations = max(1, int(len(seq_list) * self.mutation_rate))
        positions = random.sample(range(len(seq_list)), num_mutations)
        for pos in positions:
            original_aa = seq_list[pos]
            new_aa = random.choice([aa for aa in self.amino_acids if aa != original_aa])
            seq_list[pos] = new_aa
        return ''.join(seq_list)

def collate_fn_cfg(batch):
    embeddings = [item['embedding'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    max_len = max(emb.shape[0] for emb in embeddings)
    
    padded_embeddings = []
    attention_masks = []
    padded_input_ids = []
    pad_token_id = 1 

    for emb, ids in zip(embeddings, input_ids):
        emb_len = emb.shape[0]
        pad_len = max_len - emb_len
        padded_embeddings.append(torch.cat([emb, torch.zeros(pad_len, emb.shape[1])], dim=0))
        attention_masks.append(torch.cat([torch.ones(emb_len), torch.zeros(pad_len)], dim=0))
        padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)], dim=0))
        
    return {
        "embeddings": torch.stack(padded_embeddings),
        "attention_mask": torch.stack(attention_masks),
        "input_ids": torch.stack(padded_input_ids),
        "labels": labels
    }
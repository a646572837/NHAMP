# src/util.py (新版本)
import torch
import numpy as np
import sys
from collections import Counter
from torch.utils.data import Dataset

# --- 通用工具 (保持不变) ---

def parse_fasta(file_path):
    # 此函数保持不变
    sequences = []
    current_sequence = ""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('>'):
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = ""
                else:
                    current_sequence += line
            if current_sequence:
                sequences.append(current_sequence)
    except FileNotFoundError:
        print(f"!!! FATAL ERROR: File not found: '{file_path}'")
        sys.exit()
    return sequences

class LengthSampler:
    # 此类保持不变
    def __init__(self, sequences, max_len=254):
        self.dataset_len = np.clip([len(t) for t in sequences], a_min=1, a_max=max_len)
        freqs = Counter(self.dataset_len)
        self.distrib = [freqs.get(i, 0) for i in range(max_len + 1)]
        self.distrib = np.array(self.distrib) / np.sum(self.distrib)

    def sample(self, num_samples):
        return np.random.choice(len(self.distrib), size=num_samples, p=self.distrib)

# --- 新增: 专用于Diffusion模型训练的工具 ---

class DiffusionDataset(Dataset):
    """
    一个简化的Dataset，专门为Denoiser的训练服务。
    它只关心序列本身，不处理标签或浓度。
    """
    def __init__(self, sequences, tokenizer, esm_model, device):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.esm_model = esm_model.to(device)
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        with torch.no_grad():
            inputs = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs.input_ids.squeeze(0)
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.esm_model(**inputs_on_device)
            # 提取氨基酸部分的嵌入 (移除前后的<cls>和<eos> token)
            embedding = outputs.last_hidden_state[0, 1:-1]
            target_ids = input_ids[1:]
        return {
            "embedding": embedding.cpu(), 
            "input_ids": target_ids.cpu()
        }

def diffusion_collate_fn(batch):
    embeddings = [item['embedding'] for item in batch]
    input_ids = [item['input_ids'] for item in batch]
    max_len = max(ids.shape[0] for ids in input_ids)
    padded_embeddings, attention_masks, padded_input_ids = [], [], []
    pad_token_id = 1

    for emb, ids in zip(embeddings, input_ids):
        emb_len, ids_len = emb.shape[0], ids.shape[0]
        padded_embeddings.append(torch.cat([emb, torch.zeros(max_len - emb_len, emb.shape[1])], dim=0))
        padded_input_ids.append(torch.cat([ids, torch.full((max_len - ids_len,), pad_token_id, dtype=torch.long)], dim=0))
        attention_masks.append(torch.cat([torch.ones(emb_len), torch.zeros(max_len - emb_len)], dim=0))
        
    return {
        "embeddings": torch.stack(padded_embeddings),
        "attention_mask": torch.stack(attention_masks),
        "input_ids": torch.stack(padded_input_ids)
    }
class ClassifierDataset(Dataset):
    """
    一个灵活的Dataset，为Classifier的训练服务。
    它直接从DataFrame中读取数据，并能选择性地处理浓度信息。
    """
    def __init__(self, dataframe, tokenizer, esm_model, device, concentration_col=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.esm_model = esm_model.to(device)
        self.device = device
        self.concentration_col = concentration_col # 可选的浓度列名

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        seq = row['SEQUENCE']
        label = row['label']
        
        item = {}
        with torch.no_grad():
            inputs = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.esm_model(**inputs_on_device)
            # 分类器通常使用整个序列的平均嵌入作为输入
            embedding = outputs.last_hidden_state[0, 1:-1].mean(dim=0)
        
        item['embedding'] = embedding.cpu()
        item['label'] = torch.tensor(label, dtype=torch.float32)
        
        # 如果指定了浓度列，则添加浓度信息
        if self.concentration_col:
            item['concentration'] = torch.tensor(row[self.concentration_col], dtype=torch.float32)
            
        return item

def classifier_collate_fn(batch):
    """
    为ClassifierDataset配套的collate_fn。
    因为__getitem__已经将嵌入处理为固定长度的平均向量，所以这里无需填充。
    """
    embeddings = torch.stack([item['embedding'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    collated_batch = {
        "embeddings": embeddings,
        "labels": labels,
    }
    
    # 如果批次中包含浓度信息，则一并处理
    if 'concentration' in batch[0]:
        concentrations = torch.stack([item['concentration'] for item in batch])
        collated_batch['concentration'] = concentrations
        
    return collated_batch
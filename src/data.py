from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import omegaconf
from typing import Union, List, Tuple, Literal
import numpy as np

class IMDBDataset(Dataset):
    def __init__(self, data_config: omegaconf.DictConfig, split: Literal['train', 'valid', 'test']):
        tokenizer = AutoTokenizer.from_pretrained(data_config.pretrained_name)
        imdb_train = load_dataset(data_config.dataset_name, split='train')
        imdb_test  = load_dataset(data_config.dataset_name, split='test')
        
        # 전체 데이터 병합
        texts = list(imdb_train['text']) + list(imdb_test['text'])
        labels = list(imdb_train['label']) + list(imdb_test['label'])
        data_size = len(texts)
        
        # split_ratio 사용
        split_ratio = data_config.split_ratio
        split_ratio = [float(x) for x in split_ratio]
        train_cut  = int(data_size * split_ratio[0])
        valid_cut  = train_cut + int(data_size * split_ratio[1])
        
        # 셔플/시드 고정
        rng = np.random.default_rng(seed=data_config.seed)
        idx = np.arange(data_size)
        rng.shuffle(idx)

        if split == 'train':
            select_idx = idx[:train_cut]
        elif split == 'valid':
            select_idx = idx[train_cut:valid_cut]
        else:
            select_idx = idx[valid_cut:]
        
        self.split = split
        self.texts  = [texts[i]  for i in select_idx]
        self.labels = [labels[i] for i in select_idx]
        self.tokenizer = tokenizer
        self.max_len   = data_config.max_len
        
        print(f">> SPLIT: {self.split} | Total Data Length: {len(self.texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Tuple[dict, int]:        
        """
        Inputs :
            idx : int
        Outputs :
            inputs : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
            }
            label : int
        """
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        data = {k: v.squeeze(0) for k, v in encoding.items()}
        label = int(self.labels[idx])
        return data, label


    @staticmethod
    def collate_fn(batch: List[Tuple[dict, int]]) -> dict:
        """
        Inputs :
            batch : List[Tuple[dict, int]]
        Outputs :
            data_dict : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        input_ids = torch.stack([item[0]['input_ids'] for item in batch])
        attention_mask = torch.stack([item[0]['attention_mask'] for item in batch])
        
        # token_type_ids가 있을 때만 stack, 없으면 아예 넘기지 않기!
        if 'token_type_ids' in batch[0][0]:
            token_type_ids = torch.stack([item[0]['token_type_ids'] for item in batch])
        else:
            token_type_ids = None

        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        
        batch_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels
        }
        if token_type_ids is not None:
            batch_dict['token_type_ids'] = token_type_ids
        # else: key 자체를 생략

        return batch_dict

def get_dataloader(data_config, split, batch_size):
    dataset = IMDBDataset(data_config, split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=IMDBDataset.collate_fn
    )
    return dataloader
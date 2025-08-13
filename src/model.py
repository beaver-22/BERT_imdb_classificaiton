from typing import Tuple, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig                        # ①
from transformers import AutoModel, AutoConfig

class EncoderForClassification(nn.Module):
    def __init__(self, model_config: DictConfig):        # ②
        """
        model_config 예시 (OmegaConf YAML):
        model:
          pretrained_name: bert-base-uncased
          num_labels: 2
          dropout: 0.1
        """
        super().__init__()

        # 1) 사전훈련 Encoder 로드
        self.encoder = AutoModel.from_pretrained(model_config.pretrained_name)
        hidden_size = self.encoder.config.hidden_size

        # 2) 분류 헤드
        self.dropout = nn.Dropout(model_config.dropout)
        self.classifier = nn.Linear(hidden_size, model_config.num_labels) #label 2개

        # 3) 손실 함수 (Cross-Entropy)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None
    ) -> dict:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True
        }
        # 모델 config의 타입 확인, 혹은 token_type_ids가 None이 아니면만 추가
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**model_inputs)
        cls_emb = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls_emb))

        loss = None
        if label is not None:
            loss = self.criterion(logits, label)

        return {'logits': logits, 'loss': loss}

        """
        Inputs : 
            input_ids : (batch_size, max_seq_len)
            attention_mask : (batch_size, max_seq_len)
            token_type_ids : (batch_size, max_seq_len) # only for BERT
            label : (batch_size)
        Outputs :
            logits : (batch_size, num_labels)
            loss : (1)
        """
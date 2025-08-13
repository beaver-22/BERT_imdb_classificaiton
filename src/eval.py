import torch
import numpy as np
import random
from omegaconf import OmegaConf

from model import EncoderForClassification
from data import get_dataloader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----- 1. 시드 고정 -----
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# ----- 2. 모델/데이터 로드 -----
# configs는 YAML이나 omegaconf에서 불러온 dict라고 가정
configs = OmegaConf.load("../BERT_imdb/configs/bert.yaml")
# configs = OmegaConf.load("../BERT_imdb/configs/modern_bert.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderForClassification(configs.model).to(device)

# 저장된 모델 불러오기
model_path = "../BERT_imdb/checkpoints/imdb-bert/checkpoint-epoch-05-acc-0.8846.pt"  # 또는 원하는 경로
# model_path = "../BERT_imdb/checkpoints/imdb-modernbert/checkpoint-epoch-05-acc-0.9092.pt"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 테스트 데이터로더 준비
test_loader = get_dataloader(configs.data, split="test", batch_size=configs.train.batch_size)

# ----- 3. 테스트 평가 -----
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)
        outputs = model(**inputs)
        logits = outputs['logits']
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

# ----- 4. 전체 지표 출력 -----
accuracy  = accuracy_score (all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall    = recall_score   (all_labels, all_preds, average='binary')
f1        = f1_score      (all_labels, all_preds, average='binary')

print(f"Test Results (seed=42):")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1-score : {f1:.4f}")

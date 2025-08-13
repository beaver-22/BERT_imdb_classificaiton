import wandb 
from tqdm import tqdm
import os
import datetime
import glob

import torch
import torch.nn
import omegaconf
from omegaconf import OmegaConf

from utils import load_config, set_logger
from model import EncoderForClassification
from data import get_dataloader
from utils import get_optimizer

# torch.cuda.set_per_process_memory_fraction(11/24) -> 김재희 로컬과 신입생 로컬의 vram 맞추기 용도. 과제 수행 시 삭제하셔도 됩니다. 
# model과 data에서 정의된 custom class 및 function을 import합니다.
# 학습·검증 루프, wandb 로깅
"""
여기서 import 하시면 됩니다. 
"""
def train_iter(model, inputs, optimizer, device, epoch):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    loss = outputs['loss']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    wandb.log({'train_loss' : loss.item()})
    return loss

def valid_iter(model, inputs, device):
    inputs = {key : value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    loss = outputs['loss']
    accuracy = calculate_accuracy(outputs['logits'], inputs['label'])    
    return loss, accuracy

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def main(configs: omegaconf.DictConfig):
    # 1. 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. wandb 프로젝트 시작
    timestamp = datetime.datetime.now().strftime('%H%M')
    run_name = configs.log.wandb_name.replace('{timestamp}', timestamp)

    wandb.init(
        project=configs.log.wandb_project,
        entity=configs.log.wandb_entity,
        config=OmegaConf.to_container(configs, resolve=True),
        name=run_name                # << 치환된 값으로 넘겨줌!
    )
    
    # 3. 모델 로드 및 GPU로 이동
    model = EncoderForClassification(configs.model).to(device)
    wandb.watch(model)

    # 4. 데이터 로더 준비
    train_loader = get_dataloader(configs.data, 'train', batch_size=configs.train.batch_size)
    val_loader   = get_dataloader(configs.data, 'valid', batch_size=configs.train.batch_size)
    test_loader  = get_dataloader(configs.data, 'test', batch_size=configs.train.batch_size)

    # 5. 옵티마이저 준비
    optimizer = get_optimizer(
        configs.train.optimizer,      # default.yaml에서 'adam'
        model.parameters(),
        configs.train.lr
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(
        configs.model_saving.best_model_dir, 
        f"best_model_{configs.model_saving.run_time}.pt"
    )
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    checkpoint_dir = configs.checkpointing.output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_total_limit = configs.checkpointing.save_total_limit
    checkpoint_format = configs.checkpointing.checkpoint_name_format  # e.g., "checkpoint-epoch-{epoch:02d}-acc-{eval_accuracy:.4f}"

    # 6. 학습 및 검증 루프 실행
    for epoch in range(configs.train.num_epochs):
        # ---- 학습 ----
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Train-{epoch+1}", leave=False):
            loss = train_iter(model, batch, optimizer, device, epoch)
            train_losses.append(loss.item() if torch.is_tensor(loss) else float(loss))
        avg_train_loss = sum(train_losses) / len(train_losses)
        wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch+1})

        # ---- 검증 ----
        val_losses, val_accs = [], []
        for batch in tqdm(val_loader, desc=f"Val-{epoch+1}", leave=False):
            v_loss, v_acc = valid_iter(model, batch, device)
            val_losses.append(v_loss.item() if torch.is_tensor(v_loss) else float(v_loss))
            val_accs.append(v_acc)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = sum(val_accs) / len(val_accs)
        wandb.log({"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc, "epoch": epoch+1})

        # ---- checkpoint 저장 (매 epoch) ----
        checkpoint_name = checkpoint_format.format(
            epoch=epoch+1,
            eval_accuracy=avg_val_acc
        )
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name + ".pt")

        # 저장 구성요소들: model, optimizer 등
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 포함하려면: 'scheduler_state_dict': scheduler.state_dict(),
            # tokenizer 등은 별도로 저장 필요 (보통 configs 등)
            'val_acc': avg_val_acc
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # ---- 최신 N개만 남기고 자동 삭제 ----
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")), key=os.path.getmtime)
        if len(checkpoints) > save_total_limit:
            for ckpt_to_delete in checkpoints[:-save_total_limit]:
                try:
                    os.remove(ckpt_to_delete)
                    print(f"Delete old checkpoint: {ckpt_to_delete}")
                except Exception as e:
                    print(f"Could not delete {ckpt_to_delete}: {e}")

        # ---- best model 별도 저장 (기존 방식) ----
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {epoch+1} epoch (val_acc={best_val_acc:.4f})")

    # 7. 테스트 평가 (최고 성능 모델 기준)
    model.load_state_dict(torch.load(best_model_path))
    test_losses, test_accs = [], []
    for batch in tqdm(test_loader, desc="Test", leave=False):
        t_loss, t_acc = valid_iter(model, batch, device)
        test_losses.append(t_loss.item() if torch.is_tensor(t_loss) else float(t_loss))
        test_accs.append(t_acc)
    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_acc = sum(test_accs) / len(test_accs)

    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
    wandb.log({"test_loss": avg_test_loss, "test_acc": avg_test_acc})
    
if __name__ == "__main__":
    configs = load_config()
    run_time = datetime.datetime.now().strftime('%M%S')
    configs.model_saving.run_time = run_time  # (원하면 configs에 직접 추가)
    main(configs)
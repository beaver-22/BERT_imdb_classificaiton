import omegaconf
from omegaconf import OmegaConf
import torch
import logging
import sys
import os
# 공통 유틸(시드 고정, metric 등)


def load_config():
    # 1. 기본 config 경로
    config_path = "configs/default.yaml"

    # 2. 커맨드라인 인자에서 --config {경로} 전달받기
    for idx, arg in enumerate(sys.argv):
        if arg in ("--config", "-c") and (idx + 1) < len(sys.argv):
            config_path = sys.argv[idx + 1]

    # 3. 환경변수에서 지정 가능
    config_path = os.environ.get("CONFIG_PATH", config_path)

    # 4. 경로 실제 존재하는지 체크
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # 5. OmegaConf로 yaml 파싱 후 반환
    return OmegaConf.load(config_path)

def get_optimizer(name, parameters, lr):
    name = name.lower()
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr)
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr)
    elif name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    
# src/utils.py

import logging

def set_logger(log_file=None, log_level="INFO"):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    if log_file:
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    return logger

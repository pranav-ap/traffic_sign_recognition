import torch
import torch.nn as nn
import timm

from config import config
from utils import logger, count_params

torch.set_float32_matmul_precision('medium')


class TrafficSignRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()

        NUM_CLASSES = 43

        self.model = timm.create_model(
            'mobilenetv4_conv_small',
            pretrained=True,
            num_classes=NUM_CLASSES,
        )

        total_trainable_params = count_params(self.model)
        logger.info(f"Number of Trainable Parameters : {total_trainable_params}")

    def forward(self, images):
        out = self.model(images)
        return out

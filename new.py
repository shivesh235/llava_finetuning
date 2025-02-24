import csv
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import requests
from io import BytesIO
import wandb
from typing import Optional, Dict, List
import evaluate
from pathlib import Path  
# from lava_train_slack import SlakeDataModule, LlavaFineTuner
from lava_train_vqarad import VQARADDataModule, LlavaFineTuner

# Initialize processor
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    
# Model and training configuration
MODEL_NAME = "llava-finetuned-vqarad"
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 2
NUM_WORKERS = 8
MAX_LENGTH = 2048
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
MAX_EPOCHS = 2
BASE_IMG_DIR = './slack_imgs'
GRADIENT_ACCUMULATION_STEPS = 8

processor = AutoProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Set padding token

# # Initialize datamodule
# datamodule = SlakeDataModule(
# processor=processor,
# base_img_dir=BASE_IMG_DIR,
# train_batch_size=TRAIN_BATCH_SIZE,
# eval_batch_size=EVAL_BATCH_SIZE,
# num_workers=NUM_WORKERS,
# max_length=MAX_LENGTH
# )

# Initialize datamodule
datamodule = VQARADDataModule(
    processor=processor,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    max_length=MAX_LENGTH
)

# Initialize model
model = LlavaFineTuner(
model_name=MODEL_NAME,
learning_rate=LEARNING_RATE,
weight_decay=WEIGHT_DECAY,
warmup_steps=WARMUP_STEPS
)

device = 'cuda'
datamodule.setup()
print('setup')
test_loader = datamodule.test_dataloader()
print('test_loader')

model.eval()
print('model eval')
device = next(model.parameters()).device

printed = 0
with torch.no_grad():
    for batch in test_loader:
        # print(batch)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        # Generate predictions
        generated_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=8,
            num_beams=1
        )

        # Decode full outputs
        full_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)
        references = processor.batch_decode(batch["labels"], skip_special_tokens=True)

        print('preds:', full_outputs)
        print('target:', references)
        print('**********')
        
        printed += 1
        if printed >= 10:
            break
            


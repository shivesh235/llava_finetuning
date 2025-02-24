import csv
import os
import re
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
import numpy as np
import sklearn.metrics as sklm
from torchmetrics import Metric

# ---------------------------------------------------------------------------
# Helper function to preprocess text for exact match accuracy calculation.
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------------------------
# VQARAD Tokenwise
# class VQARADScore(Metric):
#     def __init__(self, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         # Define states for tracking scores
#         self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

#         self.add_state("close_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("close_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

#         self.add_state("open_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
#         self.add_state("open_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

#         # Track best scores
#         self.best_score = 0
#         self.best_close_score = 0
#         self.best_open_score = 0

#     def update(self, logits, target, types=None):
#         """Update metric state with new batch of predictions and targets."""
#         device = self.score.device  # Ensure all tensors are on the same device

#         # Convert logits to predicted class indices
#         preds = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_len]

#         # Ensure target is on the same device and has correct shape
#         target = target.to(device)

#         # Compare predictions with target
#         correct = (preds == target).float()  # Shape: [batch_size, seq_len]
        
#         # Compute accuracy per sample (averaged over sequence length)
#         sample_acc = correct.mean(dim=-1)  # Shape: [batch_size]

#         # Update total score
#         self.score += sample_acc.sum()
#         self.total += target.size(0)

#         # Split scores into open and close types
#         if types is not None:
#             types = torch.tensor(types, device=device)  # Ensure `types` is a tensor
#             close_mask = (types == 0).to(device)
#             open_mask = (types == 1).to(device)

#             if close_mask.any():
#                 self.close_score += sample_acc[close_mask.nonzero(as_tuple=True)].sum()
#                 self.close_total += close_mask.sum().float()

#             if open_mask.any():
#                 self.open_score += sample_acc[open_mask.nonzero(as_tuple=True)].sum()
#                 self.open_total += open_mask.sum().float()

#     def compute(self):
#         """Compute the final metric value."""
#         return self.score / self.total if self.total > 0 else torch.tensor(0.0)

#     def get_best_score(self):
#         """Get the best overall score."""
#         self.sync()
#         score = self.compute()
#         if score > self.best_score:
#             self.best_score = score
#             self.best_close_score = self.close_score / self.close_total if self.close_total > 0 else 0
#             self.best_open_score = self.open_score / self.open_total if self.open_total > 0 else 0
#         self.unsync()
#         return self.best_score

#     def get_best_close_score(self):
#         """Get the best close-ended score."""
#         return self.best_close_score

#     def get_best_open_score(self):
#         """Get the best open-ended score."""
#         return self.best_open_score


# Updated VQARADScore without sync/unsync calls
class VQARADScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Define states for tracking scores
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("close_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("close_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("open_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("open_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Track best scores
        self.best_score = 0.0
        self.best_close_score = 0.0
        self.best_open_score = 0.0

    def update(self, preds, targets, types=None):
        """Update metric state with new batch of predictions and targets."""
        batch_size = len(preds)
        self.total += batch_size

        # Convert to lowercase and strip whitespace for comparison
        preds = [p.lower().strip() for p in preds]
        targets = [t.lower().strip() for t in targets]
        print('preds', preds)
        print('targets', targets)

        # Check if target sequence is present in prediction
        for pred, target in zip(preds, targets):
            if target in pred:
                self.correct += 1

        # Split scores into open and close types if provided
        if types is not None:
            for pred, target, type_ in zip(preds, targets, types):
                if type_ == 0:  # Close-ended
                    self.close_total += 1
                    if target in pred:
                        self.close_correct += 1
                elif type_ == 1:  # Open-ended
                    self.open_total += 1
                    if target in pred:
                        self.open_correct += 1

    def compute(self):
        """Compute the final overall metric value."""
        return self.correct / self.total if self.total > 0 else torch.tensor(0.0)

    def get_best_score(self):
        """Get the best overall score."""
        score = self.compute()
        if score > self.best_score:
            self.best_score = score.item()
            self.best_close_score = (self.close_correct / self.close_total).item() if self.close_total > 0 else 0.0
            self.best_open_score = (self.open_correct / self.open_total).item() if self.open_total > 0 else 0.0
        return self.best_score

    def get_best_close_score(self):
        """Get the best close-ended score."""
        return self.best_close_score

    def get_best_open_score(self):
        """Get the best open-ended score."""
        return self.best_open_score

# ---------------------------------------------------------------------------
# Data loading and processing

class SlakeDataset(Dataset):
    def __init__(self, dataset_split, processor, base_img_dir='./imgs', max_length=2048):
        """
        Args:
            dataset_split: HuggingFace dataset split
            processor: Llava processor
            base_img_dir: Base directory containing images
            max_length: Maximum sequence length
        """
        # Filter dataset to only include English questions
        self.dataset = [item for item in dataset_split if item.get('q_lang') == "en"]
        self.processor = processor
        self.base_img_dir = Path(base_img_dir)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Construct full image path
        image_path = str(self.base_img_dir / item['img_name'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        # Construct prompt
        conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a medical assistant who analyse the image and answer the question in very few words."}],
                },
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": item['question']},
                    {"type": "image"},
                    ],
                },
            ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Tokenize prompt + image
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].squeeze(0)

        # Tokenize only the answer (without prompt)
        target = self.processor.tokenizer(
            item['answer'],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).input_ids.squeeze(0)

        # Extract answer type (open-ended or close-ended)
        answer_type = item.get('answer_type', "OPEN")  # Default to 'OPEN' if missing

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "labels": target,
            "answer_type": answer_type
        }
    
class SlakeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        processor,
        base_img_dir: str = './img',  # Updated default path
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 4,
        max_length: int = 2048
    ):
        super().__init__()
        self.processor = processor
        self.base_img_dir = base_img_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.max_length = max_length

    def setup(self, stage: Optional[str] = None):
        # Load dataset
        dataset = load_dataset("BoKelvin/SLAKE")

        # Create dataset instances
        self.train_dataset = SlakeDataset(dataset["train"], self.processor, self.base_img_dir, self.max_length)
        self.val_dataset = SlakeDataset(dataset["validation"], self.processor, self.base_img_dir, self.max_length)
        self.test_dataset = SlakeDataset(dataset["test"], self.processor, self.base_img_dir, self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

# ---------------------------------------------------------------------------
# Llava FineTuner Module with updated generated response accuracy calculation

class LlavaFineTuner(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        trainable_layers: int = 2  # Change this to control how many layers are trainable
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze selected layers (example: layers with indices in these ranges)
        for i, (name, module) in enumerate(self.model.named_modules()):
            if i in range(250, 302) or i in range(650, 725):
                for param in module.parameters():
                    param.requires_grad = True

        # Metrics
        self.rouge = evaluate.load('rouge')

        # VQA-RAD accuracy tracking
        self.train_vqa_open = VQARADScore()
        self.train_vqa_close = VQARADScore()
        self.val_vqa_open = VQARADScore()
        self.val_vqa_close = VQARADScore()
        self.test_vqa_open = VQARADScore()
        self.test_vqa_close = VQARADScore()

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def step(self, batch):
        images, questions, targets, answer_types = (
            batch["pixel_values"], 
            batch["input_ids"], 
            batch["labels"], 
            batch["answer_type"]
        )
        batch.pop("answer_type", None)

        outputs = self(**batch)
        loss = outputs.loss  # Assuming model provides a loss
        logits = outputs.logits

        if logits.shape[0] != targets.shape[0]:
            raise ValueError(f"Shape Mismatch: Logits {logits.shape}, Targets {targets.shape}")

        targets = targets.long()

        # Generate predictions
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            max_new_tokens=8,
            num_beams=1
        )

        # Decode generated and reference responses
        full_responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True) 
        references = self.processor.batch_decode(batch["labels"], skip_special_tokens=True)

        # Apply improved prompt stripping (choose based on data format)
        stripped_responses = []
        for i, response in enumerate(full_responses):
            stripped_responses.append(response.split("ASSISTANT: ")[1])
            

        return logits, targets, loss, answer_types, stripped_responses, references
    
    def training_step(self, batch, batch_idx):
        logits, targets, loss, answer_types, preds, targets = self.step(batch)

        # Compute accuracy separately for open-ended and close-ended questions
        for i, answer_type in enumerate(answer_types):
            if answer_type == "OPEN":
                self.train_vqa_open.update(preds[i], targets[i], types=1)
            else:
                self.train_vqa_close.update(preds[i], targets[i], types=0)

        # Compute and log metrics explicitly
        train_vqa_open_value = self.train_vqa_open.compute() if self.train_vqa_open.total > 0 else 0
        train_vqa_close_value = self.train_vqa_close.compute() if self.train_vqa_close.total > 0 else 0

        # Metrics to log
        metrics = {
            "step": self.global_step,
            "train_loss": loss.item(),
            "train_vqa_open": train_vqa_open_value,  # Replace with actual metric
            "train_vqa_close": train_vqa_close_value,  # Replace with actual metric
        }

        # Save metrics to local CSV file
        log_file = "training_log_slake.csv"
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()  # Write header only once
            writer.writerow(metrics)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_vqa_open", train_vqa_open_value, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_vqa_close", train_vqa_close_value, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, targets, loss, answer_types, preds, targets = self.step(batch)

        # Compute accuracy separately for open-ended and close-ended questions
        for i, answer_type in enumerate(answer_types):
            if answer_type == "OPEN":
                self.train_vqa_open.update(preds[i], targets[i], types=1)
            else:
                self.train_vqa_close.update(preds[i], targets[i], types=0)


        # Compute metrics based on stripped output
        try:
            rouge_scores = self.rouge.compute(predictions=preds, references=targets)
        except ZeroDivisionError:
            rouge_scores = 0.0

        # rouge_scores = self.rouge.compute(predictions=stripped_responses, references=references)
        # bleu_score = self.bleu.compute(predictions=stripped_responses, references=references)

        # Compute and log metrics explicitly
        val_vqa_open_value = self.val_vqa_open.compute() if self.train_vqa_open.total > 0 else 0
        val_vqa_close_value = self.val_vqa_close.compute() if self.train_vqa_close.total > 0 else 0

        # Metrics to log
        metrics = {
            "step": self.global_step,
            "val_loss": loss.item(),
            "val_vqa_open_value": val_vqa_open_value,  # Replace with actual metric
            "val_vqa_close_value": val_vqa_close_value,  # Replace with actual metric
            "val_rouge1": rouge_scores['rouge1'],
            "val_rougeL": rouge_scores['rougeL']
        }

        # Save metrics to local CSV file
        log_file = "validation_log_slake.csv"
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()  # Write header only once
            writer.writerow(metrics)
        

        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_vqa_open", val_vqa_open_value, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_vqa_close", val_vqa_close_value, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_rouge1", rouge_scores['rouge1'], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_rougeL", rouge_scores['rougeL'], prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits, targets, loss, answer_types, preds, targets = self.step(batch)

        # Compute accuracy separately for open-ended and close-ended questions
        for i, answer_type in enumerate(answer_types):
            if answer_type == "OPEN":
                self.train_vqa_open.update(preds[i], targets[i], types=1)
            else:
                self.train_vqa_close.update(preds[i], targets[i], types=0)

        # Compute metrics based on stripped output
        try:
            rouge_scores = self.rouge.compute(predictions=preds, references=targets)
        except ZeroDivisionError:
            rouge_scores = 0.0

        # rouge_scores = self.rouge.compute(predictions=stripped_responses, references=references)
        # bleu_score = self.bleu.compute(predictions=stripped_responses, references=references)

        # Compute and log metrics explicitly
        test_vqa_open_value = self.test_vqa_open.compute() if self.train_vqa_open.total > 0 else 0
        test_vqa_close_value = self.test_vqa_close.compute() if self.train_vqa_close.total > 0 else 0

        # Metrics to log
        metrics = {
            "step": self.global_step,
            "test_loss": loss.item(),
            "test_vqa_open_value": test_vqa_open_value,  # Replace with actual metric
            "test_vqa_close_value": test_vqa_close_value,  # Replace with actual metric
            "test_rouge1": rouge_scores['rouge1'],
            "test_rougeL": rouge_scores['rougeL']
        }

        # Save metrics to local CSV file
        log_file = "test_log_slake.csv"
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()  # Write header only once
            writer.writerow(metrics)
        

        # Log metrics
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_vqa_open", test_vqa_open_value, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_vqa_close", test_vqa_close_value, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_rouge1", rouge_scores['rouge1'], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_rougeL", rouge_scores['rougeL'], prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


def main():
    wandb.init(project='llava-slack-24')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    
    # Model and training configuration
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    NUM_WORKERS = 8
    MAX_LENGTH = 2048
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 50
    MAX_EPOCHS = 2
    BASE_IMG_DIR = './slack_imgs'
    GRADIENT_ACCUMULATION_STEPS = 8  # Simulates a larger batch size without memory spikes
    
    # Initialize processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Set padding token
    
    # Initialize datamodule
    datamodule = SlakeDataModule(
        processor=processor,
        base_img_dir=BASE_IMG_DIR,
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

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="llava-finetuned-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )

    # Initialize trainer
    trainer = pl.Trainer(
        fast_dev_run=2,
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=[0],
        precision="bf16-mixed",
        # strategy="auto", 
        # strategy=DDPStrategy(find_unused_parameters=False),
        gradient_clip_val=1.0,
        accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS,  # ✅ Reduce memory usage
        callbacks=[early_stopping],# [checkpoint_callback, early_stopping],
        logger=WandbLogger()      
    )

    # Train and evaluate
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    # Get final accuracy
    train_open_score = model.train_vqa_open.compute().item()
    train_close_score = model.train_vqa_close.compute().item()
    test_open_score = model.test_vqa_open.compute().item()
    test_close_score = model.test_vqa_close.compute().item()

    print(f"Train Open-Ended Accuracy: {train_open_score}")
    print(f"Train Close-Ended Accuracy: {train_close_score}")
    print(f"Test Open-Ended Accuracy: {test_open_score}")
    print(f"Test Close-Ended Accuracy: {test_close_score}")
    
    # Save final model
    model.model.save_pretrained("llava-finetuned-slake-2")
    processor.save_pretrained("llava-finetuned-slake-2")

    # test_model_performance(model, datamodule, processor, num_samples=20)

if __name__ == "__main__":
    main()

import csv
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset
from transformers import AutoProcessor, LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import requests
from io import BytesIO
import wandb
from typing import Optional, Dict, List
import evaluate
from pathlib import Path    
from torchmetrics import Metric


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total


class VQARADScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Define states for tracking scores
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("close_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("close_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("open_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("open_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Track best scores
        self.best_score = 0
        self.best_close_score = 0
        self.best_open_score = 0

    def update(self, logits, target, types=None):
        """Update metric state with new batch of predictions and targets."""
        device = self.score.device  # Ensure all tensors are on the same device

        # Convert logits to predicted class indices
        preds = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_len]

        # Ensure target is on the same device and has correct shape
        target = target.to(device)

        # Compare predictions with target
        correct = (preds == target).float()  # Shape: [batch_size, seq_len]
        
        # Compute accuracy per sample (averaged over sequence length)
        sample_acc = correct.mean(dim=-1)  # Shape: [batch_size]

        # Update total score
        self.score += sample_acc.sum()
        self.total += target.size(0)

        # Split scores into open and close types
        if types is not None:
            types = torch.tensor(types, device=device)  # Ensure `types` is a tensor
            close_mask = (types == 0).to(device)
            open_mask = (types == 1).to(device)

            if close_mask.any():
                self.close_score += sample_acc[close_mask.nonzero(as_tuple=True)].sum()
                self.close_total += close_mask.sum().float()

            if open_mask.any():
                self.open_score += sample_acc[open_mask.nonzero(as_tuple=True)].sum()
                self.open_total += open_mask.sum().float()

    def compute(self):
        """Compute the final metric value."""
        return self.score / self.total if self.total > 0 else torch.tensor(0.0)

    def get_best_score(self):
        """Get the best overall score."""
        self.sync()
        score = self.compute()
        if score > self.best_score:
            self.best_score = score
            self.best_close_score = self.close_score / self.close_total if self.close_total > 0 else 0
            self.best_open_score = self.open_score / self.open_total if self.open_total > 0 else 0
        self.unsync()
        return self.best_score

    def get_best_close_score(self):
        """Get the best close-ended score."""
        return self.best_close_score

    def get_best_open_score(self):
        """Get the best open-ended score."""
        return self.best_open_score

################################################

class VQARADDataset(Dataset):
    def __init__(self, dataset_split, processor, max_length=512):
        self.dataset = dataset_split
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load image
        image = item['image'].convert('RGB')

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

        # Tokenize prompt and process image
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        if item['answer'] == 'yes' or item['answer'] == 'no':
            answer_type = 'CLOSE'
        else:
            answer_type = 'OPEN'

        # Tokenize the answer
        target = self.processor.tokenizer(
            item['answer'],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).input_ids.squeeze(0)

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "pixel_values": inputs.pixel_values.squeeze(0),
            "labels": target,
            "answer_type": answer_type
        }
    

class VQARADDataModule(pl.LightningDataModule):
    def __init__(
        self,
        processor,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 2048
    ):
        """
        Args:
            processor: Llava processor for text and image processing
            train_batch_size: Batch size for training
            eval_batch_size: Batch size for validation/testing
            num_workers: Number of workers for data loading
            max_length: Maximum sequence length for tokenization
        """
        super().__init__()
        self.processor = processor
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.save_hyperparameters(ignore=['processor'])

    def setup(self, stage: Optional[str] = None):
        # Load dataset from Hugging Face
        dataset = load_dataset("flaviagiammarino/vqa-rad")
        
        # Create dataset instances
        self.train_dataset = VQARADDataset(
            dataset["train"], 
            self.processor, 
            self.max_length
        )
        self.val_dataset = VQARADDataset(
            dataset["validation"] if "validation" in dataset else dataset["test"], 
            self.processor, 
            self.max_length
        )
        self.test_dataset = VQARADDataset(
            dataset["test"], 
            self.processor, 
            self.max_length
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
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
        
        # # Configure 4-bit quantization
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4"  # normalized float 4
        # )

        # # Model loading configuration
        # model_config = {
        #     "device_map": "auto",  # Automatically distribute across available GPUs
        #     "torch_dtype": torch.float16,
        #     "quantization_config": quantization_config
        # }

        # # Load model with 4-bit quantization
        # self.model = LlavaForConditionalGeneration.from_pretrained(
        #     model_name,
        #     **model_config
        # )

        # Load model
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        for i, (name, module) in enumerate(self.model.named_modules()):
            # print(i, name, module)

            if i in range(250, 302) or i in range(650, 725):
                for param in module.parameters():
                    param.requires_grad = True


        # # List all transformer blocks inside the model
        # transformer_blocks = [module for name, module in self.model.named_modules() if "transformer" in name.lower()]

        # if not transformer_blocks:
        #     raise ValueError("No transformer layers found. Check model architecture.")

        # # Unfreeze only the last `trainable_layers` blocks
        # for block in transformer_blocks[-trainable_layers:]:  # Get last N layers
        #     for param in block.parameters():
        #         param.requires_grad = True

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
        # print('**Forward pass**')
        # print(self.model.config)
        # for key in inputs.keys():
        #     print(key)
        #     print(inputs[key].shape)
        return self.model(**inputs)
    
    def step(self, batch):
        images, questions, targets, answer_types = (
            batch["pixel_values"], 
            batch["input_ids"], 
            batch["labels"], 
            batch["answer_type"]
        )

        batch.pop("answer_type", None)

        # Get model outputs
        outputs = self(**batch)

        # Extract loss and logits from outputs
        loss = outputs.loss  # Assuming model provides a loss
        logits = outputs.logits  # Model outputs logits


        # Ensure logits and targets have compatible shapes
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(f"Shape Mismatch: Logits {logits.shape}, Targets {targets.shape}")

        # Ensure targets are long for loss computation
        targets = targets.long()

        return logits, targets, loss, answer_types

    def training_step(self, batch, batch_idx):
        logits, targets, loss, answer_types = self.step(batch)

        # Compute accuracy separately for open-ended and close-ended questions
        for i, answer_type in enumerate(answer_types):
            if answer_type == "OPEN":
                self.train_vqa_open.update(logits[i].unsqueeze(0), targets[i].unsqueeze(0), types=1)
            else:
                self.train_vqa_close.update(logits[i].unsqueeze(0), targets[i].unsqueeze(0), types=0)

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
        log_file = "training_log_vqarad.csv"
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
        logits, targets, loss, answer_types = self.step(batch)

        for i, answer_type in enumerate(answer_types):
            if answer_type == "OPEN":
                self.val_vqa_open.update(logits[i].unsqueeze(0), targets[i].unsqueeze(0), types=1)
            else:
                self.val_vqa_close.update(logits[i].unsqueeze(0), targets[i].unsqueeze(0), types=0)

        # Generate predictions
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            max_new_tokens=8,
            num_beams=1
        )

        # Decode full responses
        full_responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        references = self.processor.batch_decode(batch["labels"], skip_special_tokens=True)
        # Convert logits to text
        predicted_answer = self.processor.decode(torch.argmax(logits[0], dim=-1), skip_special_tokens=True)

        stripped_responses = []
        for i, response in enumerate(full_responses):
            stripped_responses.append(response.split("ASSISTANT: ")[1])


        # Compute metrics based on stripped output
        try:
            rouge_scores = self.rouge.compute(predictions=stripped_responses, references=references)
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
        log_file = "validation_log_vqarad.csv"
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
        logits, targets, loss, answer_types = self.step(batch)

        for i, answer_type in enumerate(answer_types):
            if answer_type == "OPEN":
                self.test_vqa_open.update(logits[i].unsqueeze(0), targets[i].unsqueeze(0), types=1)
            else:
                self.test_vqa_close.update(logits[i].unsqueeze(0), targets[i].unsqueeze(0), types=0)

        # Generate predictions
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            max_new_tokens=8,
            num_beams=4
        )

        # Decode full responses
        full_responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        references = self.processor.batch_decode(batch["labels"], skip_special_tokens=True)
        # Convert logits to text
        predicted_answer = self.processor.decode(torch.argmax(logits[0], dim=-1), skip_special_tokens=True)

        stripped_responses = []
        for i, response in enumerate(full_responses):
            stripped_responses.append(response.split("ASSISTANT: ")[1])


        # Compute metrics based on stripped output
        try:
            rouge_scores = self.rouge.compute(predictions=stripped_responses, references=references)
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
        log_file = "test_log_vqarad_test.csv"
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
    
    wandb.init(project='llava-vqarad-23')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    
    # Model and training configuration
    MODEL_NAME = "llava-finetuned-vqarad-10"
    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    NUM_WORKERS = 8
    MAX_LENGTH = 2048
    LEARNING_RATE = 2e-6
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 50
    MAX_EPOCHS = 5
    GRADIENT_ACCUMULATION_STEPS = 8  # Simulates a larger batch size without memory spikes
    

    # Initialize processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token  # Set padding token
    
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
        # fast_dev_run=2,
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
    # trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    # Get final accuracy
    # train_open_score = model.train_vqa_open.compute().item()
    # train_close_score = model.train_vqa_close.compute().item()
    test_open_score = model.test_vqa_open.compute().item()
    test_close_score = model.test_vqa_close.compute().item()

    # print(f"Train Open-Ended Accuracy: {train_open_score}")
    # print(f"Train Close-Ended Accuracy: {train_close_score}")
    print(f"Test Open-Ended Accuracy: {test_open_score}")
    print(f"Test Close-Ended Accuracy: {test_close_score}")

    
    
    # Save final model
    # model.model.save_pretrained("llava-finetuned-vqarad")
    # processor.save_pretrained("llava-finetuned-vqarad")

    # test_model_performance(model, datamodule, processor, num_samples=20)


if __name__ == "__main__":
    main()

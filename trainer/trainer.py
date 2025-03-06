import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoProcessor
from config.training_config import TrainingConfig

class LlavaTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.processor = self._setup_processor()
        self.trainer = self._setup_trainer()
        
    def _setup_processor(self):
        processor = AutoProcessor.from_pretrained(self.config.model_name)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        return processor
    
    def _setup_trainer(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.checkpoint_dir,
            filename="llava-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        )
        
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        )
        
        return pl.Trainer(
            fast_dev_run=2,
            max_epochs=self.config.max_epochs,
            accelerator="cpu",
            precision="bf16-mixed",
            gradient_clip_val=1.0,
            accumulate_grad_batches=self.config.gradient_accumulation_steps,
            callbacks=[early_stopping],
            logger=WandbLogger()
        )
    
    def train(self, model, datamodule):
        self.trainer.fit(model, datamodule=datamodule)
        self.trainer.test(model, datamodule=datamodule)
        
        # Print final metrics
        self._print_metrics(model)
        
        # Save final model
        self._save_model(model)
    
    def _print_metrics(self, model):
        train_open_score = model.train_vqa_open.compute().item()
        train_close_score = model.train_vqa_close.compute().item()
        test_open_score = model.test_vqa_open.compute().item()
        test_close_score = model.test_vqa_close.compute().item()
        
        print(f"Train Open-Ended Accuracy: {train_open_score}")
        print(f"Train Close-Ended Accuracy: {train_close_score}")
        print(f"Test Open-Ended Accuracy: {test_open_score}")
        print(f"Test Close-Ended Accuracy: {test_close_score}")
    
    def _save_model(self, model):
        model.model.save_pretrained(self.config.model_save_dir)
        self.processor.save_pretrained(self.config.model_save_dir) 
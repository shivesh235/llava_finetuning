import wandb
from config.training_config import TrainingConfig
from trainer.trainer import LlavaTrainer
from datamodules.slake_dm import SlakeDataModule
from models.llava_transfer_learning import LlavaFineTuner

def main():
    # Initialize wandb
    wandb.init(project='llava-lora-slack-27')
    
    # Load configuration
    config = TrainingConfig()
    
    # Initialize trainer
    trainer = LlavaTrainer(config)
    
    # Initialize datamodule
    datamodule = SlakeDataModule(
        processor=trainer.processor,
        base_img_dir=config.base_img_dir,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        max_length=config.max_length
    )
    
    # Initialize model
    model = LlavaFineTuner(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        exp_name=config.exp_name
    )
    
    # Train the model
    trainer.train(model, datamodule)

if __name__ == "__main__":
    main() 
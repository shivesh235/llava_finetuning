from dataclasses import dataclass
from typing import List

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "slake/llava-finetuned-slake-7"
    max_length: int = 2048
    
    # Training parameters
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 24
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_epochs: int = 2
    gradient_accumulation_steps: int = 8
    
    # Data paths
    base_img_dir: str = 'slake/slack_imgs'
    
    # LoRA configuration
    target_modules: List[str] = [
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.out_proj', 'mlp.fc1', 'mlp.fc2'
    ]
    
    # Output paths
    exp_name: str = 'slake/llava_slake_27feb_e9'
    checkpoint_dir: str = "checkpoints"
    model_save_dir: str = "slake/llava-slake-27feb-e9" 
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import AutoProcessor, LlavaProcessor


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
        prompt = f"<s>[INST] <image> \n{item['question']}[/INST]"

        # Tokenize prompt and process image
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

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
            "labels": target
        }


# Load processor for VQA-RAD
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Load the VQA-RAD dataset
dataset = load_dataset("flaviagiammarino/vqa-rad")


# Create dataset instances
train_dataset = VQARADDataset(dataset["train"], processor)
test_dataset = VQARADDataset(dataset["test"], processor)

# Create DataLoader instances
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# Test the DataLoader
for batch in train_dataloader:
    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)
    print(batch["pixel_values"].shape)
    print(batch["labels"].shape)
    break  # Remove this line to iterate through the entire DataLoader

import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

def load_quantized_model(
    model_name="llava-hf/llava-1.5-7b-hf",
    load_in_4bit=True,
    use_flash_attention=False
):
    """
    Load Llava model in 4-bit quantization.
    
    Args:
        model_name (str): Name or path of the model
        load_in_4bit (bool): Whether to load in 4bit quantization
        use_flash_attention (bool): Whether to use flash attention
    
    Returns:
        tuple: (model, processor)
    """
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"  # normalized float 4
    )

    # Model loading configuration
    model_config = {
        "device_map": "auto",  # Automatically distribute across available GPUs
        "torch_dtype": torch.float16,
        "quantization_config": quantization_config
    }

    if use_flash_attention:
        model_config["use_flash_attention_2"] = True

    try:
        # Load model with 4-bit quantization
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            **model_config
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_name)
        
        return model, processor

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # First install required packages
    # !pip install bitsandbytes transformers accelerate
    
    # Load model and processor
    model, processor = load_quantized_model()
    
    # Print model memory usage
    print("\nModel loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Optional: print detailed memory info
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
# LLaVA Fine-tuning Framework

A modular and extensible framework for fine-tuning LLaVA (Large Language and Vision Assistant) models on medical VQA datasets.

## Overview

This project provides a structured framework for fine-tuning LLaVA models on medical visual question answering (VQA) datasets. It supports multiple medical VQA datasets including SLAKE, VQARAD, and PathVQA. The framework is built with modularity and extensibility in mind, making it easy to add new datasets or modify training configurations.

## Features

- Modular architecture with clear separation of concerns
- Support for multiple medical VQA datasets
- Configurable training parameters
- Built-in metrics for both open-ended and close-ended VQA
- Integration with Weights & Biases for experiment tracking
- LoRA fine-tuning support for efficient training
- PyTorch Lightning integration for scalable training

## Project Structure

```
llava_finetuning/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── training_config.py  # Training configuration dataclass
├── datamodules/           # Dataset-specific data modules
│   ├── slake_dm.py       # SLAKE dataset module
│   ├── vqarad_dm.py      # VQARAD dataset module
│   └── pathvqa_dm.py     # PathVQA dataset module
├── models/               # Model definitions
│   └── llava_transfer_learning.py
├── trainer/             # Training logic
│   ├── __init__.py
│   └── trainer.py       # Main trainer class
├── utils/              # Utility functions
│   ├── __init__.py
│   └── metrics.py      # Custom metrics
├── train.py            # Main training script
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shivesh235/llava_finetuning.git
cd llava_finetuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure your training parameters in `config/training_config.py`

2. Run training:
```bash
python train.py
```

## Configuration

The training configuration can be modified in `config/training_config.py`. Key parameters include:

- Model configuration (model name, max sequence length)
- Training parameters (batch size, learning rate, epochs)
- LoRA configuration (target modules)
- Data paths and output directories

## Metrics

The framework includes custom metrics for both open-ended and close-ended VQA:

- `OpenEndedVQAMetric`: For evaluating open-ended question answering
- `CloseEndedVQAMetric`: For evaluating close-ended question answering

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{llava_finetuning2024,
  author = {Shivesh},
  title = {LLaVA Fine-tuning Framework},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/shivesh235/llava_finetuning}
}
```
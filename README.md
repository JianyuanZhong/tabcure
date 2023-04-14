# TabCure

<p align="center">
    <a href="https://github.com/JianyuanZhong/tabcure/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/JianyuanZhong/tabcure/Test%20Suite/main?label=main%20checks></a>
    <a href="https://JianyuanZhong.github.io/tabcure"><img alt="Docs" src=https://img.shields.io/github/deployments/JianyuanZhong/tabcure/github-pages?label=docs></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.9-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

A python toolkit for synthetic tabular data with LLMs.


## Installation

```bash
pip install git+ssh://git@github.com/JianyuanZhong/tabcure.git
```


## Development installation

Setup the development environment:

```bash
git clone git@github.com:JianyuanZhong/tabcure.git
cd tabcure
conda env create -f env.yaml
conda activate tabcure
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```

## Quick Start

```python
import pandas as pd
from tabcure import TabCure
from transformers import GenerationConfig, TrainingArguments
from peft import LoraConfig, PeftConfig, PeftModel

real_data = pd.read_csv("path to your own data.")

# train, PEFT and generation configs
trainer_config = TrainingArguments(
    num_train_epochs=20,
    logging_steps=10,
    per_device_train_batch_size=32,
    learning_rate=0.0001,
)
peft_config = LoraConfig(r=64)
gen_config = GenerationConfig(temperature=0.8)

# train
tabcure = TabCure(
    "decapoda-research/llama-7b-hf",
    trainer_config=trainer_config,
    peft_config=peft_config,
)
tabcure.lora_fit(real_data)

# generate
samples = tabcure.sample(n_samples, config=gen_config, k=10)
samples.to_csv("path to your synthetic data.")
```

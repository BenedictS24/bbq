from datasets import load_dataset
from huggingface_hub import list_datasets
print([dataset.id for dataset in list_datasets()])

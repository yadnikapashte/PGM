from datasets import load_dataset
from config import Config

def check_dataset():
    dataset_name = Config.DATASET_NAME
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split='train', streaming=True)
    sample = next(iter(ds))
    print("Dataset structure (sample):")
    print(sample.keys())
    print(sample)

if __name__ == "__main__":
    check_dataset()

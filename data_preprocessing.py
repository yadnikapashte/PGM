"""
Data Preprocessing Module
Handles dataset loading, cleaning, and splitting
"""

from datasets import load_dataset
from config import Config
import os


class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load_dataset(self):
        """Load the Swahili dataset from Hugging Face"""
        print("\n" + "=" * 60)
        print("Loading Dataset")
        print("=" * 60)

        # Load from Hugging Face dataset
        print(f"Loading dataset from Hugging Face: {self.config.DATASET_NAME}")
        self.dataset = load_dataset(self.config.DATASET_NAME)

        # 1. Rename columns to lowercase if they are capitalized
        for split in list(self.dataset.keys()):
            cols = self.dataset[split].column_names
            if 'Swahili' in cols:
                print(f"Renaming 'Swahili' to 'swahili' in {split} split")
                self.dataset[split] = self.dataset[split].rename_column('Swahili', 'swahili')
            if 'English' in cols:
                print(f"Renaming 'English' to 'english' in {split} split")
                self.dataset[split] = self.dataset[split].rename_column('English', 'english')

        # 2. Automatically split the dataset if only 'train' exists
        if len(self.dataset.keys()) == 1 and 'train' in self.dataset:
            print("\nOnly 'train' split found. Splitting into train, validation, and test...")
            
            # Split into train and temporary 'test+val'
            train_test_valid = self.dataset['train'].train_test_split(
                test_size=self.config.VAL_RATIO + self.config.TEST_RATIO,
                seed=self.config.RANDOM_SEED
            )
            
            # Split 'test+val' into separate validation and test sets
            test_valid = train_test_valid['test'].train_test_split(
                test_size=self.config.TEST_RATIO / (self.config.VAL_RATIO + self.config.TEST_RATIO),
                seed=self.config.RANDOM_SEED
            )
            
            from datasets import DatasetDict
            self.dataset = DatasetDict({
                'train': train_test_valid['train'],
                'validation': test_valid['train'],
                'test': test_valid['test']
            })
            print(f"Split completed: train={len(self.dataset['train'])}, validation={len(self.dataset['validation'])}, test={len(self.dataset['test'])}")

        return self.dataset
    
    def _load_local_dataset(self):
        """Load dataset from unzipped local Swahili text files"""
        from datasets import Dataset

        # Paths to the unzipped files
        base_path = "Swahili data/Swahili data"
        train_file = os.path.join(base_path, "train.txt")
        valid_file = os.path.join(base_path, "valid.txt")
        test_file = os.path.join(base_path, "test.txt")

        # Read Swahili sentences from files
        def read_sentences(file_path):
            sentences = []
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            sentences.append(line)
            return sentences

        train_sentences = read_sentences(train_file)
        valid_sentences = read_sentences(valid_file)
        test_sentences = read_sentences(test_file)

        # Create dataset with Swahili sentences (English empty for now)
        data = {
            'swahili': train_sentences + valid_sentences + test_sentences,
            'english': [''] * (len(train_sentences) + len(valid_sentences) + len(test_sentences))
        }

        dataset = Dataset.from_dict(data)

        # Split into train, validation, test
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))

        dataset_dict = {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }

        print(f"Loaded local dataset with {len(train_sentences)} train, {len(valid_sentences)} valid, {len(test_sentences)} test examples")
        return dataset_dict
    
    def clean_dataset(self, examples):
        """
        Clean the dataset by filtering out:
        - Empty sentences
        - Very long sentences (>500 characters)
        - Sentences with null values
        - Sentences that are too short
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            List of boolean values indicating valid entries
        """
        swahili = examples['swahili']
        english = examples['english']
        
        valid = []
        for sw, en in zip(swahili, english):
            # Check if both sentences are valid strings
            if not (sw and en and isinstance(sw, str) and isinstance(en, str)):
                valid.append(False)
                continue
            
            sw_stripped = sw.strip()
            en_stripped = en.strip()
            
            # Check length constraints
            if (len(sw_stripped) < self.config.MIN_SENTENCE_LENGTH or 
                len(en_stripped) < self.config.MIN_SENTENCE_LENGTH):
                valid.append(False)
                continue
            
            if (len(sw) > self.config.MAX_SENTENCE_LENGTH or 
                len(en) > self.config.MAX_SENTENCE_LENGTH):
                valid.append(False)
                continue
            
            valid.append(True)
        
        return valid
    
    def apply_cleaning(self):
        """Apply cleaning filter to the dataset"""
        print("\n" + "=" * 60)
        print("Cleaning Dataset")
        print("=" * 60)

        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        # Clean each split separately
        for split in ['train', 'validation', 'test']:
            original_size = len(self.dataset[split])
            self.dataset[split] = self.dataset[split].filter(self.clean_dataset, batched=True)
            cleaned_size = len(self.dataset[split])
            print(f"{split.capitalize()} - Original size: {original_size}, Cleaned size: {cleaned_size}, Removed: {original_size - cleaned_size} examples")

        return self.dataset
    
    def save_preprocessed_data(self):
        """Save the preprocessed dataset to disk"""
        print("\n" + "=" * 60)
        print("Saving Preprocessed Data")
        print("=" * 60)
        
        if self.dataset is None:
            raise ValueError("Dataset not processed. Call process_all() first.")
            
        save_path = self.config.PREPROCESSED_DATA_DIR
        self.dataset.save_to_disk(save_path)
        print(f"Preprocessed dataset saved to: {save_path}")
        
        # Save as CSV for easy inspection if size is manageable
        for split in ['train', 'validation', 'test']:
            csv_path = os.path.join(save_path, f"{split}.csv")
            try:
                # Save first 1000 examples as CSV for quick preview
                self.dataset[split].select(range(min(1000, len(self.dataset[split])))).to_csv(csv_path)
                print(f"  - {split} preview saved to {csv_path}")
            except Exception as e:
                print(f"  - Could not save {split} CSV preview: {e}")
                
        return save_path

    def load_preprocessed_dataset(self):
        """Load the preprocessed dataset from disk if it exists"""
        from datasets import load_from_disk
        save_path = self.config.PREPROCESSED_DATA_DIR
        
        if os.path.exists(save_path) and os.path.exists(os.path.join(save_path, "dataset_dict.json")):
            print(f"\nLoading preprocessed dataset from: {save_path}")
            self.dataset = load_from_disk(save_path)
            self.train_dataset = self.dataset['train']
            self.val_dataset = self.dataset['validation']
            self.test_dataset = self.dataset['test']
            print(f"Successfully loaded: train={len(self.train_dataset)}, validation={len(self.val_dataset)}, test={len(self.test_dataset)}")
            return True
        return False
    
    def split_dataset(self):
        """Assign dataset splits (already split in load_dataset)"""
        print("\n" + "=" * 60)
        print("Splitting Dataset")
        print("=" * 60)

        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        # Dataset is already split in load_dataset
        self.train_dataset = self.dataset['train']
        self.val_dataset = self.dataset['validation']
        self.test_dataset = self.dataset['test']

        print(f"Train set: {len(self.train_dataset)} examples")
        print(f"Validation set: {len(self.val_dataset)} examples")
        print(f"Test set: {len(self.test_dataset)} examples")

        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_sample_data(self, num_samples=5):
        """Get sample data for inspection"""
        if self.train_dataset is None:
            raise ValueError("Dataset not split. Call split_dataset() first.")
        
        print("\n" + "=" * 60)
        print(f"Sample Data ({num_samples} examples)")
        print("=" * 60)
        
        for i in range(min(num_samples, len(self.train_dataset))):
            example = self.train_dataset[i]
            print(f"\nExample {i+1}:")
            print(f"  Swahili: {example['swahili']}")
            print(f"  English: {example['english']}")
    
    def process_all(self, force_reprocess=False):
        """Run the complete preprocessing pipeline or load from disk"""
        if not force_reprocess and self.load_preprocessed_dataset():
            print("Skipping preprocessing pipeline as data exists on disk.")
            return self.train_dataset, self.val_dataset, self.test_dataset
            
        self.load_dataset()
        self.apply_cleaning()
        self.split_dataset()
        self.save_preprocessed_data()
        
        return self.train_dataset, self.val_dataset, self.test_dataset


if __name__ == "__main__":
    # Test the data preprocessing pipeline
    preprocessor = DataPreprocessor()
    train, val, test = preprocessor.process_all()
    preprocessor.get_sample_data()
    
    print("\nData preprocessing complete!")

"""
Data Preprocessing Module
Handles dataset loading, cleaning, and splitting
"""

from datasets import load_dataset
from config import Config


class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load_dataset(self):
        """Load the Swahili-English parallel dataset from Hugging Face"""
        print("\n" + "=" * 60)
        print("Loading Dataset")
        print("=" * 60)
        
        self.dataset = load_dataset(self.config.DATASET_NAME)
        print(f"Original dataset size: {len(self.dataset['train'])}")
        
        return self.dataset
    
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
        
        original_size = len(self.dataset['train'])
        self.dataset = self.dataset.filter(self.clean_dataset, batched=True)
        cleaned_size = len(self.dataset['train'])
        
        print(f"Original size: {original_size}")
        print(f"Cleaned size: {cleaned_size}")
        print(f"Removed: {original_size - cleaned_size} examples")
        
        return self.dataset
    
    def split_dataset(self):
        """Split dataset into train, validation, and test sets"""
        print("\n" + "=" * 60)
        print("Splitting Dataset")
        print("=" * 60)
        
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # First split: train vs (val + test)
        train_test_split = self.dataset['train'].train_test_split(
            test_size=self.config.VAL_RATIO + self.config.TEST_RATIO,
            seed=self.config.RANDOM_SEED
        )
        
        # Second split: val vs test
        val_test_split = train_test_split['test'].train_test_split(
            test_size=self.config.TEST_RATIO / (self.config.VAL_RATIO + self.config.TEST_RATIO),
            seed=self.config.RANDOM_SEED
        )
        
        self.train_dataset = train_test_split['train']
        self.val_dataset = val_test_split['train']
        self.test_dataset = val_test_split['test']
        
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
    
    def process_all(self):
        """Run the complete preprocessing pipeline"""
        self.load_dataset()
        self.apply_cleaning()
        self.split_dataset()
        
        return self.train_dataset, self.val_dataset, self.test_dataset


if __name__ == "__main__":
    # Test the data preprocessing pipeline
    preprocessor = DataPreprocessor()
    train, val, test = preprocessor.process_all()
    preprocessor.get_sample_data()
    
    print("\n✓ Data preprocessing complete!")

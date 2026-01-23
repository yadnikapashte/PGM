"""
Tokenization Module
Handles tokenization of Swahili-English parallel data for BLOOM model
"""

from transformers import AutoTokenizer
from config import Config


class TranslationTokenizer:
    """Tokenizer for translation tasks using BLOOM"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        self.tokenizer = None
    
    def load_tokenizer(self):
        """Load BLOOM tokenizer"""
        print("\n" + "=" * 60)
        print("Loading Tokenizer")
        print("=" * 60)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Added padding token (using EOS token)")
        
        print(f"Tokenizer loaded: {self.config.MODEL_NAME}")
        print(f"Vocabulary size: {len(self.tokenizer)}")
        print(f"Pad token: {self.tokenizer.pad_token}")
        print(f"EOS token: {self.tokenizer.eos_token}")
        
        return self.tokenizer
    
    @staticmethod
    def create_translation_prompt(swahili_text, english_text=None):
        """
        Create a prompt for translation task.
        
        Format: Translate Swahili to English: [swahili] -> [english]
        
        Args:
            swahili_text: Source Swahili sentence
            english_text: Target English translation (optional, for training)
            
        Returns:
            Formatted prompt string
        """
        if english_text:
            return f"Translate Swahili to English: {swahili_text} -> {english_text}"
        else:
            return f"Translate Swahili to English: {swahili_text} ->"
    
    def tokenize_dataset(self, examples):
        """
        Tokenize the input-output pairs for causal language modeling.
        
        The model learns to predict the English translation given the Swahili input.
        
        Args:
            examples: Batch of examples containing 'swahili' and 'english' fields
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        # Create prompts for each example
        prompts = [
            self.create_translation_prompt(sw, en)
            for sw, en in zip(examples['swahili'], examples['english'])
        ]
        
        # Tokenize with truncation and padding
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding='max_length',
            max_length=self.config.MAX_LENGTH,
            return_tensors=None
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def tokenize_all_splits(self, train_dataset, val_dataset, test_dataset):
        """
        Tokenize all dataset splits.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            
        Returns:
            Tuple of (tokenized_train, tokenized_val, tokenized_test)
        """
        print("\n" + "=" * 60)
        print("Tokenizing Datasets")
        print("=" * 60)
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        print("Tokenizing training set...")
        tokenized_train = train_dataset.map(
            self.tokenize_dataset,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        print("Tokenizing validation set...")
        tokenized_val = val_dataset.map(
            self.tokenize_dataset,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        print("Tokenizing test set...")
        tokenized_test = test_dataset.map(
            self.tokenize_dataset,
            batched=True,
            remove_columns=test_dataset.column_names
        )
        
        print("✓ Tokenization complete!")
        print(f"  Train samples: {len(tokenized_train)}")
        print(f"  Val samples: {len(tokenized_val)}")
        print(f"  Test samples: {len(tokenized_test)}")
        
        return tokenized_train, tokenized_val, tokenized_test
    
    def decode_example(self, tokenized_example):
        """Decode a tokenized example back to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        input_ids = tokenized_example['input_ids']
        decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        return decoded_text
    
    def show_tokenization_example(self, dataset, num_examples=2):
        """Show examples of tokenized data"""
        print("\n" + "=" * 60)
        print("Tokenization Examples")
        print("=" * 60)
        
        for i in range(min(num_examples, len(dataset))):
            example = dataset[i]
            print(f"\nExample {i+1}:")
            print(f"  Swahili: {example['swahili']}")
            print(f"  English: {example['english']}")
            
            prompt = self.create_translation_prompt(
                example['swahili'], 
                example['english']
            )
            print(f"  Prompt: {prompt}")
            
            tokenized = self.tokenizer(prompt, truncation=True, max_length=self.config.MAX_LENGTH)
            print(f"  Token count: {len(tokenized['input_ids'])}")


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    train, val, test = preprocessor.process_all()
    
    # Initialize tokenizer
    tok = TranslationTokenizer()
    tok.load_tokenizer()
    
    # Show examples
    tok.show_tokenization_example(train, num_examples=3)
    
    # Tokenize all splits
    tok_train, tok_val, tok_test = tok.tokenize_all_splits(train, val, test)
    
    print("\n✓ Tokenization complete!")

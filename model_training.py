"""
Model Training Module
Handles BLOOM model loading and fine-tuning for translation
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from config import Config
import os


class TranslationModelTrainer:
    """Handles model loading and training operations"""
    
    def __init__(self, tokenizer, config=None):
        self.config = config or Config()
        self.tokenizer = tokenizer
        self.model = None
        self.trainer = None
        
    def load_model(self):
        """Load BLOOM model for causal language modeling"""
        print("\n" + "=" * 60)
        print("Loading BLOOM Model")
        print("=" * 60)
        
        # Load model with appropriate precision
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=torch.float16 if self.config.USE_FP16 else torch.float32
        )
        
        # Update model config to match tokenizer
        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"Model loaded: {self.config.MODEL_NAME}")
        print(f"Model parameters: {self.model.num_parameters():,}")
        print(f"Model dtype: {self.model.dtype}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        return self.model
    
    def setup_training(self, train_dataset, val_dataset):
        """
        Setup training configuration and trainer.
        
        Args:
            train_dataset: Tokenized training dataset
            val_dataset: Tokenized validation dataset
        """
        print("\n" + "=" * 60)
        print("Setting Up Training")
        print("=" * 60)
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Create output directories
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.SAVE_DIR,
            overwrite_output_dir=True,
            num_train_epochs=self.config.EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            warmup_steps=self.config.WARMUP_STEPS,
            learning_rate=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            fp16=self.config.USE_FP16,
            logging_dir=self.config.LOG_DIR,
            logging_steps=self.config.LOGGING_STEPS,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=self.config.SAVE_TOTAL_LIMIT,
            report_to="none",
            push_to_hub=False,
            gradient_accumulation_steps=1,
            dataloader_num_workers=0,
        )
        
        # Data collator for causal language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        print("Training configuration:")
        print(f"  Output directory: {self.config.SAVE_DIR}")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  Learning rate: {self.config.LEARNING_RATE}")
        print(f"  Epochs: {self.config.EPOCHS}")
        print(f"  Warmup steps: {self.config.WARMUP_STEPS}")
        print(f"  Mixed precision (FP16): {self.config.USE_FP16}")
        print(f"  Device: {self.config.DEVICE}")
        
        return self.trainer
    
    def train(self):
        """Execute the training process"""
        print("\n" + "=" * 60)
        print("Starting Fine-tuning")
        print("=" * 60)
        
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_training() first.")
        
        # Train the model
        train_result = self.trainer.train()
        
        # Get training metrics
        metrics = train_result.metrics
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Training runtime: {metrics.get('train_runtime', 'N/A'):.2f} seconds")
        print(f"Samples per second: {metrics.get('train_samples_per_second', 'N/A'):.2f}")
        
        return train_result
    
    def save_model(self):
        """Save the fine-tuned model and tokenizer"""
        print("\n" + "=" * 60)
        print("Saving Model and Tokenizer")
        print("=" * 60)
        
        if self.trainer is None:
            raise ValueError("Trainer not setup. Cannot save model.")
        
        # Save model
        self.trainer.save_model(self.config.SAVE_DIR)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.config.SAVE_DIR)
        
        print(f"✓ Model saved to: {self.config.SAVE_DIR}")
        print(f"✓ Tokenizer saved to: {self.config.SAVE_DIR}")
        
    def evaluate(self, eval_dataset=None):
        """
        Evaluate the model on validation set.
        
        Args:
            eval_dataset: Dataset to evaluate on (uses validation set if None)
        """
        print("\n" + "=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_training() first.")
        
        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        print(f"Evaluation loss: {metrics.get('eval_loss', 'N/A'):.4f}")
        print(f"Perplexity: {torch.exp(torch.tensor(metrics.get('eval_loss', 0))):.2f}")
        
        return metrics


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from tokenization import TranslationTokenizer
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    train, val, test = preprocessor.process_all()
    
    # Tokenize
    print("\nTokenizing data...")
    tokenizer_wrapper = TranslationTokenizer()
    tokenizer = tokenizer_wrapper.load_tokenizer()
    tok_train, tok_val, tok_test = tokenizer_wrapper.tokenize_all_splits(train, val, test)
    
    # Train
    print("\nInitializing trainer...")
    trainer = TranslationModelTrainer(tokenizer)
    trainer.load_model()
    trainer.setup_training(tok_train, tok_val)
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    
    print("\n✓ Training pipeline complete!")

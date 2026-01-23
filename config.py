"""
Configuration file for Swahili-English Translation System
Contains all hyperparameters and settings for training and evaluation
"""

import torch


class Config:
    """Configuration parameters for the translation system"""
    
    # Model settings
    MODEL_NAME = "bigscience/bloom-560m"
    DATASET_NAME = "michsethowusu/english-swahili_sentence-pairs"
    
    # Training hyperparameters
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Data split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    RANDOM_SEED = 42
    
    # Directory settings
    SAVE_DIR = "./swahili_translation_model"
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"
    
    # Evaluation settings
    NUM_EVAL_SAMPLES = 200
    MAX_NEW_TOKENS = 50
    NUM_BEAMS = 4
    TEMPERATURE = 0.7
    
    # Data cleaning thresholds
    MAX_SENTENCE_LENGTH = 500
    MIN_SENTENCE_LENGTH = 1
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_FP16 = torch.cuda.is_available()
    
    # Logging
    LOGGING_STEPS = 100
    SAVE_TOTAL_LIMIT = 2
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("=" * 60)
        print("CONFIGURATION PARAMETERS")
        print("=" * 60)
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Dataset: {cls.DATASET_NAME}")
        print(f"Device: {cls.DEVICE}")
        print(f"Mixed Precision (FP16): {cls.USE_FP16}")
        print(f"\nTraining Parameters:")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Max Length: {cls.MAX_LENGTH}")
        print(f"  Warmup Steps: {cls.WARMUP_STEPS}")
        print(f"\nData Split:")
        print(f"  Train: {cls.TRAIN_RATIO * 100}%")
        print(f"  Validation: {cls.VAL_RATIO * 100}%")
        print(f"  Test: {cls.TEST_RATIO * 100}%")
        print("=" * 60)

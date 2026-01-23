"""
Main Pipeline Script
Orchestrates the complete Swahili-English translation system
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_preprocessing import DataPreprocessor
from tokenization import TranslationTokenizer
from model_training import TranslationModelTrainer
from evaluation import TranslationEvaluator
from inference import TranslationInference


def run_complete_pipeline(skip_training=False, skip_evaluation=False):
    """
    Run the complete translation pipeline.
    
    Args:
        skip_training: Skip training if model already exists
        skip_evaluation: Skip evaluation step
    """
    print("\n" + "="*60)
    print("SWAHILI-ENGLISH TRANSLATION SYSTEM")
    print("Using BLOOM-560M")
    print("="*60)
    
    # Print configuration
    Config.print_config()
    
    # Step 1: Data Preprocessing
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, test_dataset = preprocessor.process_all()
    preprocessor.get_sample_data(num_samples=3)
    
    # Step 2: Tokenization
    print("\n" + "="*60)
    print("STEP 2: TOKENIZATION")
    print("="*60)
    
    tokenizer_wrapper = TranslationTokenizer()
    tokenizer = tokenizer_wrapper.load_tokenizer()
    tokenizer_wrapper.show_tokenization_example(train_dataset, num_examples=2)
    
    tokenized_train, tokenized_val, tokenized_test = tokenizer_wrapper.tokenize_all_splits(
        train_dataset, val_dataset, test_dataset
    )
    
    # Step 3: Model Training
    if not skip_training:
        print("\n" + "="*60)
        print("STEP 3: MODEL TRAINING")
        print("="*60)
        
        trainer = TranslationModelTrainer(tokenizer)
        trainer.load_model()
        trainer.setup_training(tokenized_train, tokenized_val)
        trainer.train()
        trainer.evaluate()
        trainer.save_model()
    else:
        print("\n" + "="*60)
        print("STEP 3: SKIPPING TRAINING (using existing model)")
        print("="*60)
    
    # Step 4: Evaluation
    if not skip_evaluation:
        print("\n" + "="*60)
        print("STEP 4: EVALUATION")
        print("="*60)
        
        evaluator = TranslationEvaluator()
        evaluator.load_model()
        evaluator.show_sample_translations(test_dataset, num_examples=5)
        results = evaluator.evaluate_bleu(test_dataset, show_progress=True)
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"BLEU Score: {results['bleu_score']:.2f}")
        print(f"Target: > 25.0")
        print(f"Status: {'✓ ACHIEVED' if results['bleu_score'] > 25.0 else '✗ NOT ACHIEVED'}")
        print(f"Samples evaluated: {results['num_samples']}")
        print("="*60)
    
    print("\n✓ Pipeline complete!")
    print(f"\nModel saved to: {Config.SAVE_DIR}")
    print("You can now use the model for translation with inference.py")


def run_inference_only():
    """Run inference on custom Swahili sentences"""
    print("\n" + "="*60)
    print("INFERENCE MODE")
    print("="*60)
    
    inference = TranslationInference()
    inference.load_model_and_tokenizer()
    inference.interactive_translate()


def run_evaluation_only():
    """Run evaluation on existing model"""
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    # Load test data
    preprocessor = DataPreprocessor()
    _, _, test_dataset = preprocessor.process_all()
    
    # Evaluate
    evaluator = TranslationEvaluator()
    evaluator.load_model()
    evaluator.show_sample_translations(test_dataset, num_examples=5)
    results = evaluator.evaluate_bleu(test_dataset, show_progress=True)
    
    return results


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Swahili-English Translation System using BLOOM"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'inference', 'full'],
        default='full',
        help='Mode to run: train, evaluate, inference, or full pipeline'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training step (use existing model)'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip evaluation step'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_complete_pipeline(
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation
        )
    elif args.mode == 'train':
        run_complete_pipeline(
            skip_training=False,
            skip_evaluation=True
        )
    elif args.mode == 'evaluate':
        run_evaluation_only()
    elif args.mode == 'inference':
        run_inference_only()


if __name__ == "__main__":
    # If run without arguments, execute full pipeline
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided - run full pipeline
        run_complete_pipeline()
    else:
        # Arguments provided - use argparse
        main()

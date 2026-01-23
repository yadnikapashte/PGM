"""
Evaluation Module
Handles BLEU score calculation and model evaluation
"""

import numpy as np
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from config import Config
from inference import TranslationInference


class TranslationEvaluator:
    """Handles evaluation of translation quality"""
    
    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.model_path = model_path or self.config.SAVE_DIR
        self.inference = None
        self.bleu_metric = BLEU()
    
    def load_model(self):
        """Load model for evaluation"""
        self.inference = TranslationInference(model_path=self.model_path, config=self.config)
        self.inference.load_model_and_tokenizer()
        
        return self.inference
    
    def evaluate_bleu(self, test_dataset, num_samples=None, show_progress=True):
        """
        Evaluate translation quality using BLEU score.
        
        Args:
            test_dataset: Dataset containing Swahili-English pairs
            num_samples: Number of samples to evaluate (None = all)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary containing BLEU score and additional metrics
        """
        print("\n" + "=" * 60)
        print("Evaluating with BLEU Score")
        print("=" * 60)
        
        if self.inference is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Determine number of samples to evaluate
        total_samples = len(test_dataset)
        num_samples = num_samples or min(self.config.NUM_EVAL_SAMPLES, total_samples)
        
        # Randomly sample if needed
        if num_samples < total_samples:
            indices = np.random.choice(total_samples, num_samples, replace=False)
        else:
            indices = range(total_samples)
        
        print(f"Evaluating on {num_samples} samples...")
        
        predictions = []
        references = []
        
        # Create progress bar if needed
        iterator = tqdm(indices, desc="Translating") if show_progress else indices
        
        # Generate translations
        for idx in iterator:
            example = test_dataset[int(idx)]
            swahili = example['swahili']
            english_ref = example['english']
            
            # Generate translation
            try:
                english_pred = self.inference.translate(swahili)
                predictions.append(english_pred)
                references.append([english_ref])  # BLEU expects list of references
            except Exception as e:
                print(f"\nError translating index {idx}: {e}")
                continue
        
        # Calculate BLEU score
        if len(predictions) == 0:
            raise ValueError("No successful translations generated.")
        
        bleu_score = self.bleu_metric.corpus_score(predictions, references)
        
        # Print results
        print("\n" + "=" * 60)
        print("BLEU Evaluation Results")
        print("=" * 60)
        print(f"Total samples evaluated: {len(predictions)}")
        print(f"BLEU Score: {bleu_score.score:.2f}")
        print(f"  - BLEU-1: {bleu_score.precisions[0]:.2f}")
        print(f"  - BLEU-2: {bleu_score.precisions[1]:.2f}")
        print(f"  - BLEU-3: {bleu_score.precisions[2]:.2f}")
        print(f"  - BLEU-4: {bleu_score.precisions[3]:.2f}")
        print(f"Brevity Penalty: {bleu_score.bp:.3f}")
        print(f"\nTarget BLEU Score: > 25.0")
        
        if bleu_score.score > 25.0:
            print("Status: ✓ TARGET ACHIEVED!")
        else:
            print(f"Status: ✗ Target not achieved (Gap: {25.0 - bleu_score.score:.2f})")
        
        return {
            'bleu_score': bleu_score.score,
            'bleu_1': bleu_score.precisions[0],
            'bleu_2': bleu_score.precisions[1],
            'bleu_3': bleu_score.precisions[2],
            'bleu_4': bleu_score.precisions[3],
            'brevity_penalty': bleu_score.bp,
            'num_samples': len(predictions),
            'predictions': predictions,
            'references': references
        }
    
    def show_sample_translations(self, test_dataset, num_examples=5):
        """
        Display sample translations for qualitative evaluation.
        
        Args:
            test_dataset: Test dataset
            num_examples: Number of examples to show
        """
        print("\n" + "=" * 60)
        print(f"Sample Translations ({num_examples} examples)")
        print("=" * 60)
        
        if self.inference is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        for i in range(min(num_examples, len(test_dataset))):
            example = test_dataset[i]
            swahili = example['swahili']
            english_ref = example['english']
            
            try:
                english_pred = self.inference.translate(swahili)
                
                print(f"\n{'='*60}")
                print(f"Example {i+1}:")
                print(f"{'='*60}")
                print(f"Swahili:    {swahili}")
                print(f"Reference:  {english_ref}")
                print(f"Predicted:  {english_pred}")
                
                # Calculate sentence-level BLEU
                sent_bleu = self.bleu_metric.sentence_score(english_pred, [english_ref])
                print(f"Sent-BLEU:  {sent_bleu.score:.2f}")
                
            except Exception as e:
                print(f"\nError on example {i+1}: {e}")
    
    def compare_with_baseline(self, test_dataset, baseline_translations, num_samples=None):
        """
        Compare model performance with a baseline (e.g., Google Translate).
        
        Args:
            test_dataset: Test dataset
            baseline_translations: List of baseline translations
            num_samples: Number of samples to compare
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "=" * 60)
        print("Comparing with Baseline")
        print("=" * 60)
        
        if self.inference is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        num_samples = num_samples or min(len(test_dataset), len(baseline_translations))
        
        model_predictions = []
        references = []
        
        print(f"Evaluating {num_samples} samples...")
        
        for i in range(num_samples):
            example = test_dataset[i]
            swahili = example['swahili']
            english_ref = example['english']
            
            # Get model prediction
            english_pred = self.inference.translate(swahili)
            
            model_predictions.append(english_pred)
            references.append([english_ref])
        
        # Calculate BLEU for model
        model_bleu = self.bleu_metric.corpus_score(model_predictions, references)
        
        # Calculate BLEU for baseline
        baseline_bleu = self.bleu_metric.corpus_score(
            baseline_translations[:num_samples], 
            references
        )
        
        print(f"\nModel BLEU:    {model_bleu.score:.2f}")
        print(f"Baseline BLEU: {baseline_bleu.score:.2f}")
        print(f"Difference:    {model_bleu.score - baseline_bleu.score:+.2f}")
        
        return {
            'model_bleu': model_bleu.score,
            'baseline_bleu': baseline_bleu.score,
            'difference': model_bleu.score - baseline_bleu.score
        }


def run_full_evaluation(model_path=None, test_dataset=None):
    """
    Run complete evaluation pipeline.
    
    Args:
        model_path: Path to the fine-tuned model
        test_dataset: Test dataset to evaluate on
    """
    evaluator = TranslationEvaluator(model_path=model_path)
    evaluator.load_model()
    
    if test_dataset is None:
        print("No test dataset provided. Loading from preprocessing...")
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        _, _, test_dataset = preprocessor.process_all()
    
    # Show sample translations
    evaluator.show_sample_translations(test_dataset, num_examples=5)
    
    # Calculate BLEU score
    results = evaluator.evaluate_bleu(test_dataset, show_progress=True)
    
    return results


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    # Load test data
    print("Loading test dataset...")
    preprocessor = DataPreprocessor()
    _, _, test_dataset = preprocessor.process_all()
    
    # Run evaluation
    results = run_full_evaluation(test_dataset=test_dataset)
    
    print("\n✓ Evaluation complete!")

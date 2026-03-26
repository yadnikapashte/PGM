"""
Evaluation Module
Handles BLEU score calculation and model evaluation
"""

import numpy as np
import json
import os
from datetime import datetime
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from config import Config
from inference_py import TranslationInference
try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TranslationEvaluator:
    """Handles evaluation of translation quality"""
    
    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.model_path = model_path or self.config.SAVE_DIR
        self.inference = None
        self.bleu_metric = BLEU()
        
        # Create log directory if it doesn't exist
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        
        # Initialize log file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_log_file = os.path.join(self.config.LOG_DIR, f"evaluation_{timestamp}.json")
        self.accuracy_log_file = os.path.join(self.config.LOG_DIR, f"accuracy_metrics_{timestamp}.json")
        self.bleu_log_file = os.path.join(self.config.LOG_DIR, f"bleu_scores_{timestamp}.json")
        self.confusion_log_file = os.path.join(self.config.LOG_DIR, f"confusion_matrix_{timestamp}.json")
    
    def load_model(self):
        """Load model for evaluation"""
        self.inference = TranslationInference(model_path=self.model_path, config=self.config)
        self.inference.load_model_and_tokenizer()
        
        return self.inference
    
    def _save_to_log(self, log_file, data, message=""):
        """Save data to log file in JSON format"""
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            if message:
                print(f"[LOG] {message}: {log_file}")
        except Exception as e:
            print(f"[!] Error saving log file {log_file}: {e}")
    
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
            print(f"Status: [OK] TARGET ACHIEVED! (BLEU: {bleu_score.score:.2f} > 25.0)")
        else:
            print(f"Status: [!] Target not achieved (Gap: {25.0 - bleu_score.score:.2f})")
            print(f"Current BLEU: {bleu_score.score:.2f}, Target: 25.0")
        
        results = {
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
        
        # Save BLEU results to log
        bleu_log_data = {
            'bleu_score': bleu_score.score,
            'bleu_1': bleu_score.precisions[0],
            'bleu_2': bleu_score.precisions[1],
            'bleu_3': bleu_score.precisions[2],
            'bleu_4': bleu_score.precisions[3],
            'brevity_penalty': bleu_score.bp,
            'num_samples': len(predictions)
        }
        self._save_to_log(self.bleu_log_file, bleu_log_data, "BLEU evaluation results saved")
        
        return results
    
    def calculate_accuracy_metrics(self, test_dataset, num_samples=None):
        """
        Calculate word-level and sentence-level accuracy metrics.
        
        Args:
            test_dataset: Dataset containing Swahili-English pairs
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary containing accuracy metrics
        """
        print("\n" + "=" * 60)
        print("Calculating Accuracy Metrics")
        print("=" * 60)
        
        if self.inference is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        total_samples = len(test_dataset)
        num_samples = num_samples or min(self.config.NUM_EVAL_SAMPLES, total_samples)
        
        if num_samples < total_samples:
            indices = np.random.choice(total_samples, num_samples, replace=False)
        else:
            indices = range(total_samples)
        
        sentence_accuracy = 0
        word_matches = 0
        total_words = 0
        predictions = []
        references = []
        
        for idx in indices:
            example = test_dataset[int(idx)]
            swahili = example['swahili']
            english_ref = example['english']
            
            try:
                english_pred = self.inference.translate(swahili)
                predictions.append(english_pred)
                references.append(english_ref)
                
                # Exact match accuracy
                if english_pred.lower().strip() == english_ref.lower().strip():
                    sentence_accuracy += 1
                
                # Word-level accuracy
                pred_words = english_pred.lower().split()
                ref_words = english_ref.lower().split()
                
                total_words += len(ref_words)
                for pred_w, ref_w in zip(pred_words, ref_words):
                    if pred_w == ref_w:
                        word_matches += 1
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        num_evaluated = len(predictions)
        sentence_acc = (sentence_accuracy / num_evaluated * 100) if num_evaluated > 0 else 0
        word_acc = (word_matches / total_words * 100) if total_words > 0 else 0
        
        print(f"\nSamples evaluated: {num_evaluated}")
        print(f"Sentence-level Accuracy: {sentence_acc:.2f}%")
        print(f"Word-level Accuracy: {word_acc:.2f}%")
        print(f"Exact matches: {sentence_accuracy}/{num_evaluated}")
        
        metrics = {
            'sentence_accuracy': sentence_acc,
            'word_accuracy': word_acc,
            'exact_matches': sentence_accuracy,
            'total_samples': num_evaluated,
            'predictions': predictions,
            'references': references
        }
        
        # Generate confusion matrix if sklearn is available
        if HAS_SKLEARN and len(predictions) > 1:
            print("\nGenerating confusion matrix...")
            self._generate_confusion_matrix(predictions, references)
        
        return metrics
    
    def _generate_confusion_matrix(self, predictions, references):
        """
        Generate and display a confusion matrix for word-level predictions.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
        """
        try:
            # Extract words and create labels
            all_pred_words = []
            all_ref_words = []
            
            for pred, ref in zip(predictions, references):
                pred_words = pred.lower().split()
                ref_words = ref.lower().split()
                
                # Pad to same length
                max_len = max(len(pred_words), len(ref_words))
                pred_words += ['<PAD>'] * (max_len - len(pred_words))
                ref_words += ['<PAD>'] * (max_len - len(ref_words))
                
                all_pred_words.extend(pred_words[:max_len])
                all_ref_words.extend(ref_words[:max_len])
            
            # Get unique words
            unique_words = sorted(set(all_pred_words + all_ref_words))
            word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
            
            # Convert to indices
            pred_indices = np.array([word_to_idx[w] for w in all_pred_words])
            ref_indices = np.array([word_to_idx[w] for w in all_ref_words])
            
            # Create confusion matrix
            cm = confusion_matrix(ref_indices, pred_indices, labels=range(len(unique_words)))
            
            # Print top 10 most confused words
            print("\n" + "=" * 60)
            print("Top Confusion Pairs (Word-level)")
            print("=" * 60)
            
            confusion_pairs = []
            for i in range(len(unique_words)):
                for j in range(len(unique_words)):
                    if i != j and cm[i, j] > 0:
                        confusion_pairs.append((cm[i, j], unique_words[i], unique_words[j]))
            
            # Sort by frequency
            confusion_pairs.sort(reverse=True)
            
            for count, ref_word, pred_word in confusion_pairs[:10]:
                print(f"  Reference: '{ref_word}' -> Predicted: '{pred_word}' (Count: {count})")
            
            # Calculate diagonal accuracy
            correct_predictions = np.trace(cm)
            total_predictions = np.sum(cm)
            matrix_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            
            print(f"\nConfusion Matrix Accuracy: {matrix_accuracy:.2f}%")
            
            # Save confusion matrix to log
            confusion_log_data = {
                'confusion_pairs': [
                    {'count': int(count), 'reference': ref_word, 'predicted': pred_word}
                    for count, ref_word, pred_word in confusion_pairs[:20]
                ],
                'matrix_accuracy': matrix_accuracy,
                'unique_words_count': len(unique_words)
            }
            self._save_to_log(self.confusion_log_file, confusion_log_data, "Confusion matrix saved")
            
            return cm
            
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            return None
    
    def generate_accuracy_report(self, test_dataset, num_samples=None):
        """
        Generate a comprehensive accuracy report with multiple metrics.
        
        Args:
            test_dataset: Dataset to evaluate
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with all accuracy metrics
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE ACCURACY REPORT")
        print("=" * 60)
        
        # Calculate accuracy metrics
        accuracy_metrics = self.calculate_accuracy_metrics(test_dataset, num_samples)
        
        # Calculate BLEU scores
        predictions = accuracy_metrics.pop('predictions')
        references = accuracy_metrics.pop('references')
        
        references_for_bleu = [[ref] for ref in references]
        bleu_score = self.bleu_metric.corpus_score(predictions, references_for_bleu)
        
        # Combine all metrics
        all_metrics = {
            **accuracy_metrics,
            'bleu_score': bleu_score.score,
            'bleu_1': bleu_score.precisions[0],
            'bleu_2': bleu_score.precisions[1],
            'bleu_3': bleu_score.precisions[2],
            'bleu_4': bleu_score.precisions[3],
            'brevity_penalty': bleu_score.bp
        }
        
        # Print comprehensive report
        print("\n" + "=" * 60)
        print("ACCURACY METRICS SUMMARY")
        print("=" * 60)
        print(f"Sentence-level Accuracy: {all_metrics['sentence_accuracy']:.2f}%")
        print(f"Word-level Accuracy:     {all_metrics['word_accuracy']:.2f}%")
        print(f"Exact Matches:           {all_metrics['exact_matches']}/{all_metrics['total_samples']}")
        print(f"\nBLEU Scores:")
        print(f"  BLEU-1: {all_metrics['bleu_1']:.2f}")
        print(f"  BLEU-2: {all_metrics['bleu_2']:.2f}")
        print(f"  BLEU-3: {all_metrics['bleu_3']:.2f}")
        print(f"  BLEU-4: {all_metrics['bleu_4']:.2f}")
        print(f"  Overall BLEU: {all_metrics['bleu_score']:.2f}")
        print(f"  Brevity Penalty: {all_metrics['brevity_penalty']:.3f}")
        
        # Save logs
        self._save_accuracy_logs(all_metrics)
        
        return all_metrics
    
    def _save_accuracy_logs(self, all_metrics):
        """Save accuracy metrics to log files"""
        # Save full metrics
        self._save_to_log(self.accuracy_log_file, all_metrics, "Accuracy metrics saved")
        
        # Save BLEU scores separately
        bleu_data = {
            'bleu_score': all_metrics['bleu_score'],
            'bleu_1': all_metrics['bleu_1'],
            'bleu_2': all_metrics['bleu_2'],
            'bleu_3': all_metrics['bleu_3'],
            'bleu_4': all_metrics['bleu_4'],
            'brevity_penalty': all_metrics['brevity_penalty']
        }
        self._save_to_log(self.bleu_log_file, bleu_data, "BLEU scores saved")
        
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

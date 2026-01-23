"""
Inference Module
Handles translation generation using the fine-tuned BLOOM model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config


class TranslationInference:
    """Handles inference and translation generation"""
    
    def __init__(self, model_path=None, config=None):
        self.config = config or Config()
        self.model_path = model_path or self.config.SAVE_DIR
        self.model = None
        self.tokenizer = None
        self.device = self.config.DEVICE
    
    def load_model_and_tokenizer(self):
        """Load the fine-tuned model and tokenizer"""
        print("\n" + "=" * 60)
        print("Loading Model for Inference")
        print("=" * 60)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.config.USE_FP16 else torch.float32
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from: {self.model_path}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Model dtype: {self.model.dtype}")
        
        return self.model, self.tokenizer
    
    def translate(self, swahili_text, max_new_tokens=None, num_beams=None, 
                  temperature=None, do_sample=False):
        """
        Translate a Swahili sentence to English.
        
        Args:
            swahili_text: Input Swahili sentence
            max_new_tokens: Maximum tokens to generate (default from config)
            num_beams: Number of beams for beam search (default from config)
            temperature: Sampling temperature (default from config)
            do_sample: Whether to use sampling
            
        Returns:
            Translated English text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        num_beams = num_beams or self.config.NUM_BEAMS
        temperature = temperature or self.config.TEMPERATURE
        
        # Create prompt
        prompt = self._create_prompt(swahili_text)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the translation (after "->")
        translation = self._extract_translation(full_output)
        
        return translation
    
    def translate_batch(self, swahili_texts, max_new_tokens=None, num_beams=None):
        """
        Translate multiple Swahili sentences to English.
        
        Args:
            swahili_texts: List of Swahili sentences
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search
            
        Returns:
            List of translated English texts
        """
        translations = []
        
        for text in swahili_texts:
            translation = self.translate(text, max_new_tokens, num_beams)
            translations.append(translation)
        
        return translations
    
    @staticmethod
    def _create_prompt(swahili_text):
        """Create translation prompt"""
        return f"Translate Swahili to English: {swahili_text} ->"
    
    @staticmethod
    def _extract_translation(full_output):
        """Extract translation from model output"""
        if "->" in full_output:
            translation = full_output.split("->")[-1].strip()
        else:
            translation = full_output
        
        return translation
    
    def interactive_translate(self):
        """Interactive translation mode"""
        print("\n" + "=" * 60)
        print("Interactive Translation Mode")
        print("=" * 60)
        print("Enter Swahili text to translate (or 'quit' to exit)")
        print("-" * 60)
        
        while True:
            swahili_input = input("\nSwahili: ").strip()
            
            if swahili_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not swahili_input:
                print("Please enter some text.")
                continue
            
            try:
                translation = self.translate(swahili_input)
                print(f"English: {translation}")
            except Exception as e:
                print(f"Error during translation: {e}")


def translate_examples(model_path=None):
    """
    Function to demonstrate translation with example sentences.
    
    Args:
        model_path: Path to the fine-tuned model
    """
    # Example Swahili sentences
    examples = [
        "Habari yako?",
        "Ninakupenda sana.",
        "Ninaenda shuleni kesho.",
        "Chakula kiko tayari.",
        "Tunaishi Nairobi.",
    ]
    
    # Initialize inference
    inference = TranslationInference(model_path=model_path)
    inference.load_model_and_tokenizer()
    
    print("\n" + "=" * 60)
    print("Example Translations")
    print("=" * 60)
    
    for i, swahili in enumerate(examples, 1):
        translation = inference.translate(swahili)
        print(f"\n{i}. Swahili: {swahili}")
        print(f"   English: {translation}")


if __name__ == "__main__":
    # Run example translations
    translate_examples()
    
    # Start interactive mode
    inference = TranslationInference()
    inference.load_model_and_tokenizer()
    inference.interactive_translate()

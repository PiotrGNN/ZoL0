"""
sentiment_ai.py
--------------
Market sentiment analysis using transformer models.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import os
import re

# Konfiguracja logowania
logger = logging.getLogger("sentiment_analyzer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class MarketSentimentAnalyzer:
    """
    Market sentiment analyzer using FinBERT or similar finance-specific models.
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: str = "saved_models/sentiment"
    ):
        self.logger = logging.getLogger("MarketSentimentAnalyzer")
        self.device = device
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.last_update = datetime.now()
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            self._load_model()
            self.logger.info(f"Model loaded successfully from {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Loads the transformer model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """Preprocesses input text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions (common in social media)
        text = re.sub(r'@\w+', '', text)
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        return text

    def analyze(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Analyzes market sentiment from text.
        
        Args:
            texts: Single text or list of texts to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]
                
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Tokenize
            inputs = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                
            # Convert predictions to numpy for easier handling
            predictions = predictions.cpu().numpy()
            
            # Calculate aggregated sentiment
            sentiments = []
            for pred in predictions:
                if len(pred) == 3:  # Positive, Negative, Neutral
                    sentiment = {
                        "positive": float(pred[0]),
                        "negative": float(pred[1]),
                        "neutral": float(pred[2]),
                        "compound": float(pred[0] - pred[1])  # Compound score
                    }
                else:  # Binary classification
                    sentiment = {
                        "positive": float(pred[1]),
                        "negative": float(pred[0]),
                        "neutral": 0.0,
                        "compound": float(pred[1] - pred[0])
                    }
                sentiments.append(sentiment)
            
            # Calculate aggregated results
            avg_sentiment = {
                "positive": np.mean([s["positive"] for s in sentiments]),
                "negative": np.mean([s["negative"] for s in sentiments]),
                "neutral": np.mean([s["neutral"] for s in sentiments]),
                "compound": np.mean([s["compound"] for s in sentiments])
            }
            
            # Update last update time
            self.last_update = datetime.now()
            
            return {
                "individual_sentiments": sentiments,
                "aggregated_sentiment": avg_sentiment,
                "timestamp": self.last_update.isoformat(),
                "analyzed_texts": len(texts)
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def fine_tune(
        self,
        texts: List[str],
        labels: List[int],
        validation_split: float = 0.2,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ) -> Dict[str, Any]:
        """
        Fine-tunes the model on market-specific data.
        
        Args:
            texts: List of texts to train on
            labels: Corresponding sentiment labels
            validation_split: Proportion of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            
        Returns:
            Training metrics
        """
        try:
            # Prepare dataset
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=validation_split
            )
            
            # Convert to HuggingFace datasets
            def create_dataset(texts, labels):
                return Dataset.from_dict({
                    "text": texts,
                    "label": labels
                })
            
            train_dataset = create_dataset(train_texts, train_labels)
            val_dataset = create_dataset(val_texts, val_labels)
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.cache_dir, "results"),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir=os.path.join(self.cache_dir, "logs"),
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer
            )
            
            # Train model
            train_result = trainer.train()
            
            # Evaluate
            eval_result = trainer.evaluate()
            
            # Save fine-tuned model
            self.model.save_pretrained(
                os.path.join(self.cache_dir, "fine_tuned")
            )
            self.tokenizer.save_pretrained(
                os.path.join(self.cache_dir, "fine_tuned")
            )
            
            return {
                "train_loss": float(train_result.training_loss),
                "eval_loss": float(eval_result["eval_loss"]),
                "train_samples": len(train_texts),
                "eval_samples": len(val_texts),
                "epochs": epochs,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in fine-tuning: {str(e)}")
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Returns the current status of the analyzer"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "last_update": self.last_update.isoformat(),
            "cache_dir": self.cache_dir
        }

    def save(self, path: str) -> bool:
        """Saves the model and tokenizer"""
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def load(self, path: str) -> bool:
        """Loads a saved model and tokenizer"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MarketSentimentAnalyzer()
    
    # Example texts
    texts = [
        "Bitcoin price surges to new all-time high as institutional adoption grows",
        "Market crashes as major hedge fund liquidates positions",
        "Trading volumes remain stable despite regulatory uncertainty"
    ]
    
    # Analyze sentiment
    results = analyzer.analyze(texts)
    
    # Print results
    for i, (text, sentiment) in enumerate(zip(texts, results["individual_sentiments"])):
        print(f"\nText {i+1}: {text}")
        print(f"Sentiment: {sentiment}")
    
    print("\nAggregated Sentiment:", results["aggregated_sentiment"])
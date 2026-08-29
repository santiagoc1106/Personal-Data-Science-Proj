"""
Movie Overview + Emotion Analysis Pipeline
===========================================
1. Fine-tunes DistilBERT on IMDB for sentiment classification (with early-stopping
   fix to avoid overfitting).
2. Uses Qwen2.5-1.5B-Instruct to generate a short plot overview for any movie title.
3. Runs that overview through an emotion classifier to get a full emotion breakdown.

Run sections independently by commenting out what you don't need re-run each time
(e.g. skip TRAINING if you already have a saved model in ./my-model-fixed).
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import evaluate

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"       # only show actual errors, not info/warnings
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"      # suppress download progress bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"        # silences a common tokenizer warning
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1" # suppress advisory-style warnings

import warnings
warnings.filterwarnings("ignore")  # catch any remaining Python-level warnings (e.g. deprecation notices)

from transformers.utils import logging
logging.set_verbosity_error()


# ============================================================
# SECTION 1: TRAINING THE SENTIMENT CLASSIFIER
# ============================================================

def train_sentiment_model(save_dir="./my-model-fixed"):
    """Fine-tunes DistilBERT on IMDB for binary sentiment classification.

    Uses load_best_model_at_end so the final model is whichever epoch had the
    LOWEST validation loss, not just the last epoch trained — this avoids
    keeping an overfit checkpoint even if its raw accuracy looked slightly higher.
    """

    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    print("Loading tokenizer and tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        fp16=True,
        logging_steps=10,
        load_best_model_at_end=True,       # keep the best-generalizing checkpoint
        metric_for_best_model="eval_loss", # judge "best" by validation loss
        greater_is_better=False,           # lower loss is better
        report_to="none",                  # skip wandb/etc prompts
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Saving fixed model to {save_dir}...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    return trainer.model, tokenizer


def load_sentiment_model(save_dir="./my-model-fixed"):
    """Loads a previously trained + saved sentiment model instead of retraining."""
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    return model, tokenizer


def check_sentiment(text, model, tokenizer):
    """Returns (predicted_label, probabilities) for a piece of text.

    Useful for sanity-checking model calibration — an overfit model tends to
    be falsely overconfident even on ambiguous text, while a well-fit model
    should show more balanced probabilities when the sentiment is genuinely mixed.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        probs = torch.softmax(outputs.logits, dim=-1)
    return prediction, probs


# ============================================================
# SECTION 2: MOVIE OVERVIEW + EMOTION PIPELINE
# ============================================================

def load_generator():
    """Loads the instruction-tuned text-generation model for plot overviews."""
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
    )


def load_emotion_classifier():
    """Loads the emotion classifier (returns scores across all emotion labels)."""
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )


def generate_movie_overview(movie_title, generator, emotion_classifier):
    """Generates a short plot overview for a movie and analyzes its emotional tone.

    Note: the overview is LLM-generated from the model's own knowledge, not pulled
    from a factual database — plot details (character names, settings, specific
    events) may be inaccurate, especially for smaller/older movies. Treat the
    overview as illustrative text for the emotion analysis, not a factual summary.
    """
    messages = [
        {"role": "user", "content": f"Please write a 2-3 sentence overview of the plot of {movie_title}"}
    ]

    result = generator(messages, max_new_tokens=100, do_sample=True, temperature=0.7)
    overview_text = result[0]["generated_text"][-1]["content"]

    emotions = emotion_classifier(overview_text)[0]  # sorted highest -> lowest

    print(f"Movie: {movie_title}")
    print(f"Overview: {overview_text}")
    print(f"\nTop emotion: {emotions[0]['label']} ({emotions[0]['score']:.2%})")
    print("\nFull emotion scale:")
    for emotion in emotions:
        print(f"  {emotion['label']}: {emotion['score']:.2%}")

    return overview_text, emotions


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --- Sentiment model: train fresh, or load if already saved ---
    TRAIN_FROM_SCRATCH = False  # set True to retrain instead of loading a saved model

    if TRAIN_FROM_SCRATCH:
        sentiment_model, sentiment_tokenizer = train_sentiment_model()
    else:
        sentiment_model, sentiment_tokenizer = load_sentiment_model("./my-model-fixed")

    # Quick sanity check
    label, probs = check_sentiment(
        "This movie was absolutely fantastic!", sentiment_model, sentiment_tokenizer
    )
    print(f"Sentiment check -> label: {label}, probs: {probs}")

    # --- Movie overview + emotion pipeline ---
    generator = load_generator()
    emotion_classifier = load_emotion_classifier()

    movie_title = input("Enter a movie title to generate an overview and analyze its emotional tone: ")

    generate_movie_overview(movie_title, generator, emotion_classifier)
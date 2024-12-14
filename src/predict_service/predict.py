import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
from peft import PeftModel, PeftConfig
from loguru import logger
from time import time


def initialize_model_with_lora(base_model_name, lora_weights_path):
    logger.info(f"start download tokenizer{base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    logger.info(f"end download tokenizer{base_model_name}")

    logger.info(f"start download model{base_model_name}")
    open_time = time()
    model = XLMRobertaForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    close_time = time()
    logger.info(f"end download model{base_model_name}, time: {close_time - open_time}")

    logger.info(f"start download lora adapter {lora_weights_path}")
    open_time = time()
    model = PeftModel.from_pretrained(model, lora_weights_path)
    close_time = time()
    logger.info(f"end download lora adapter {lora_weights_path} time: {close_time - open_time}")

    model.eval()
    return model, tokenizer


def process_text(model, tokenizer, text, max_length=512):

    logger.info(f"get token for {text}")
    tokens = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**tokens)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    predictions = probabilities[0, 1].item()
    return predictions
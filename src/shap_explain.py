import shap, torch
from transformers import AutoTokenizer
from src.models.text_model import TextClassifier

class HFTextPredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = device

    def __call__(self, texts):
        enc = self.tok(texts, truncation=True, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            return out["logits"].softmax(1).cpu().numpy()


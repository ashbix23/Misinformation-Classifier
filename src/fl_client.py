import flwr as fl, torch, os
from torch.optim import AdamW
from src.data.datasets import LoaderConfig, build_text_loaders
from src.models.text_model import TextClassifier
from src.config import Paths, DATASETS

class FakeNewsClient(fl.client.NumPyClient):
    def __init__(self, csv_path: str, model_name: str = "roberta-base", epochs: int = 1, batch_size: int = 16):
        cfg = LoaderConfig(batch_size=batch_size)
        self.train_dl, _, self.test_dl, extras = build_text_loaders(csv_path, model_name, cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TextClassifier(model_name, lora=True).to(self.device)
        self.opt = AdamW(self.model.parameters(), lr=5e-5)
        self.epochs = epochs
        self.class_weights = extras["class_weights"].to(self.device)

    def get_parameters(self, config):
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        sd = self.model.state_dict()
        for (k,_), p in zip(sd.items(), parameters):
            sd[k] = torch.tensor(p)
        self.model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        crit = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        for _ in range(self.epochs):
            for b in self.train_dl:
                self.opt.zero_grad()
                b = {k:(v.to(self.device) if torch.is_tensor(v) else v) for k,v in b.items()}
                out = self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                loss = crit(out["logits"], b["labels"])
                loss.backward(); self.opt.step()
        return self.get_parameters(config), len(self.train_dl.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for b in self.test_dl:
                b = {k:(v.to(self.device) if torch.is_tensor(v) else v) for k,v in b.items()}
                pred = self.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])["logits"].argmax(1)
                correct += (pred == b["labels"]).sum().item()
                total += b["labels"].size(0)
        acc = correct / max(1,total)
        return float(acc), len(self.test_dl.dataset), {"accuracy": float(acc)}


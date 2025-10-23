import os
import torch
from torch.optim import AdamW
from tqdm import tqdm
import argparse

from src.config import Paths, DATASETS
from src.utils.seed import set_seed
from src.utils.metrics import classification_report_dict
from src.utils.io import save_json
from src.data.datasets import LoaderConfig, build_text_loaders, build_multimodal_loaders
from src.models.text_model import TextClassifier
from src.models.clip_fusion import CLIPFusionClassifier

# -------------------------------------------------------------
# Training and Evaluation Functions
# -------------------------------------------------------------
def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in tqdm(loader, leave=False):
        optimizer.zero_grad()
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(**{k: batch[k] for k in batch if k in ["input_ids", "attention_mask", "labels", "image_features"]})
        logits, labels = outputs["logits"], batch["labels"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = model(**{k: batch[k] for k in batch if k in ["input_ids", "attention_mask", "labels", "image_features"]})
        y_pred.extend(outputs["logits"].argmax(1).cpu().tolist())
        y_true.extend(batch["labels"].cpu().tolist())
    return classification_report_dict(y_true, y_pred)


# -------------------------------------------------------------
# Main Training Loop
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train text or multimodal FakeNews-Next models.")
    parser.add_argument("--mode", choices=["text", "fusion"], default="text",
                        help="text for RoBERTa model, fusion for CLIP-based multimodal model")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Datasets to train on (e.g., ISOT FakeNews LIAR)")
    parser.add_argument("--model_name", default="roberta-base", help="Base model name (Hugging Face identifier)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    paths = Paths()
    os.makedirs(paths.models_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for ds in args.datasets:
        print(f"========== Training on {ds} ==========")
        csv = os.path.join(paths.processed_dir, DATASETS[ds])
        cfg = LoaderConfig(batch_size=args.batch_size, seed=args.seed)

        if args.mode == "text":
            train_dl, val_dl, test_dl, extras = build_text_loaders(csv, args.model_name, cfg)
            model = TextClassifier(args.model_name, lora=args.lora).to(device)
            class_weights = extras["class_weights"].to(device)
        else:
            train_dl, val_dl, test_dl = build_multimodal_loaders(csv, args.model_name, cfg)
            model = CLIPFusionClassifier(args.model_name).to(device)
            class_weights = None

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(model.parameters(), lr=args.lr)

        best_f1, best_path = -1.0, None
        save_dir = os.path.join(paths.text_dir if args.mode == "text" else paths.fusion_dir, ds)
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")
            train_loss, train_acc = train_epoch(model, train_dl, optimizer, device, criterion)
            val_report = eval_epoch(model, val_dl, device)
            print(f"[{ds}] Epoch {epoch}: loss={train_loss:.4f}, train_acc={train_acc:.3f}, val_f1={val_report['f1']:.3f}")

            if val_report["f1"] > best_f1:
                best_f1 = val_report["f1"]
                best_path = os.path.join(save_dir, "best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved: {best_path}")

        model.load_state_dict(torch.load(best_path, map_location=device))
        test_report = eval_epoch(model, test_dl, device)
        results[ds] = {
            "val_best_f1": best_f1,
            "test": test_report,
            "ckpt": best_path
        }
        print(f"Test results for {ds}: {test_report}")

    save_json(results, paths.results_file)
    print(f"All training complete. Results saved to {paths.results_file}")


if __name__ == "__main__":
    main()


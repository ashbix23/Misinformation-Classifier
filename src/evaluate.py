import argparse, os, torch
from src.config import Paths, DATASETS
from src.utils.metrics import classification_report_dict
from src.data.datasets import LoaderConfig, build_text_loaders
from src.models.text_model import TextClassifier

@torch.no_grad()
def soft_ensemble(models, loader, device):
    for m in models: m.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        probs = []
        for m in models:
            out = m(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs.append(out["logits"].softmax(1))
        avg = torch.stack(probs).mean(0)
        y_pred.extend(avg.argmax(1).cpu().tolist())
        y_true.extend(batch["labels"].cpu().tolist())
    return classification_report_dict(y_true, y_pred)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ISOT","FakeNews","LIAR"])
    p.add_argument("--ensemble", choices=["soft","none"], default="soft")
    p.add_argument("--model_name", default="roberta-base")
    args = p.parse_args()

    paths = Paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LoaderConfig(batch_size=32)

    # use ISOT loader as reference test set for cross-dataset ensemble demo
    csv = os.path.join(paths.processed_dir, DATASETS["ISOT"])
    _, _, test_dl, _ = build_text_loaders(csv, args.model_name, cfg)

    ckpts = []
    for ds in ["ISOT","FakeNews","LIAR"]:
        ckpts.append(os.path.join(paths.text_dir, ds, "best.pt"))

    models = []
    for ck in ckpts:
        m = TextClassifier(args.model_name, lora=True).to(device)
        m.load_state_dict(torch.load(ck, map_location=device))
        m.eval(); models.append(m)

    report = soft_ensemble(models, test_dl, device)
    print("Ensemble on ISOT test:", report)

if __name__ == "__main__":
    main()


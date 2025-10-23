import gradio as gr, torch, os
from transformers import AutoTokenizer
from src.models.text_model import TextClassifier
from src.config import Paths

MODEL_NAME = "roberta-base"
CKPT = os.path.join(Paths().text_dir, "ISOT", "best.pt")  # swap to ensemble wrapper later

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(MODEL_NAME, lora=True).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def predict(text: str):
    enc = tok(text, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        probs = out["logits"].softmax(1).squeeze(0)
    label = int(probs.argmax().item())
    return f"Prediction: {'FAKE' if label==1 else 'REAL'} (confidence {probs.max().item():.2f})"

iface = gr.Interface(fn=predict, inputs=gr.Textbox(lines=4), outputs="text", title="FakeNews-Next")
if __name__ == "__main__":
    iface.launch()


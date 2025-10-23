from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def classification_report_dict(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "confusion": cm}


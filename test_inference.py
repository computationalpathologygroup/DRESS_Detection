from models.model_clam import CLAM_SB
from dataset_modules.dataset_generic import Generic_MIL_Dataset

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

# df = pd.read_csv('dataset_csv/dataset_split.csv')
# df_test = df[df['split'] == 'test']
# df_test.to_csv('dataset_csv/test_split.csv', index=False)

# Initialize model
model = CLAM_SB(
    gate=True,
    size_arg="small",
    dropout=0.25,
    k_sample=8,
    n_classes=2,
    subtyping=False,
    embed_dim=1536
)

# Load the trained weights
checkpoint = torch.load("results/s_2_checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint, strict=False)
model.eval()

test_dataset = Generic_MIL_Dataset(
    csv_path='dataset_csv/test_split.csv',
    data_dir='features_path/',
    shuffle=False,
    seed=123,
    print_info=True,
    label_dict={'DRESS': 0, 'MDE': 1},
    patient_strat=False,
    ignore=[]
)

loader = DataLoader(test_dataset, batch_size=1)
all_probs, all_preds, all_labels = [], [], []

with torch.no_grad():
    for features, label, _ in loader:
        logits, prob, pred, _, _ = model(features.squeeze(0), label=label, return_topk_features=False)
        all_probs.append(prob.item())
        all_preds.append(pred.item())
        all_labels.append(label.item())

acc = accuracy_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)

print(f"Accuracy:  {acc:.4f}")
print(f"ROC AUC:   {auc:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from tqdm import tqdm

from models.model_clam import ABMIL  # Assuming ABMIL was added under model_clam.py
from dataset_modules.dataset_generic import Generic_MIL_Dataset


def load_model(model_path, device, input_dim=1024, n_classes=2):
    model = ABMIL(n_classes=n_classes, embed_dim=input_dim)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_ensemble(models, datasets, slide_ids, labels, device):
    all_probs = []

    for model, dataset in zip(models, datasets):
        model_probs = []
        for i in range(len(dataset)):
            features, _, _ = dataset[i]
            features = features.to(device)
            with torch.no_grad():
                output = model(features)
                if isinstance(output, tuple):
                    output = output[0]
                prob = torch.softmax(output, dim=1).cpu().numpy()  # shape: [1, 2]
            model_probs.append(prob)
        model_probs = np.vstack(model_probs)  # shape: [num_samples, 2]
        all_probs.append(model_probs)

    avg_probs = np.mean(all_probs, axis=0)  # shape: [num_samples, 2]
    preds = np.argmax(avg_probs, axis=1)

    assert len(labels) == len(preds), f"Length mismatch: labels={len(labels)}, preds={len(preds)}"

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, avg_probs[:, 1])

    return {
        "slide_ids": slide_ids,
        "y_true": labels,
        "y_pred": preds,
        "y_prob": avg_probs,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_paths', nargs='+', required=True)
    parser.add_argument('--feature_dirs', nargs='+', required=True)
    parser.add_argument('--split_dir', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./ensemble_results')
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=1536)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metadata_df = pd.read_csv(args.metadata)
    metadata_df['slide_id'] = metadata_df['slide_id'].astype(str).str.strip()

    folds = sorted([
    f for f in os.listdir(args.split_dir)
    if re.fullmatch(r'splits_\\d+\\.csv', f)
])
    results = []

    full_df = pd.read_csv(args.metadata)
    best_auc = 0
    best_fold_df = None

    for fold_idx, split_file in enumerate(folds):
        print(f"=== Fold {fold_idx} ===")
        # split_path = os.path.join(args.split_dir, f'splits_{fold}.csv')
        # split_df = pd.read_csv(split_path)
        # test_ids = split_df['test'].dropna().astype(str).tolist()
        # test_df = full_df[full_df['case_id'].astype(str).isin(test_ids)]
        split_df = pd.read_csv(os.path.join(args.split_dir, split_file))
        test_ids = split_df['test'].dropna().astype(str).str.strip().tolist()
        test_df = metadata_df[metadata_df['slide_id'].isin(test_ids)].reset_index(drop=True)

        if test_df.empty:
            print(f"[!] No matching test slides for fold {fold_idx}, skipping.")
            continue

        datasets = []
        for feat_dir in args.feature_dirs:
            ds = Generic_MIL_Dataset(
                data_dir=feat_dir,
                csv_path=args.metadata,
                shuffle=False,
                seed=42,
                print_info=False,
                label_dict={'DRESS': 0, 'MDE': 1},
                patient_strat=False,
                ignore=[]
            )
            ds.slide_data = test_df.reset_index(drop=True)
            ds.load_from_h5(True)
            datasets.append(ds)

        models = [load_model(p, device, args.input_dim, args.n_classes) for p in args.model_paths]
        slide_ids = test_df['case_id'].tolist()
        label_map = {'DRESS': 0, 'MDE': 1}
        labels = test_df['label'].map(label_map).tolist()

        eval_result  = evaluate_ensemble(models, datasets, slide_ids, labels, device)

        metrics = eval_result ['metrics']
        print(f"AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")

        df_out = pd.DataFrame({
            "slide_id": eval_result["slide_ids"],
            "y_true": eval_result["y_true"],
            "y_pred": eval_result["y_pred"],
            "prob_class_0": eval_result["y_prob"][:, 0],
            "prob_class_1": eval_result["y_prob"][:, 1],
        })
        df_out.to_csv(os.path.join(args.output_dir, f'ensemble_fold{fold_idx}.csv'), index=False)

        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_fold_df = df_out.copy()
            best_fold_df.to_csv(os.path.join(args.output_dir, f'ensemble_best.csv'), index=False)
            print("✅ Updated best ensemble saved.")

if __name__ == '__main__':
    main()

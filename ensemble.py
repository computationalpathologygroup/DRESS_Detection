import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from models.model_clam import CLAM_SB
from dataset_modules.dataset_generic import Generic_MIL_Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(
        description="Late fusion ensemble using Generic_MIL_Dataset.")
    parser.add_argument('--model_paths', nargs='+', required=True,
                        help='Paths to model checkpoint files (one per encoder)')
    parser.add_argument('--feature_dirs', nargs='+', required=True,
                        help='Directories containing .h5 files for each encoder (order must match models)')
    parser.add_argument('--dataset_csv', type=str,
                        required=True, help='Path to dataset_split.csv')
    parser.add_argument('--output_dir', type=str, default='ensemble_results.csv',
                        help='Path to save CSV with predictions')
    parser.add_argument('--embed_dim', type=int, default=1536,
                        help='Feature embedding dimension')
    parser.add_argument('--size_arg', type=str,
                        choices=['small', 'big'], default='small', help='Model size argument')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of output classes')
    parser.add_argument('--dropout', type=float,
                        default=0.25, help='Dropout used in model')
    parser.add_argument('--k_sample', type=int, default=15,
                        help='Number of patches sampled for instance-level training')
    parser.add_argument('--random_topk', action='store_true', help='Use random Top-K patch selection instead of attention-based')

    return parser.parse_args()

def load_clam_model(checkpoint_path, args, device):
    model = CLAM_SB(
        gate=True,
        dropout=args.dropout,
        k_sample=args.k_sample,
        subtyping=False,
        embed_dim=args.embed_dim,
        size_arg=args.size_arg,
        n_classes=args.n_classes,
        instance_loss_fn=torch.nn.CrossEntropyLoss()
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


def get_model_probs(model, features, device, use_random_topk=True):
    with torch.no_grad():
        features = features.to(device)
        _, probs, _, _, _ = model(features, return_topk_features=True, use_random_topk=use_random_topk)
        return probs.cpu().numpy()  # shape: [1, num_classes]

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    full_df = pd.read_csv(args.dataset_csv)
    full_df['case_id'] = full_df['case_id'].astype(str)  # Ensure matching

    all_acc, all_f1, all_auc = [], [], []

    for fold in range(args.cv_folds):
        print(f"\n=== Fold {fold + 1}/{args.cv_folds} ===")
        split_path = os.path.join(args.cv_split_dir, f'splits_{fold}.csv')
        print(f"Loading split file: {split_path}")

        split_df = pd.read_csv(split_path, dtype=str)
        test_ids = split_df['test'].dropna().astype(str).tolist()
        test_df = full_df[full_df['case_id'].isin(test_ids)].reset_index(drop=True)

        datasets = []
        for i, feat_dir in enumerate(args.feature_dirs):
            dataset = Generic_MIL_Dataset(
                csv_path=test_df,
                data_dir=feat_dir,
                shuffle=False,
                seed=42,
                print_info=(i == 0),
                label_dict={'DRESS': 0, 'MDE': 1},
                patient_strat=False,
                ignore=[]
            )
            dataset.load_from_h5(True)
            datasets.append(dataset)

        models = [load_clam_model(mp, args, device) for mp in args.model_paths]
        num_samples = len(datasets[0])

        fold_preds, fold_probs, fold_labels, slide_ids = [], [], [], []
        for model, dataset in zip(models, datasets):
            features, label, _ = dataset[i]
            features = features.to(device)
            probs = get_model_probs(model, features, device, use_random_topk=args.random_topk)  # shape: [1, C]
            probs_list.append(probs)

        for i in tqdm(range(num_samples)):
            slide_id = datasets[0].slide_data.loc[i, 'slide_id']
            slide_ids.append(slide_id)

            probs_list = []
            for model, dataset in zip(models, datasets):
                features, label, _ = dataset[i]
                features = features.to(device)
                probs = get_model_probs(model, features, device)
                probs_list.append(probs)

            probs = np.stack(probs_list, axis=0)
            avg_probs = np.mean(probs, axis=0)
            pred = np.argmax(avg_probs, axis=1)[0]

            fold_preds.append(pred)
            fold_probs.append(avg_probs)
            fold_labels.append(label)

        y_true = fold_labels
        y_pred = fold_preds
        y_probs = np.vstack(fold_probs)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs[:, 1])

        print(f"Fold {fold}: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
        all_acc.append(acc)
        all_f1.append(f1)
        all_auc.append(auc)

        pd.DataFrame({
            "slide": slide_ids,
            "ground_truth": y_true,
            "prediction": y_pred,
            "prob_class_0": y_probs[:, 0],
            "prob_class_1": y_probs[:, 1]
        }).to_csv(os.path.join(args.output_dir, f"fold_{fold}_results.csv"), index=False)

    print("\n=== Cross-Validation Summary ===")
    print(f"Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"F1 Score: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"AUC     : {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")

if __name__ == "__main__":
    main()

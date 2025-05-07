import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from models.model_clam import CLAM_SB
from dataset_modules.dataset_generic import Generic_MIL_Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Late fusion ensemble using Generic_MIL_Dataset.")
    parser.add_argument('--model_paths', nargs='+', required=True,
                        help='Paths to model checkpoint files (one per encoder)')
    parser.add_argument('--feature_dirs', nargs='+', required=True,
                        help='Directories containing .h5 files for each encoder (order must match models)')
    parser.add_argument('--dataset_csv', type=str,
                        required=True, help='Path to dataset_split.csv')
    parser.add_argument('--output_csv', type=str, default='ensemble_results.csv',
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
    return parser.parse_args()


def load_from_h5(self, toggle):
    self.use_h5 = toggle


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


def get_model_probs(model, features, device):
    with torch.no_grad():
        features = features.to(device)
        _, probs, _, _, _ = model(features)
        return probs.cpu().numpy()  # shape: [1, num_classes]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert len(args.model_paths) == len(
        args.feature_dirs), "Mismatch: You must provide one feature_dir per model_path."

    print("Loading models...")
    models = [load_clam_model(mp, args, device) for mp in args.model_paths]
    print(f"{len(models)} models loaded.")

    print("Preparing test dataset...")
    datasets = []
    for i, feat_dir in enumerate(args.feature_dirs):
        dataset = Generic_MIL_Dataset(
            csv_path=args.dataset_csv,
            data_dir=feat_dir,
            shuffle=False,
            seed=42,
            print_info=(i == 0),
            label_dict={'DRESS': 0, 'MDE': 1},
            patient_strat=False,
            ignore=[]
        )
        dataset.load_from_h5(True)  # 👈 force h5 loading
        datasets.append(dataset)

    num_samples = len(datasets[0])
    all_preds, all_probs, all_labels, all_filenames = [], [], [], []
    print(f"slide id: {len(datasets[0].slide_data['case_id'].values)}")
    print(f"encoded Labels: {len(datasets[0].slide_data['label'].values)}")

    print("Running ensemble inference...")
    for i in tqdm(range(num_samples)):
        slide_id = datasets[0].slide_data.loc[i, 'slide_id']
        # label = datasets[0].slide_data.loc[i, 'label'].values
        # label_int = 0 if label == 'DRESS' else 1
        # all_labels.append(label)
        all_filenames.append(slide_id)

        probs_list = []

        for model, dataset in zip(models, datasets):
            features, label, _ = dataset[i]
            features = features.to(device)
            probs = get_model_probs(model, features, device)  # shape: [1, C]
            probs_list.append(probs)

        #--------Mean fusion--------
        # avg_probs = sum(probs_list) / len(probs_list)
        # pred = np.argmax(avg_probs, axis=1)[0]

        #--------Max fusion--------
        # avg_probs = np.max(probs_list, axis=0)
        # pred = np.argmax(avg_probs, axis=1)[0]

        #--------Product fusion--------
        probs = np.stack(probs_list, axis=0)
        fused_probs = np.prod(probs, axis=0)
        pred = np.argmax(fused_probs, axis=1)[0]

        all_preds.append(pred)
        all_probs.append(fused_probs)
        all_labels.append(label)

    # Metrics
    y_true = all_labels
    y_pred = all_preds
    y_probs = np.vstack(all_probs)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs[:, 1])

    print("\n=== Ensemble Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    # Save predictions to CSV
    df = pd.DataFrame({
        "slide": all_filenames,
        "ground_truth": y_true,
        "prediction": y_pred,
        "prob_class_0": y_probs[:, 0],
        "prob_class_1": y_probs[:, 1]
    })
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

import argparse
import os
import h5py
import torch
import torch.nn as nn
import random
import pandas as pd
import numpy as np
from models.model_clam import CLAM_SB
from dataset_modules.dataset_generic import Generic_MIL_Dataset
from visualization import zoom_coords
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ZoomFusionClassifier(nn.Module):
    def __init__(self, feature_dim=1536, n_classes=2, fusion='avg'):
        super().__init__()
        self.fusion = fusion
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, feats_10x, feats_20x):
        if self.fusion == 'avg':
            fused = (feats_10x + feats_20x) / 2
        elif self.fusion == 'sum':
            fused = feats_10x + feats_20x
        else:
            raise ValueError(
                "Invalid fusion method. Choose 'avg', 'sum'")

        pooled = fused.mean(dim=0, keepdim=True)  # Global average pooling
        logits = self.classifier(pooled)  # Shape [1, n_classes]
        probs = torch.softmax(logits, dim=1)
        return probs, logits


def load_model(checkpoint_path, device):
    model = CLAM_SB(
        gate=True,
        size_arg="small",
        dropout=0.25,
        k_sample=500,
        n_classes=2,
        subtyping=False,
        embed_dim=1536
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model


def find_nearest(coords_array, target_coord):
    distances = np.linalg.norm(coords_array - target_coord, axis=1)
    nearest_idx = np.argmin(distances)
    return nearest_idx


def extract_topk_features_lowmag(model, dataset, sample_idx):
    slide, label, coords = dataset[sample_idx]
    features = slide.to(device)

    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, result_dict = model(
            features, label=torch.tensor(label), return_topk_features=True)

    topk_ids = result_dict['topk_ids']
    topk_features = features[topk_ids]
    topk_coords_10x = coords[topk_ids.cpu().numpy()]

    return topk_coords_10x, topk_features, label


def map_topk_to_highmag(topk_coords_10x, feature_20x_path, zoom_factor=2):
    topk_coords_20x_est = zoom_coords(topk_coords_10x, zoom_factor=zoom_factor)

    with h5py.File(feature_20x_path, 'r') as f:
        coords_20x = np.array(f['coords'])
        features_20x = np.array(f['features'])

    nearest_indices = [find_nearest(coords_20x, coord)
                       for coord in topk_coords_20x_est]
    features_highmag_topk = features_20x[nearest_indices]

    return features_highmag_topk


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Run Zoom Fusion Inference")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--checkpoint_path", type=str,
                        required=True, help="Path to the model checkpoint")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the dataset CSV")
    parser.add_argument("--features_dir_10x", type=str,
                        required=True, help="Path to 10x feature directory")
    parser.add_argument("--features_dir_20x", type=str,
                        required=True, help="Path to 20x feature directory")
    parser.add_argument("--output_csv", type=str,
                        required=True, help="Path to output CSV file")
    parser.add_argument("--fusion", type=str, default="abg",
                        choices=['avg', 'sum'], help='Fusion method')
    args = parser.parse_args()

    set_seed(args.seed)
    all_true = []
    all_pred = []
    all_probs = []

    model = load_model(args.checkpoint_path, device)

    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_path,
        data_dir=args.features_dir_10x,
        shuffle=False,
        seed=24,
        print_info=True,
        label_dict={'DRESS': 0, 'MDE': 1},
        patient_strat=False,
        ignore=[]
    )

    for sample_idx in range(len(dataset)):

        slide_path = dataset.slide_data.loc[sample_idx, 'path']
        topk_coords_10x, topk_features_10x, true_label = extract_topk_features_lowmag(
            model, dataset, sample_idx)
        print(
            f"[Sample {sample_idx}] Number of Top-K 10x coords: {len(topk_coords_10x)}")

        feature_20x_path = os.path.join(
            args.features_dir_20x, 'h5_files',
            os.path.basename(slide_path) + ".h5"
        )

        features_highmag_topk = map_topk_to_highmag(
            topk_coords_10x, feature_20x_path)

        # Convert to tensor if needed
        if not isinstance(features_highmag_topk, torch.Tensor):
            topk_features_20x = torch.from_numpy(
                features_highmag_topk).to(topk_features_10x.device)
        else:
            topk_features_20x = features_highmag_topk

        fusion_model = ZoomFusionClassifier(
            feature_dim=1536, n_classes=2, fusion=args.fusion).to(device)
        prob_fused, logits_fused = fusion_model(
            topk_features_10x, topk_features_20x)
        pred_fused = torch.softmax(prob_fused, dim=1)
        pred_label = torch.argmax(pred_fused, dim=1).item()
        prob_np = prob_fused.squeeze(0).detach().cpu().numpy()
        all_true.append(int(true_label))
        all_pred.append(pred_label)
        all_probs.append(prob_np[1])  # Prob of class 1

        row = {
            "true_label": int(true_label),
            "pred_label": pred_label,
            "prob_0": prob_np[0],
            "prob_1": prob_np[1],
        }
        # write to CSV (keep this as-is)
        df = pd.DataFrame([row])
        output_csv = "dataset_csv/zoom_fusion_results_sum_seed2.csv"
        if not os.path.exists(args.output_csv) and sample_idx == 0:
            df.to_csv(args.output_csv, index=False)
        else:
            df.to_csv(args.output_csv, mode='a', header=False, index=False)
    precision = precision_score(all_true, all_pred)
    recall = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    accuracy = accuracy_score(all_true, all_pred)
    auc = roc_auc_score(all_true, all_probs)

    print("=========METRICS=========")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    main()

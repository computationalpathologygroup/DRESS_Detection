import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.wsi_dataset import Wsi_Region
from utils.transform_utils import get_eval_transforms
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore
from utils.constants import MODEL2CONSTANTS
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score2percentile(score, ref):   #expect single scalar value
    percentile = percentileofscore(ref, score)
    return percentile   


def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level=-1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)

    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    heatmap = wsi_object.visHeatmap(
        scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap


def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object


def compute_from_patches(wsi_object, img_transforms, feature_extractor=None, clam_pred=None, model=None, batch_size=512,
                         attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']
    print("Patch extraction region:")
    print("  top_left:", top_left)
    print("  bot_right:", bot_right)
    print("  patch size:", patch_size)

    roi_dataset = Wsi_Region(wsi_object, t=img_transforms, **wsi_kwargs)
    roi_loader = get_simple_loader(
        roi_dataset, batch_size=batch_size, num_workers=0)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)
    mode = "w"
    for idx, (roi, coords) in enumerate(tqdm(roi_loader)):
        roi = roi.to(device)
        coords = coords.numpy()
        with torch.inference_mode():
            features = feature_extractor(roi)

            if attn_save_path is not None:
                A = model(features, attention_only=True)

                if A.size(0) > 1:  # CLAM multi-branch attention
                    A = A[clam_pred]

                A = A.view(-1, 1).cpu().numpy()

                # if ref_scores is not None:
                #     ref_scores = np.array(ref_scores).flatten()
                #     percentiles = np.array([score2percentile(score, ref_scores) for score in A.flatten()])
                #     for score_idx in range(len(A)):
                #         print("A shape:", A.shape)    # [N_patches,1]  each patches attention score
                #         print("A[score_idx,0]shape:", A[score_idx,0].shape)
                #         # A[score_idx] = score2percentile(A[score_idx], ref_scores)
                #         A[score_idx] = score2percentile(A[score_idx].item(), ref_scores)   #Access the scalar value at [score_idx,0]
                # asset_dict = {'attention_scores': A, 'coords': coords}
                # save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)

                if ref_scores is not None:
                    # Convert ref_scores to numpy array if it isn't already
                    ref_scores = np.array(ref_scores).flatten()
                    
                    # Initialize array for percentiles
                    percentiles = np.zeros_like(A)
                    
                    # Iterate through each attention score
                    for i in range(A.shape[0]):
                        print(f"Type of A: {type(A)}")
                        print(f"Shape of A: {A.shape}")
                        print(f"Type of A[i,0]: {type(A[i,0])}")
                        print(f"Type of ref_scores: {type(ref_scores)}")
                        print(f"Shape of ref_scores: {ref_scores.shape}")
                                                

                        # Get scalar value from array
                        score = float(A[i, 0])
                        # Calculate percentile
                        percentiles[i, 0] = score2percentile(score, ref_scores)
                    
                    A = percentiles

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object

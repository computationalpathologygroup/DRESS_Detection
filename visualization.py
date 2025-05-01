import os
import h5py
import openslide
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_topk_patches(
    wsi_path,
    topk_coords,
    patch_size=256,
    coord_level_mag=10,
    wsi_level_mag=10,
    output_path=None,
    level=2
):
    """
    Visualize top-k patch locations on a WSI.

    Args:
        wsi_path (str): Path to the WSI file.
        topk_coords (np.ndarray): Array of (x, y) coordinates.
        patch_size (int): Size of each patch at extraction.
        coord_level_mag (int): Magnification used during feature extraction.
        wsi_level_mag (int): Magnification of the WSI.
        output_path (str, optional): If given, saves the overlay image.
        level (int): Pyramid level of the WSI for visualization.
    """
    slide = openslide.OpenSlide(wsi_path)
    wsi_img = slide.read_region(
        (0, 0), level, slide.level_dimensions[level]).convert("RGB")

    scale_factor = coord_level_mag / wsi_level_mag
    scaled_coords = (topk_coords / scale_factor).astype(int)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wsi_img)

    for (x, y) in scaled_coords:
        rect = patches.Rectangle(
            (x, y),
            patch_size * scale_factor,
            patch_size * scale_factor,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

    ax.set_title('Top-k Patch Locations')
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close(fig)


def zoom_coords(low_mag_coords, patch_size=256, zoom_factor=2):
    """
    Zooms coordinates from lower magnification to higher magnification.

    Args:
        low_mag_coords (np.ndarray): Array of (x, y) coordinates at low mag.
        zoom_factor (float): Zoom scaling factor (e.g., 2 for 10x -> 20x).

    Returns:
        np.ndarray: Zoomed coordinates (x, y).
    """
    return (low_mag_coords * zoom_factor).astype(int)


def extract_and_save_patches(slide_path, coords, output_dir, patch_size=256, level=0):
    """
    Extract and save patches from a WSI.

    Args:
        slide_path (str): Path to WSI file.
        coords (np.ndarray): Coordinates array.
        output_dir (str): Directory to save patches.
        patch_size (int): Patch size.
        level (int): Pyramid level.
    """
    os.makedirs(output_dir, exist_ok=True)
    slide = openslide.OpenSlide(slide_path)

    for idx, (x, y) in enumerate(coords):
        patch = slide.read_region(
            (int(x), int(y)), level, (patch_size, patch_size)).convert('RGB')
        patch.save(os.path.join(output_dir, f"patch_{idx}_x{x}_y{y}.png"))

    print(f"Saved {len(coords)} patches to {output_dir}")


if __name__ == "__main__":
    # Settings
    slide_path = "../Data/DRESS/SR-2cb30640Lbcf2-1702576355563.svs"
    h5_path = "../RESULTS_DIRECTORY/features_UNIv2_20x/h5_files/SR-2cb30640Lbcf2-1702576355563.h5"
    output_dir = "../RESULTS_DIRECTORY/topk/20x"
    patch_size = 256
    level = 0

    # Load coordinates
    with h5py.File(h5_path, "r") as f:
        coords = f['coords'][:]

    # (49152, 376832), (49408, 376832), (49152, 377088), (49408, 377088),
    # (40960, 290816), (41216, 290816), (40960, 291072), (41216, 291072),
    # (49152, 286720), (49408, 286720), (49152, 286976), (49408, 286976),
    # (45056, 286720), (45312, 286720), (45056, 286976), (45312, 286976),
    # (73728, 290816), (73984, 290816), (73728, 291072), (73984, 291072),
    # (86016, 200704), (86272, 200704), (86016, 200960), (86272, 200960),
    # (49152, 290816), (49408, 290816), (49152, 291072), (49408, 291072),
    # (73728, 196608), (73984, 196608), (73728, 196864), (73984, 196864)
    topk_coords = np.array([
        [24576, 188416],
        [20480, 145408],
        [24576, 143360],
        [22528, 143360],
        [36864, 145408],
        [43008, 100352],
        [24576, 145408],
        [36864,  98304],
        [36864, 190464],
        [22528, 141312],
        [28672, 100352],
        [24576, 190464],
        [36864, 188416],
        [36864, 143360],
        [24576,  98304]
    ])

    # Save extracted patches
    extract_and_save_patches(slide_path, topk_coords,
                             output_dir, patch_size, level)

    # Optionally visualize
    visualize_topk_patches(
        wsi_path=slide_path,
        topk_coords=topk_coords,
        patch_size=patch_size,
        coord_level_mag=20,
        wsi_level_mag=20,
        output_path=os.path.join(output_dir, "topk_overlay.png"),
        level=level
    )

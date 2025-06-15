import os
import h5py
import argparse
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
    """Visualize top-k patch locations on a WSI."""
    slide = openslide.OpenSlide(wsi_path)
    wsi_img = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")

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


def extract_and_save_patches(slide_path, coords, output_dir, patch_size=256, level=0):
    """Extract and save patches from a WSI."""
    os.makedirs(output_dir, exist_ok=True)
    slide = openslide.OpenSlide(slide_path)

    for idx, (x, y) in enumerate(coords):
        patch = slide.read_region((int(x), int(y)), level, (patch_size, patch_size)).convert('RGB')
        patch.save(os.path.join(output_dir, f"patch_{idx}_x{x}_y{y}.png"))

    print(f"Saved {len(coords)} patches to {output_dir}")


def load_coords_from_h5(h5_path, topk=None):
    """Load coordinates from an H5 file."""
    with h5py.File(h5_path, "r") as f:
        coords = f['coords'][:]
    return coords if topk is None else coords[:topk]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and visualize top-k patches from a WSI.")
    parser.add_argument("--wsi", type=str, required=True, help="Path to the whole-slide image (.svs)")
    parser.add_argument("--h5", type=str, required=True, help="Path to the H5 file containing patch coordinates")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save patches and overlay")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size in pixels")
    parser.add_argument("--level", type=int, default=0, help="Magnification level to extract patches from")
    parser.add_argument("--coord_mag", type=int, default=20, help="Magnification level of coordinates in H5")
    parser.add_argument("--wsi_mag", type=int, default=20, help="Magnification level to display WSI overlay")
    parser.add_argument("--topk", type=int, default=15, help="Number of top-k coordinates to use")
    args = parser.parse_args()

    coords = load_coords_from_h5(args.h5, args.topk)

    extract_and_save_patches(
        slide_path=args.wsi,
        coords=coords,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        level=args.level
    )

    visualize_topk_patches(
        wsi_path=args.wsi,
        topk_coords=coords,
        patch_size=args.patch_size,
        coord_level_mag=args.coord_mag,
        wsi_level_mag=args.wsi_mag,
        output_path=os.path.join(args.output_dir, "topk_overlay.png"),
        level=args.level
    )

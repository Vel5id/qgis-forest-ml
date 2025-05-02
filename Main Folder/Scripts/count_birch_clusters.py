import numpy as np
from scipy.ndimage import label

def count_birch_clusters(birch_mask, patch_size):
    """
    birch_mask: 2D numpy array of type uint8, where each pixel =1 if it's birch, =0 otherwise.
    patch_size: size of each block (64)
    """
    H, W = birch_mask.shape

    # 1) Create M×N grid of patches
    M = H // patch_size
    N = W // patch_size
    grid = np.zeros((M, N), dtype=np.uint8)

    for r in range(M):
        for c in range(N):
            block = birch_mask[
                r*patch_size:(r+1)*patch_size,
                c*patch_size:(c+1)*patch_size
            ]
            # Should we check majority of pixels ==1, or just one?
            # Here we set 1 if at least one pixel in the block == 1
            if block.max() == 1:
                grid[r, c] = 1

    # 2) Connected components
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]], dtype=int)
    labels, n_clusters = label(grid, structure=structure)

    # 3) Cluster sizes
    sizes = np.bincount(labels.ravel())[1:]  # skip background label 0

    # 4) Separation
    n_single = np.sum(sizes == 1)
    n_multi  = np.sum(sizes > 1)

    return {
        "total_clusters": int(n_clusters),
        "single_patch":   int(n_single),
        "multi_patch":    int(n_multi),
        "sizes":          sizes  # can be returned for detailed analysis
    }

# Example usage:
import rasterio
with rasterio.open("birch_mask.tif") as src:
    birch_mask = src.read(1)

res = count_birch_clusters(birch_mask, patch_size=64)
print(f"Total clusters: {res['total_clusters']}")
print(f"Single-patch clusters: {res['single_patch']}")
print(f"Multi-patch clusters: {res['multi_patch']}")

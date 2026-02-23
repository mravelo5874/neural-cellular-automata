import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------- configuration ----------
input_dir = f'tensor_recordings\\bonzai_rec_Feb.23.2026_11.48.25\\'
output_video = "nca_channels.mp4"
fps = 20
dpi = 100
# ---------------------------------

# custom cmap: -1 -> cyan, +1 -> yellow
# we first map data from [-1,1] to [0,1] and then apply this cmap
cyan_to_yellow = LinearSegmentedColormap.from_list(
    "cyan_yellow",
    [(0.0, 0.0, 1.0),  # cyan-ish: (R,G,B) ~ (0,1,1)
     (1.0, 1.0, 0.0)]  # yellow: (1,1,0)
)

plt.style.use("dark_background")  # black figure/axes by default[web:24]

def make_frame_image(tensor_1_16_x_x):
    """
    tensor_1_16_x_x: np.ndarray with shape [1, 16, X, X]
    returns: HxWx3 uint8 BGR frame image (for OpenCV)
    """
    assert tensor_1_16_x_x.ndim == 4 and tensor_1_16_x_x.shape[1] == 16
    _, C, H, W = tensor_1_16_x_x.shape
    data = tensor_1_16_x_x[0]  # [16, H, W]

    # RGBA visualization from first 4 channels
    # assume values in [0,1]; rescale if needed
    rgba = np.stack(
        [data[0], data[1], data[2]],  # ignore A for display, or use it as mask if you prefer
        axis=-1
    )  # [H, W, 3]
    rgba = np.clip(rgba, 0.0, 1.0)
    rgba_rgb = (rgba * 255).astype(np.uint8)

    fig, axes = plt.subplots(4, 4, figsize=(4, 4), dpi=dpi)

    # force black backgrounds everywhere
    fig.patch.set_facecolor("black")
    fig.patch.set_alpha(1.0)
    for ax in axes.ravel():
        ax.set_facecolor("black")
        ax.patch.set_alpha(1.0)

    for idx in range(16):
        ax = axes[idx // 4, idx % 4]
        ch = data[idx]

        ch_clipped = np.clip(ch, -1.0, 1.0)
        ch_norm = (ch_clipped + 1.0) / 2.0

        im = ax.imshow(
            ch_norm,
            cmap=cyan_to_yellow,
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0
        )
        ax.axis("off")
        ax.set_title(f"C{idx}", fontsize=6, color="white")

    fig.tight_layout(pad=0.1)

    # draw figure to an RGB array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape((h, w, 4))
    plt.close(fig)

    # convert ARGB -> RGB (drop alpha, reorder channels)
    grid_img = buf[:, :, 1:]  # now RGB

    # resize RGBA visualization to match grid height
    gh, gw, _ = grid_img.shape
    rgba_resized = cv2.resize(rgba_rgb, (gh, gh), interpolation=cv2.INTER_NEAREST)

    rh, rw, _ = rgba_resized.shape
    if rh != gh:
        rgba_resized = cv2.resize(rgba_resized, (rw, gh), interpolation=cv2.INTER_NEAREST)

    # concatenate: grid on the left, RGBA composite on the right
    frame_rgb = np.concatenate([grid_img, rgba_resized], axis=1)  # [H, W_total, 3]

    # convert matplotlib RGB -> OpenCV BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr


def main():
    files = sorted(
        glob.glob(os.path.join(input_dir, "*.npy")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )
    if not files:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    # build first frame to know video size
    first_tensor = np.load(files[0])
    first_frame = make_frame_image(first_tensor)
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    iter_count = 0
    try:
        for path in files:
            print(f"[{iter_count}] opening file: {path}")
            iter_count += 1
            tensor = np.load(path)  # [1, 16, X, X]
            frame = make_frame_image(tensor)
            writer.write(frame)
    finally:
        writer.release()

    print(f"Saved video to {output_video}")


if __name__ == "__main__":
    main()
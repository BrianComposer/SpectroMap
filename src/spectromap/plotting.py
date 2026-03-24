from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



def _base_figure(figsize=(12, 6)):
    fig = plt.figure(figsize=figsize, dpi=180, constrained_layout=True)
    return fig


def save_spectrogram_plot(
    output_file: Path,
    freqs: np.ndarray,
    times: np.ndarray,
    spectrogram_db: np.ndarray,
    title: str,
) -> None:
    fig = _base_figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(times, freqs, spectrogram_db, shading="auto", cmap="magma")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Power [dB]")

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_ylim(freqs.min(), freqs.max())
    ax.grid(alpha=0.12)

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def save_constellation_plot(
    output_file: Path,
    freqs: np.ndarray,
    times: np.ndarray,
    spectrogram_db: np.ndarray,
    peak_mask: np.ndarray,
    title: str = "Constellation map",
):
    """
    Plot ONLY constellation points (no spectrogram background).
    """

    # --- Extract peak coordinates ---
    peak_indices = np.where(peak_mask)

    t_vals = times[peak_indices[1]]
    f_vals = freqs[peak_indices[0]]

    # --- Plot ---
    plt.figure(figsize=(10, 6))

    plt.scatter(
        t_vals,
        f_vals,
        s=8,
        alpha=0.8
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(title)

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()


def save_melody_plot(
    output_file: Path,
    freqs: np.ndarray,
    times: np.ndarray,
    spectrogram_db: np.ndarray,
    melody_points: np.ndarray,
    title: str,
) -> None:
    fig = _base_figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    ax.pcolormesh(times, freqs, spectrogram_db, shading="auto", cmap="magma", alpha=0.78)

    if melody_points.size > 0:
        ax.plot(
            melody_points[:, 0],
            440.0 * (2.0 ** ((melody_points[:, 1] - 69.0) / 12.0)),
            linewidth=2.0,
            color="springgreen",
            label="Extracted melody",
        )

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.12)

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)



import numpy as np
import matplotlib.pyplot as plt


def plot_similarity_matrix(df, output_path):
    files = sorted(set(df["file_a"]).union(set(df["file_b"])))
    index_map = {f: i for i, f in enumerate(files)}

    n = len(files)
    matrix = np.zeros((n, n))

    for _, row in df.iterrows():
        i = index_map[row["file_a"]]
        j = index_map[row["file_b"]]
        matrix[i, j] = row["similarity"]
        matrix[j, i] = row["similarity"]

    # --- Enmascarar diagonal ---
    mask = np.eye(n, dtype=bool)
    matrix_masked = np.ma.masked_array(matrix, mask=mask)

    # 🔥 CLAVE: calcular escala SIN la diagonal
    valid_values = matrix[~mask]

    vmin = valid_values.min()
    vmax = valid_values.max()

    # --- Plot ---
    plt.figure(figsize=(10, 8))

    # cmap = plt.cm.viridis.copy()
    # cmap = plt.cm.Greys.copy()
    # cmap = plt.cm.Reds.copy()
    cmap = plt.cm.Purples.copy()
    # cmap = plt.cm.Oranges.copy()
    cmap.set_bad(color="white")  # diagonal invisible

    im = plt.imshow(
        matrix_masked,
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, label="Similarity")

    plt.xticks(range(n), files, rotation=45, ha="right")
    plt.yticks(range(n), files)

    plt.title("Similarity Matrix (no self-comparison)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
"""
TRIBE v2 brain visualization script.

Loads predictions produced by run_tribev2.py and renders them onto the
fsaverage5 cortical surface using nilearn.

Usage:
    # Re-run inference and visualize immediately
    python visualize_tribev2.py

    # Visualize previously saved predictions
    python visualize_tribev2.py --preds cache/predictions.npy

    # Save an animated GIF instead of showing interactively
    python visualize_tribev2.py --save brain.gif

    # Plot a specific set of views (left, right, dorsal, ventral, medial_left, ...)
    python visualize_tribev2.py --views left right dorsal

    # Show stimulus text/audio overlay alongside brain (requires segments)
    python visualize_tribev2.py --stimuli
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

CACHE_DIR = Path("./cache")

AVAILABLE_VIEWS = [
    "left", "right", "dorsal", "ventral",
    "medial_left", "medial_right",
    "anterior", "posterior",
]


def parse_args():
    p = argparse.ArgumentParser(description="TRIBE v2 brain visualization")
    p.add_argument("--preds", type=Path, default=None,
                   help="Path to .npy predictions file. If omitted, re-runs inference.")
    p.add_argument("--input", type=Path, default=None,
                   help="Input file for inference: .txt, .mp3/.wav, or .mp4. "
                        "Ignored if --preds is given.")
    p.add_argument("--save", type=Path, default=None,
                   help="Save figure to this path (.png or .gif) instead of "
                        "displaying interactively. Use .gif for animated output.")
    p.add_argument("--views", nargs="+", default=["left", "right"],
                   choices=AVAILABLE_VIEWS,
                   help="Surface views to render. Default: left right.")
    p.add_argument("--stimuli", action="store_true",
                   help="Show stimulus text/audio strip above the brain plots "
                        "(requires --preds to be omitted so segments are available).")
    p.add_argument("--timesteps", type=int, default=None,
                   help="How many TR timesteps to plot. Defaults to all.")
    p.add_argument("--llama", action="store_true",
                   help="Use real LLaMA 3.2-3B encoder when re-running inference.")
    return p.parse_args()


def run_inference(use_llama: bool, input_path: Path | None = None):
    """Run the full pipeline and return (preds, segments)."""
    from tribev2.demo_utils import TribeModel

    config_update = {
        "data.text_feature.model_name": "microsoft/Phi-3.5-mini-instruct" if not use_llama else "meta-llama/Llama-3.2-3B",
        "data.text_feature.device": "cpu",
        "data.audio_feature.device": "cpu",
        "data.video_feature.image.device": "cpu",
        "data.video_feature.image.batch_size": 1,
        "data.video_feature.num_frames": 16,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        config_update=config_update,
    )

    if input_path is None:
        text = (
            "To be or not to be, that is the question. "
            "Whether tis nobler in the mind to suffer "
            "the slings and arrows of outrageous fortune."
        )
        input_path = CACHE_DIR / "demo_shakespeare.txt"
        input_path.write_text(text)

    suffix = input_path.suffix.lower()
    print("Building events dataframe...")
    if suffix == ".txt":
        df = model.get_events_dataframe(text_path=input_path)
    elif suffix in {".mp3", ".wav", ".flac", ".ogg"}:
        df = model.get_events_dataframe(audio_path=input_path)
    elif suffix in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        df = model.get_events_dataframe(video_path=input_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    print(f"Events: {len(df)} rows\n")

    print("Running predict()...")
    preds, segments = model.predict(events=df)
    np.save(CACHE_DIR / "predictions.npy", preds)
    return preds, segments


def make_plotter():
    # Use nilearn backend — works headless and doesn't need a display server.
    from tribev2.plotting.cortical import PlotBrainNilearn
    return PlotBrainNilearn(mesh="fsaverage5")


def plot_mean_activation(plotter, preds: np.ndarray, views: list[str],
                         save: Path | None):
    """Single-frame plot: mean absolute activation across all timesteps."""
    mean_act = np.abs(preds).mean(axis=0)
    print(f"Plotting mean activation across {len(preds)} timesteps, views={views}...")

    fig, axarr = plotter.get_fig_axes(views=views)
    plotter.plot_surf(
        signals=mean_act,
        axes=axarr,
        views=views,
        cmap="hot",
        norm_percentile=99,
        colorbar=True,
        colorbar_title="Mean |activation|",
    )
    fig.suptitle("TRIBE v2 — mean cortical activation", fontsize=10, y=1.02)
    _save_or_show(fig, save, suffix="_mean")


def plot_timesteps(plotter, preds: np.ndarray, segments, views: list[str],
                   n_timesteps: int | None, show_stimuli: bool, save: Path | None):
    """Multi-panel plot: one column per TR timestep."""
    if n_timesteps is not None:
        preds = preds[:n_timesteps]
        if segments is not None:
            segments = segments[:n_timesteps]

    print(f"Plotting {len(preds)} timestep(s), views={views}...")

    # plot_timesteps returns the figure
    view_arg = views[0] if len(views) == 1 else views[0]  # base view
    fig = plotter.plot_timesteps(
        preds,
        segments=segments,
        views=view_arg,
        cmap="fire",
        norm_percentile=99,
        vmin=0.6,
        alpha_cmap=(0, 0.2),
        show_stimuli=show_stimuli and segments is not None,
    )
    _save_or_show(fig, save, suffix="_timesteps")


def _save_or_show(fig, save: Path | None, suffix: str = ""):
    if save is None:
        plt.show()
        return

    out = save.with_stem(save.stem + suffix) if save.suffix == ".gif" else save
    out = out.with_suffix(save.suffix if save.suffix else ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def main():
    args = parse_args()

    segments = None

    if args.preds is not None:
        print(f"Loading predictions from {args.preds}...")
        preds = np.load(args.preds)
    else:
        preds, segments = run_inference(use_llama=args.llama, input_path=args.input)

    print(f"Predictions shape: {preds.shape}  (n_timesteps={preds.shape[0]}, n_vertices={preds.shape[1]})")
    print(f"Value range: [{preds.min():.4f}, {preds.max():.4f}]\n")

    plotter = make_plotter()

    # Always render the mean activation map (single clear image)
    plot_mean_activation(plotter, preds, views=args.views, save=args.save)

    # Also render the per-timestep grid if we have more than one TR
    if preds.shape[0] > 1:
        plot_timesteps(
            plotter, preds, segments,
            views=args.views,
            n_timesteps=args.timesteps,
            show_stimuli=args.stimuli,
            save=args.save,
        )


if __name__ == "__main__":
    # Use non-interactive backend when saving to file
    import sys
    if "--save" in sys.argv:
        matplotlib.use("Agg")
    main()

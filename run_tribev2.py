"""
TRIBE v2 local test script.

Usage:
    python run_tribev2.py                        # text mode (Shakespeare demo)
    python run_tribev2.py --input path/to/file   # .txt, .mp3/.wav, or .mp4
    python run_tribev2.py --llama                # use real LLaMA 3.2-3B (requires HF access)

By default, uses microsoft/Phi-3.5-mini-instruct as the text encoder substitute
(same hidden dim as LLaMA, open access, but predictions are not neuroscientifically
meaningful).  Pass --llama once you have approved access to meta-llama/Llama-3.2-3B.
"""

import argparse
from pathlib import Path

import numpy as np

DEMO_TEXT = """\
To be or not to be, that is the question.
Whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune,
or to take arms against a sea of troubles.
"""

CACHE_DIR = Path("./cache")


def parse_args():
    p = argparse.ArgumentParser(description="TRIBE v2 inference test")
    p.add_argument("--input", type=Path, default=None,
                   help="Path to .txt, .mp3/.wav, or .mp4 file. "
                        "Defaults to a short Shakespeare demo text.")
    p.add_argument("--phi", action="store_true",
                   help="Use microsoft/Phi-3.5-mini-instruct as a text encoder substitute "
                        "(pipeline-valid but not neuroscientifically meaningful).")
    return p.parse_args()


def load_model(use_phi: bool):
    from tribev2.demo_utils import TribeModel

    # Offload text + audio extractors to CPU so V-JEPA2 ViT-G fits in 16 GB VRAM.
    # Features are cached after the first run so this slowdown is one-time only.
    config_update = {
        "data.text_feature.model_name": "microsoft/Phi-3.5-mini-instruct" if use_phi else "meta-llama/Llama-3.2-3B",
        "data.text_feature.device": "cpu",
        "data.audio_feature.device": "cpu",
        "data.video_feature.image.batch_size": 1,   # process 1 clip at a time
        "data.video_feature.num_frames": 16,         # 16 frames instead of 64
    }

    if use_phi:
        print("[warn] Using Phi-3.5-mini-instruct as text encoder substitute.")
        print("       Predictions are pipeline-valid but NOT neuroscientifically meaningful.\n")

    print("Loading TRIBE v2 checkpoint from HuggingFace...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        config_update=config_update,
    )
    print("Checkpoint loaded.\n")
    return model


def build_events(model, input_path: Path | None):
    suffix = input_path.suffix.lower() if input_path else None

    if input_path is None:
        text_file = CACHE_DIR / "demo_shakespeare.txt"
        text_file.write_text(DEMO_TEXT)
        print(f"No input given — using built-in Shakespeare demo text ({text_file})")
        return model.get_events_dataframe(text_path=text_file)

    if suffix == ".txt":
        print(f"Input: text file  → {input_path}")
        return model.get_events_dataframe(text_path=input_path)

    if suffix in {".mp3", ".wav", ".flac", ".ogg"}:
        print(f"Input: audio file → {input_path}")
        return model.get_events_dataframe(audio_path=input_path)

    if suffix in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        print(f"Input: video file → {input_path}")
        return model.get_events_dataframe(video_path=input_path)

    raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .mp3/.wav, or .mp4.")


def main():
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(use_phi=args.phi)

    print("Building events dataframe (TTS + transcription on first run)...")
    df = build_events(model, args.input)
    print(f"Events: {len(df)} rows")
    show_cols = [c for c in ["type", "start", "duration", "text"] if c in df.columns]
    print(df[show_cols].head(10).to_string())
    print()

    print("Running brain encoder (feature extraction + TRIBE v2 forward pass)...")
    preds, segments = model.predict(events=df)

    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Predictions shape : {preds.shape}  (n_timesteps, n_vertices)")
    print(f"Value range       : [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"Kept segments     : {len(segments)} / {len(df)} events")
    print()
    print("Top-5 most active vertices (mean absolute activation):")
    vertex_activity = np.abs(preds).mean(axis=0)
    top5 = np.argsort(vertex_activity)[-5:][::-1]
    for v in top5:
        print(f"  vertex {v:>6d}  mean |activation| = {vertex_activity[v]:.4f}")

    out_path = CACHE_DIR / "predictions.npy"
    np.save(out_path, preds)
    print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    main()

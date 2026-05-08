# Egocentric Brain Encoding with TRIBE v2

Local inference and visualization scripts for [TRIBE v2](https://huggingface.co/facebook/tribev2), a multimodal brain encoder that predicts cortical fMRI responses to text, audio, and video stimuli.

## Files

| File | Purpose |
|------|---------|
| `run_tribev2.py` | Run TRIBE v2 inference on a stimulus file; saves predictions to `cache/predictions.npy` |
| `visualize_tribev2.py` | Load predictions and render them onto the fsaverage5 cortical surface |
| `m2-res_854p.mp4` | Example video stimulus |
| `m2-res_854p.wav` | Audio extracted from the example video |
| `m2-res_854p.tsv` | Transcription/event timing for the example stimulus |
| `cache/` | Auto-generated directory; stores model weights, intermediate features, and `predictions.npy` |

## Requirements

- Python 3.10+
- `tribev2` package (from `facebook/tribev2` on HuggingFace)
- `nilearn` (for cortical surface visualization)
- `matplotlib`, `numpy`
- ~16 GB VRAM (V-JEPA2 ViT-G); text and audio extractors are offloaded to CPU automatically

### Text encoder

By default the scripts use `microsoft/Phi-3.5-mini-instruct` as a drop-in text encoder substitute (open access, same hidden dimension as LLaMA). Predictions produced this way are pipeline-valid but **not neuroscientifically meaningful**. To get meaningful predictions, pass `--llama` after requesting access to `meta-llama/Llama-3.2-3B` on HuggingFace.

## Usage

### 1. Run inference

```bash
# Built-in Shakespeare demo (no input file needed)
python run_tribev2.py

# Text file
python run_tribev2.py --input path/to/stimulus.txt

# Audio file (.mp3, .wav, .flac, .ogg)
python run_tribev2.py --input path/to/audio.wav

# Video file (.mp4, .avi, .mkv, .mov, .webm)
python run_tribev2.py --input m2-res_854p.mp4

# Use the real LLaMA 3.2-3B encoder (requires HF gated-model access)
python run_tribev2.py --input stimulus.txt --llama
```

Predictions are saved to `cache/predictions.npy` as an array of shape `(n_timesteps, n_vertices)`.

### 2. Visualize predictions

```bash
# Re-run inference and show brain interactively
python visualize_tribev2.py

# Load previously saved predictions
python visualize_tribev2.py --preds cache/predictions.npy

# Save to PNG instead of displaying
python visualize_tribev2.py --preds cache/predictions.npy --save brain.png

# Save animated GIF
python visualize_tribev2.py --save brain.gif

# Choose surface views (default: left right)
python visualize_tribev2.py --views left right dorsal ventral medial_left medial_right

# Show stimulus text/audio overlay (only when re-running inference)
python visualize_tribev2.py --stimuli

# Limit number of TR timesteps plotted
python visualize_tribev2.py --timesteps 20
```

Available views: `left`, `right`, `dorsal`, `ventral`, `medial_left`, `medial_right`, `anterior`, `posterior`.

## Caching

Model weights and extracted features are cached in `./cache/` after the first run. Subsequent runs skip the slow feature-extraction step and load from disk.


## Dataset

We are using BuildAI's open-sourced Egocentric dataset: https://huggingface.co/datasets/builddotai/Egocentric-100K/tree/main
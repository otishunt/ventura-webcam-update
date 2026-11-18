# Models

What each model name (M0, M1, M1 aug, M2, …) and run name means, where the code lives,
and how far each one has got. The design rationale is in `README.md` (§3 Models, §6 Experiments);
the commands are in `docs/RUNBOOK.md`.

Every model outputs per-frame class probabilities over the memes in `configs/pipeline.yaml`
`classes` plus `no_meme`. All of them go through the same **hysteresis smoother**
(`memepose/smoothing.py`, `smoother:` in the config): a meme is shown only after it is the
confident top class for `on_frames` frames in a row (0.8 for 14 frames, about 0.47 s) and hidden
after confidence stays below 0.6 for 3 frames. All of them are scored the same way: person-level
splits only (`--split lopo | newcond | newperson | loso`), with frame macro-F1, clip macro-F1,
false triggers per minute on free-behavior clips, and delay until a meme is shown.

## Overview

| name | input | what it is | code | runs on | run names |
|---|---|---|---|---|---|
| **M0** | landmarks, per frame | hand-written rules, thresholds fitted per fold | `memepose/models/rules.py` | Mac | `m0_lopo`, `m0_lopo_v2` |
| **M1 noaug** | 1 s of landmark features | GRU, real data only | `memepose/models/temporal.py` | Mac | `m1_lopo_noaug` |
| **M1 aug** | 1 s of landmark features | same GRU, plus classic feature-space augmentation | `memepose/models/temporal.py` | Mac | `m1_lopo_aug`, `abl_*_s{0,1,2}` |
| **M2** | face and upper-body crop, per frame | pretrained image backbone, fine-tuned | `memepose/models/frame.py` | Delta GPU | `m2_<backbone>_<split>_s<seed>` |
| **M2 syn / ctl** | crops + synthetic edits | M2 with accepted synthetic frames, or the matched control | `memepose/models/frame.py` | Delta GPU | `m2_..._syn_*`, `m2_..._ctl_*` (not run yet) |
| M3 | landmarks + crops | M1 and M2 combined (stretch goal) | — | — | not built |

## M0: rules

The floor every learned model has to beat. Each meme has one interpretable score built from
MediaPipe blendshapes and keypoints: fist closure, wrist height, finger extension, eyelid
closure, head roll, mouth shape and so on (`rule_scores()` in `rules.py`). On each training fold,
every class gets the threshold that maximizes its one-vs-rest F1. At test time the meme with the
largest normalized margin above its threshold wins. If no meme clears its threshold, the frame is
`no_meme`. There is no training beyond picking those thresholds.

`m0_lopo_v2` is the same idea with revised rules. Its frame macro-F1 is about the same as v1's
(0.32 vs 0.35), but it triggers far more often on free behavior (33 vs 12 false triggers/min).

## M1: landmark temporal model

- **Input:** windows of 30 frames (about 1 s at 30 fps) of about 280 normalized landmark
  features: face blendshapes, pose and both hands, independent of camera distance and framing.
  Each window predicts the class of its **last** frame, so the model is causal and can run live.
- **Architecture:** a linear input projection, then a 2-layer one-directional GRU (hidden size
  128, dropout 0.3). The last time step and the mean over the window are concatenated and passed
  to an MLP head (`GRUNet`). A dilated causal TCN (`TCNNet`) is the alternative: `m1.arch: tcn`.
- **Training:** AdamW for up to 40 epochs, early stopping on a validation set made of 15% of the
  training clips (grouped by clip), class weights `(1/freq)^0.5`.

### M1 noaug vs M1 aug

They differ only in `m1.augment`:

- **M1 noaug** (`--set m1.augment=none`): real windows only.
- **M1 aug** (`m1.augment: classic`, the default): feature-space augmentation applied to each
  training window:
  - `noise_std 0.03`: Gaussian noise, in standardized units
  - `time_scale [0.8, 1.25]`: speed jitter, a stand-in for frame-rate differences
  - `feature_dropout_p 0.1`: zero a whole hand or face block, as when MediaPipe loses a hand
  - `flip_p`: mirror flip as a fixed feature permutation. **Now 0.0**, because of the ablation
    below.

`scripts/run_mac.sh` trains both of them, plus M0, on every Mac run.

### M1 augmentation ablation (`abl_<variant>_s<seed>`)

`python -m memepose.ablate` runs LOPO for each variant with seeds 0, 1 and 2, all with flip on
as the starting point (`full`), then removes one augmentation at a time. Results in
`runs/ablation_summary.md`:

| variant | what changes | frame macro-F1 (mean ± sd over 3 seeds) |
|---|---|---|
| `none` | no augmentation | 0.719 ± 0.006 |
| `full` | all four, flip included | 0.664 ± 0.031 |
| `no_flip` | full minus flip | **0.739 ± 0.019** |
| `no_noise` | full minus noise | 0.711 ± 0.029 |
| `no_timescale` | full minus time-scale | 0.678 ± 0.041 |
| `no_dropout` | full minus feature dropout | 0.649 ± 0.011 |

The flip in feature space was what hurt. It ruins one-sided memes, for example
`shrek_side_eye` F1 fell from 0.77 to 0.38. That is why `flip_p` is now 0, so the current
"M1 aug" is effectively `no_flip`. With flip off, the other three augmentations help a little
(0.739 vs 0.719 for `none`).

## M2: frame model on crops

- **Input:** one 224 × 224 face and upper-body crop per frame. Crops are exported on the Mac
  (`memepose.generate.frames`) from every 3rd labelled frame and copied to Delta. They stay
  in memory at 256 px.
- **Backbones** (`m2.backbone`):
  - `dinov2_s`: `facebook/dinov2-small`, ViT-S/14, 22M parameters, self-supervised. The main
    model. Features are the CLS token concatenated with the mean of the patch tokens.
  - `efficientnet_b0`: `google/efficientnet-b0`, a 5M-parameter CNN, the speed baseline.
  - `tiny`: a small randomly initialized CNN, only for tests and smoke runs.
  - The head is LayerNorm → dropout → linear.
- **Training:** 10 epochs, 1 warm-up epoch, separate learning rates for the backbone (2e-5) and
  the head (1e-3), weight decay 0.05, label smoothing 0.1, early stopping (patience 3) on 15% of
  training clips, class weights `(1/freq)^0.5`.
- **Augmentation** (on the GPU): random scale and shift of the crop window, a horizontal flip
  (`flip_p 0.5`; a real pixel mirror, unlike M1's feature permutation), color jitter and a little
  grayscale.
- **Evaluation:** each crop's prediction is held forward to the following frames (causal: frame
  *t* uses the latest crop at or before *t*), so M2 goes through the same smoother and clip
  metrics as M0 and M1.
- **Run on Delta:** `sbatch slurm/m2.slurm` (DINOv2, LOPO, seeds 0 1 2), or
  `--export=ALL,BACKBONE=efficientnet_b0`. Run names are `m2_<backbone>_<split>_s<seed>`, plus a
  `TAG` when one is set.

### M2 experiment arms

All arms run with `m2.equal_steps: true`: steps per epoch = real training crops / batch size,
so adding synthetic data changes *what* the model sees, not *how much* training it gets.

| arm | how to submit | what it tests |
|---|---|---|
| real + classic aug | `sbatch slurm/m2.slurm` | baseline (done) |
| **syn** | `TAG=syn`, `--synthetic-images .../synthetic/main/synthetic_frames.csv` | does adding accepted synthetic edits help? |
| **ctl** | `TAG=ctl`, `--synthetic-control --synthetic-images ...` | control arm: the same source frames as `syn`, unedited, so any gain from `syn` can't be explained by simply having more copies of those frames |
| noflip | `TAG=noflip`, `--set m2.aug.flip_p=0` | does the pixel flip hurt one-sided memes the way M1's flip did? |
| newcond | `SPLIT=newcond` | held-out sessions in new rooms, lighting and outfits (needs `split.heldout_sessions` set) |

A fold never trains on the test person's real frames **or their synthetic variants**: synthetic
frames follow their source clip.

## Results so far (LOPO)

| model | run(s) | frame macro-F1 | clip macro-F1 | false triggers/min | notes |
|---|---|---|---|---|---|
| M0 rules | `m0_lopo` | 0.350 ± 0.072 | 0.330 ± 0.122 | 12.4 | 3 folds |
| M0 rules v2 | `m0_lopo_v2` | 0.319 ± 0.061 | 0.330 ± 0.082 | 32.5 | 4 folds |
| M1 noaug | `m1_lopo_noaug` | 0.708 ± 0.028 | 0.725 ± 0.023 | 1.6 | 1 seed, 3 folds |
| M1 aug (no flip) | `m1_lopo_aug` | 0.715 ± 0.021 | 0.756 ± 0.027 | 2.2 | 1 seed, 3 folds |
| M1 aug (no flip), ablation | `abl_no_flip_s*` | 0.739 ± 0.019 | 0.757 ± 0.025 | 3.8 | 3 seeds, sd across seeds |
| **M2 DINOv2 ViT-S** | `m2_dinov2_s_lopo_s*` | **0.772 ± 0.008** | 0.703 ± 0.024 | 1.1 | 3 seeds, sd across seeds |
| M2 EfficientNet-B0 | `m2_efficientnet_b0_lopo_s*` | 0.519 ± 0.019 | 0.114 ± 0.012 | 0.0 | 3 seeds; almost never fires (6% of meme clips detected) |

Sources: `runs/comparison.md`, `runs/ablation_summary.md` (Mac) and
`/work/hdd/bjeg/meme-pose/runs/comparison.md` (Delta). These numbers aren't fully comparable.
The fold counts differ: folds whose test person has fewer than `split.min_test_clips` clips are
left out of the mean. Some single-seed rows report sd across folds, others sd across seeds. And
the smoother was re-tuned on 2026-11-15, so the false trigger rates of older runs used different
smoother settings.

Takeaways so far:
- DINOv2 M2 has the best frame F1 and very few false triggers.
- DINOv2 M2's clip F1 (0.70) is below M1's, because it detects only about 63% of meme clips: it
  is conservative.
- EfficientNet-B0 at these settings mostly predicts `no_meme`.
- The synthetic arms (syn and ctl) are waiting on bulk generation.

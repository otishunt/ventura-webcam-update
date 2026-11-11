# Handoff: Meme Pose Recognition with Synthetic Domain Randomization

DATA 4620 final project · Team: Otis, Carson, Minh, Kevin · Last updated 2026-11-10 (decisions recorded, pipeline repo started)

Paste or attach this file at the start of a new chat. It covers the project
goal, what already exists, the constraints, and a bottom-up system design to
build next. Items marked **[DECIDE]** are open decisions for the team.

---

## 1. Project in one paragraph

A live webcam system that recognizes which meme a person is imitating (12
memes plus "no meme") and replaces the camera feed with that meme's image. The
research angle: training data can only come from the 4 team members, so we
use generative models to make **synthetic, appearance-randomized versions**
of our own clips. The synthetic clips vary age, gender, skin tone, hair,
clothing, lighting, background and camera. The goal is a model that works on
people who look nothing like us. The core experiment is whether
generative domain randomization beats real data alone and beats classic
augmentation.

## 2. Hard constraints

- **Data:** only the 4 team members can be recorded. No outside participants.
  Consent is covered because the subjects are the researchers themselves.
- **Compute:** NCSA Delta through the professor's ACCESS allocation. It uses Slurm, has
  A40, A100 and H200 GPU nodes, x86_64 CPUs and no Docker (conda/venv instead). Data
  lives on `/work/hdd/<proj>`; move it with Globus. Generation and training run there.
  The live demo runs on an M4 MacBook.
- **Deadline:** about 5 weeks from 2026-11-10 (mid-December). The proposal is done.
- **Run everything locally:** use open-weight models on our own GPUs. Our
  consent promised videos are not sent to cloud services, so no hosted
  LLM, VLM or video APIs on participant frames.
- **Course rules:**
  - The professor requires real data with known provenance, so evaluation
    must be on real recordings. The professor **approved** synthetic augmentation of
    our own real data.
  - We must write and evaluate a good amount of our own code.
  - We must show at least one prototype → evaluate → improve cycle.
  - The proposal (3+ pages, prose, with a system diagram) needs these sections:
    Introduction, Background, Data and Methods, Evaluation, Expected Outcomes,
    and Ethical Implications.

## 3. What already exists (recorder app)

Location on Otis's Mac: `~/Desktop/DATA 4620 Final Project/meme-recorder/`

- **Stack:** FastAPI backend plus a plain HTML/JS frontend. Clips are
  converted to 30 fps H.264 MP4 with ffmpeg. An automatic quality check uses
  OpenCV and **must use OpenCV 4**: OpenCV 5 removed `CascadeClassifier`, and
  `requirements.txt` pins `<5`.
- **Run (local only):** double-click `meme-recorder/run_local.command`, or run
  `./run_local.command`. It sets up `.venv` if needed, creates an admin token in
  `.admin_token`, serves on 127.0.0.1 only and opens `http://localhost:8000`. No tunnel
  is used anymore; the Cloudflare and Tailscale routes are retired.
- **Flow:** Start → consent (consent version 1.4) → camera setup → randomized
  prompts → review and redo → done. Participant IDs exist only behind the scenes.
- **Protocol** (`config/memes.yaml`, version `2026.3`, 46 prompts per session):
  - 12 memes × 3 repeats
  - 9 "no_meme" negatives × 1
  - one 60-second free-behavior clip

  The app rereads this file on every request, so edits take effect without a restart.
- **Clip timing:** about 1.0 s neutral face → "NOW" cue → hold until about
  3.2 s → "RELAX" → back to neutral until 4.5 s. Cue times are saved in
  milliseconds as `cue_on_ms` and `cue_off_ms`. These are weak labels, since
  the actual pose lags the cue by human reaction time.
- **Mirroring:** the preview is mirrored like a selfie camera, but saved video is
  **not**. The current memes don't depend on direction, so horizontal-flip augmentation is safe.
- **Data layout:**
  ```
  data/consent/consent_log.jsonl        # names live only here
  data/participants/P###/participant.json   # sessions, recording conditions, person_id
  data/participants/P###/clips/<clip_id>.{raw.webm,mp4,json}
  ```
  `python tools/build_manifest.py` writes `data/manifest.csv`, one row per
  usable clip. Columns include: clip_id, person_id, participant_id,
  session_id, prompt_id, kind, label, looks_like, status, protocol_version,
  mp4_path, cue_on_ms, cue_off_ms, fps, qc metrics, setting, lighting,
  glasses.
  **Split data by `person_id`**: repeat sessions by the same person get
  separate participant IDs but share one person_id.
- **Other tools:** `tools/review.py` (list flagged clips, reject or restore)
  and `tools/withdraw.py`.
- **Test data:** old test data is in `data_test/`, which can be deleted.

### Label set (13 classes)

| id | meme | body region that matters |
|---|---|---|
| success_fist | Success Kid (fist, pursed lips) | face + hand |
| arms_raised | Breaking free (arms up, fists) | upper body |
| chill_guy | Chill Guy (relaxed, smirk) | face + posture (hard: close to neutral) |
| point_at_self | "Who, me?" | face + hand |
| sleepy_squint | Lightskin stare dog (half-closed eyes) | face (hard pair with holding_laugh) |
| pinch_grin | Reddit mod (pinched fingers, sly grin) | face + hand |
| timeout | Shaq timeout (T hands) | hands |
| hands_on_head | Shocked, hands on head | face + hands |
| shrek_side_eye | Shrek side-eye, raised brow | face (hard pair with confused_tilt) |
| holding_laugh | Speed trying not to laugh | face |
| confused_tilt | Swaggy P "???" | face + head |
| point_sideways | Pointing Wojak | arm + face |
| no_meme | 9 tricky everyday movements + 60 s free clip | — |

Each negative's `looks_like` field records which memes it resembles. Use it to
measure false triggers per meme pair.

## 4. Core design principle

**Never let regeneration change the label.** Each synthetic clip must keep
the source clip's geometry and timing: eye direction, finger positions, arm
height, and pose start and end. So generation is conditioned on *structure
extracted from the real clip*, and text is used only for the attributes being
changed.

We decided **against** an "LLM describes the video → generate video from the
text" pipeline as the main path. A text bottleneck loses exactly the
fine-grained geometry and timing that define the labels. It can still appear
as an **ablation**, to show why direct structural conditioning works better.
We also prefer modern controllable **diffusion** video models and face reenactment
over GANs. GANs handle textures and styles but are weak at identity and
attribute editing in stable, flicker-free video.

## 5. Bottom-up system design

### L0 — Capture: maximize variation within each person
With only 4 people, the variation has to come from each person's recordings.
Each member records as many sessions as turn out to be needed (status 2026-11-10: one
session from one person), changing as much as possible between sessions:
- rooms and backgrounds
- lighting: window, overhead, lamp, dim
- cameras: laptop, external webcam, phone
- clothing, hair, glasses or none, hats
- distance from the camera and seated vs standing

Log these details in the app's setup fields.
**Reserve a held-out test session per person** (section L7), recorded in
conditions never used for training or generation.

Rough yield: 4 people × 6 sessions × 46 clips ≈ 1,100 real clips (about 860
meme clips, about 72 per meme).

### L1 — Preprocess
- Decode the MP4s. For every frame, run MediaPipe face landmarks (478 points
  plus 52 blendshape scores), pose (33 points) and hands (21 points per hand).
  Save one compressed `.npz` file per clip, keyed by `clip_id`.
- Build tighter labels: shift `cue_on_ms` and `cue_off_ms` by an estimated
  reaction delay of about 300–500 ms, or detect pose onset from landmark
  motion energy. Save per-frame labels.
- Make crops: a face crop and an upper-body crop per frame, for the
  pixel-based model.

### L2 — Structural "encoder": what is kept from each real clip
For every frame: face landmarks, body pose skeleton, hand keypoints, and
optionally depth, edge or segmentation maps. This is what generation is
conditioned on, and it carries the label.

Optional: a **local** vision-language model writes a caption per clip for
metadata or ablations only. It is never the conditioning signal.

### L3 — Generation (domain randomization)
Three tiers, cheapest first:
1. **Classic augmentation (baseline, required):**
   - color jitter, gamma, blur, noise and JPEG artifacts
   - random crop and scale
   - horizontal flip
   - background replacement using person segmentation
   - frame-rate jitter
2. **Face-region identity editing:** take a real clip and change apparent
   age, gender, skin tone and hair, keeping the expression and motion. Use
   face reenactment (drive a different face with our landmarks) or
   identity-editing video diffusion. This covers face-only memes well:
   sleepy_squint, holding_laugh, shrek_side_eye and confused_tilt.
3. **Full-frame structure-conditioned video diffusion:** generate the whole
   person and scene from the pose, hand and face structure, plus a text prompt
   describing the attributes. This is needed for body and hand memes.
   Expect hand failures, which L4 filtering handles.

**Attribute sampling:** a balanced grid, not "whatever the model defaults to".
- Use neutral, specific prompts (age range, perceived gender presentation,
  skin tone on the Monk scale, hair, clothing, lighting, background, camera
  quality).
- Fix and log the random seeds.
- Store every synthetic clip's attributes and source clip in a
  `synthetic_manifest.csv`.

**Model choice (open):** use whichever open-weight controllable video
diffusion and face-reenactment models are current and run on our GPUs.
Starting points:
- *Video Editing Meets Diffusion Models: A Comprehensive Survey* (TechRxiv,
  Feb 2026)
- the GitHub list "Awesome-Controllable-Video-Diffusion"
- IP-FaceDiff, DiffFERV and MagicMirror, as examples of identity-preserving
  face video editing

Pick based on:
- pose, landmark and hand conditioning support
- a license that allows research use
- how much GPU memory it needs
- seconds of compute per 4.5-second clip

Prototype on 20 clips before scaling.

**Generator fine-tuning (decided: yes).** The team wants to fine-tune a generator on our
clips. Under leave-one-person-out this must be done **per fold**: fine-tune only on the 3
training people, then generate that fold's synthetic data. A generator fine-tuned on all
4 people has seen the test person, which leaks. Budget 4 fine-tunes, plus 1 for the
new-conditions test if that generator excludes the held-out sessions.

### L4 — Filter and check quality (synthetic clips)
Reject a synthetic clip if any of these checks fail:
- **Label kept:** rerun MediaPipe on the synthetic clip. Compare its
  normalized landmarks with the source clip's, frame by frame (face,
  pose and hands separately). Reject above a threshold. Hands must stay
  detected in hand memes.
- **Identity stays stable over time:** a face-recognition embedding should
  barely change between frames, so there's no identity flicker.
- **Attribute actually changed:** the synthetic face should look clearly
  different from the source person, measured by embedding distance.
- **Human check:** hand-review a random sample of about 5% per meme and per
  attribute. Report rejection rates by meme and by attribute; these are
  results in their own right.

### L5 — Models (increasing capacity)
- **M0 Rules:** thresholds on MediaPipe blendshapes and keypoints. This is the
  floor every other model has to beat.
- **M1 Landmark model:** a temporal model over per-frame landmark features,
  about 1-second windows, using a GRU, a temporal CNN or a small transformer.
  Since landmarks already strip out appearance, synthetic data is
  expected to help M1 **only a little**. Predicting this and confirming it is a finding.
- **M2 Pixel model:** a video classifier on face and upper-body crops.
  Either a 2D image model with a temporal head, or a fine-tuned pretrained
  video model. **This is where appearance randomization should matter most.**
- **M3 (stretch):** combine M1 and M2.
- **Rejecting non-memes and steadying output (all models):**
  - include the no_meme class
  - add a confidence threshold
  - require the same prediction for N consecutive frames before showing a
    meme, and keep it until confidence drops (hysteresis)

### L6 — Training experiments
| | real only | + classic aug | + synthetic | synthetic only |
|---|---|---|---|---|
| M1 | ✓ | ✓ | ✓ | (optional) |
| M2 | ✓ | ✓ | ✓ | ✓ |

Also run:
- a synthetic-to-real ratio sweep: 0, 0.5×, 1×, 2× and 4× synthetic data
  relative to real
- an ablation of tier 2 (face edits) vs tier 3 (full-frame generation)
- optionally, the LLM-text-bottleneck ablation

### L7 — Evaluation (the hardest part with 4 people)
Never train on a test person's synthetic variants.
1. **Leave one person out (main metric):** 4 folds. Synthetic data for each
   fold is generated **only from the training people's clips**.
2. **New-conditions test:** each person's reserved session in an unseen
   room, lighting and outfit, never used for training or generation.
3. **Synthetic stress test:** test clips made by a **different generator or
   different seeds and attributes** than the training data. Report accuracy
   per attribute (age, skin tone, gender presentation). This partly
   reuses the same kind of data the model trained on, so present it as a
   stress test, **not** evidence of fairness for real people.
4. ~~Optional real-world check on public datasets~~: **dropped** (the classes are too
   specific to match public datasets).

**Metrics:**
- macro-F1 and per-class F1, plus confusion matrices (watch the known hard pairs)
- false triggers per minute on the free-behavior clips
- delay from striking a pose to the meme appearing, and frames per second on the laptop
- **robustness gap:** accuracy on real data minus accuracy on each synthetic attribute group, compared across training setups

**State the limitation plainly in the report:** 4 people can't establish demographic
fairness. The claim is that robustness to appearance variation improved under
controlled tests.

### L8 — Live demo
Laptop webcam → MediaPipe and/or crops → model → smoothing and hysteresis →
when a meme is detected, show the meme image (`config/meme_images/<id>.jpg`)
in place of the feed. Target at least 15 frames per second end to end.

## 6. Improvement cycle (required by the course)
- **v1:** M0, M1 and M2 trained on real data plus classic augmentation, then
  evaluated with leave-one-person-out and the new-conditions test.
- **Failure analysis:** which classes, conditions and look-alike pairs fail?
- **v2:** add synthetic data (tiers 2 and 3), then re-evaluate on the same
  splits and show the change.

## 7. Suggested timeline (about 5 weeks; adjust once the deadline is confirmed)
| week | work |
|---|---|
| 1 | Team records the sessions (L0) and preprocessing (L1). M0 and M1 baselines on real data. Pick the generation models and prototype on 20 clips. |
| 2 | M2 baseline. Classic augmentation. Tier-2 face editing pipeline plus L4 filter. |
| 3 | Generate synthetic data for each fold at scale on the server. Tier 3 for 2–3 body memes. |
| 4 | Training matrix (L6) and evaluation (L7). Live demo. |
| 5 | Ablations if time allows, report writing, figures, presentation. |

## 8. Ethics notes for the report
- **Stereotyping risk:** changing someone's apparent ethnicity or gender with a
  generative model can produce stereotyped results. Use neutral prompts,
  audit samples by hand, and report problems.
- **Generator bias:** the generators have their own demographic biases (some
  groups render worse). Report rejection rates per attribute.
- **Synthetic faces:** keep altered videos of team members inside the team.
- **Fairness claims:** no claims of fairness for real populations (see L7).
- **Meme images:** used only as on-screen prompts. Cite sources in
  `config/meme_images/SOURCES.md`.

## 9. Decisions (2026-11-10)
1. Synthetic augmentation: **approved** by the professor.
2. Deadline: about 5 weeks (mid-December); the proposal is done.
3. Sessions per member: as many as needed. Reserve one held-out session per person
   (`split.heldout_sessions` in the pipeline config).
4. Generation models: **open**. Fine-tuning on our clips is wanted, and it must be per fold (L3).
5. Public dataset check: **no**.
6. M2 as the main model: **open** (recommended: yes).
7. Demo laptop: M4 MacBook. Compute: NCSA Delta (see README for Slurm jobs).
8. Team: everyone can work on every part.

## 10. Status
Pipeline repo: `ventura-webcam-update` (GitHub otishunt/ventura-webcam-update). It has L1
extraction and labels, M0 rules, M1 GRU/TCN, and lopo/newcond/loso evaluation with a
synthetic-data hook. Everything is tested on fake data. See README.md. Next: run it on the
first real session, then build M2 (crops plus a pixel model) and the tier-2 face-editing
pipeline with the L4 filter.

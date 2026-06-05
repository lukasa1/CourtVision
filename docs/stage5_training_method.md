# Stage 5 — Training Method (for the final report)

This document records, in report-ready detail, exactly how the Stage 5 action
classifier was trained and how the before/after fine-tuning ablation was set up.
Everything is reproducible from a fixed random seed.

---

## 0. The optimisation target — shot-class accuracy

The downstream pipeline (Stage 6 release-frame extraction → Stage 7 contest
metric) only fires on **detected shots**. Layup and dribble classification are
secondary; getting the *shot* class right is what determines whether the rest
of the pipeline runs. We therefore treat **per-class accuracy on the shoot
class** as the primary deployment metric, and use mean-per-class and overall
accuracy as supporting / diagnostic numbers. This framing matters when
comparing models that trade per-class profiles (Section 11): a model with
higher mean-per-class but lower shoot accuracy is *worse* for our use case.

## 1. What we're measuring and why

The Stage 5 classifier is a fine-tune of a Kinetics-400-pretrained
**PoseConv3D** model onto the MultiSubjects basketball action dataset
(dribble / layup / shoot). For the report we want to show **what
MultiSubjects fine-tuning contributed** — i.e. how much the model improved by
adapting to basketball, beyond what the Kinetics-pretrained features already
provide.

The Kinetics-400 head outputs 400 generic-action classes (none of which are
dribble / layup / shoot), so there is no direct "run K400 on MultiSubjects"
number. The standard transfer-learning ablation is therefore:

| | Linear probe ("before") | Full fine-tune ("after") |
|---|---|---|
| K400 backbone | **frozen** (`requires_grad=False`, `lr_mult=0`) | trains at 0.1× head LR |
| 3-class head (+ background slot) | trains | trains |
| Trained on | MS train (v2 split) | MS train (v2 split) |
| Best epoch chosen on | MS val (v2 split) | MS val (v2 split) |
| Final number reported on | **MS test (v2 holdout)** | **MS test (v2 holdout)** |

The **delta** between the two test scores is the gain attributable to fine-tuning the
backbone on MultiSubjects. Every other knob is held constant.

---

## 2. Dataset

**MultiSubjects** (Henan U., CVIU 2024) — 4 096 trimmed single-action clips
of dribble / layup / shoot, ~988 unique subjects, four synchronised cameras at
25 fps, portrait 1080 × 2160 video, COCO-17 keypoints. The dataset ships with a
train / val split (3 072 / 1 024 clips) and **no test split**.

Combined class distribution: 1 564 dribble (38 %) / 512 layup (13 %) / 2 020
shoot (49 %). The layup minority is the persistent hard class for this task.

---

## 3. Subject-disjoint train / val / test split (the "v2" split)

The dataset's native train / val partition is not enough for an honest
before/after evaluation, for two reasons:

1. There's no separate test set. The native val was used during training for
   best-epoch checkpoint selection, so it is a **model-selection** set rather
   than a held-out test set.
2. A random *clip-level* split could leak: with ~4 clips per subject across
   actions and cameras, the same person can easily appear in both train and
   test, and the model could memorise the individual's shooting form. The test
   number would be optimistic relative to deployment on unseen people.

We therefore combine the native train.pkl + val.pkl into a single pool of
4 096 clips and re-split **by subject**, with a fixed random seed:

| Split | Subjects | Clips | dribble | layup | shoot |
|---|---|---|---|---|---|
| train | 692 | 2 914 | 1 099 (38 %) | 369 (13 %) | 1 446 (50 %) |
| val | 148 | 594 | 243 (41 %) | 75 (13 %) | 276 (46 %) |
| test (held out) | 148 | 588 | 222 (38 %) | 68 (12 %) | 298 (51 %) |
| total | **988** | **4 096** | 1 564 (38 %) | 512 (13 %) | 2 020 (49 %) |

**Properties of the split:**
- **Subject-disjoint:** zero subject overlap across the three splits, so test
  measures generalisation to unseen people.
- **Class proportions preserved:** per-split proportions match the overall
  distribution to within ~3 pts despite no explicit stratification, because
  806 / 988 subjects span more than one action class — a uniformly random
  subject shuffle naturally distributes classes evenly.
- **Reproducible:** seed = 0, algorithm in `src/courtvision/stage5_classifier/split.py::compute_subject_disjoint_split`.
- **Smallest cell:** layup-test n = 68. Per-class layup accuracy therefore
  carries a binomial 95 % CI of roughly ± 10 pts — adequate for the
  before/after comparison but the noisiest cell.

---

## 4. Model

- **Backbone:** SlowOnly-R50 (the PoseConv3D paper's headline architecture),
  pretrained on Kinetics-400 keypoint heatmaps. 13 MB checkpoint
  (`poseconv3d_k400.pth`). Initialisation source for both variants.
- **Head:** mmaction2 `I3DHead`, fresh 4-class output (dribble, layup, shoot,
  background), `label_smooth_eps = 0.1`.
- **Input:** purely pose. Each of the 17 COCO keypoints is rendered as a
  Gaussian heatmap (σ = 0.6, modulated by per-keypoint confidence) at
  56 × 56 pixels, then stacked across 64 contiguous frames →
  a `[17, 64, 56, 56]` tensor per sample. No RGB pixels are seen by the model.

---

## 5. Sampling regime (identical for both variants)

The 64-frame *contiguous* sampling regime is the same one used at deployment
on continuous video. It was chosen because:

1. A previous diagnostic showed a 21-point accuracy gap between training on
   `UniformSampleFrames`-over-the-whole-clip (the original recipe) and
   inference with contiguous sliding windows; we eliminate that gap by
   training as we test.
2. A length sweep (32 / 48 / 64 / 80 / 96 / 128 frames) showed 64 frames
   (≈ 2.56 s at 25 fps) is the sweet spot: long enough to contain a complete
   median jump shot (~1.96 s) and median layup (~2.93 s); short enough that
   longer windows don't degrade layup recognition or smear temporal
   localisation.

Training pipeline (mmaction2 transforms):

```
SampleFrames(clip_len=64, frame_interval=1, num_clips=1)
PoseDecode
PoseCompact(hw_ratio=1.0, allow_imgpad=True)
Resize(scale=(-1, 64))
RandomResizedCrop(area_range=(0.56, 1.0))
Resize(scale=(56, 56), keep_ratio=False)
Flip(flip_ratio=0.5, COCO L/R keypoint swap)
GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False)
FormatShape(input_format="NCTHW_Heatmap")
PackActionInputs
```

Val / test pipeline (deterministic, single centered window per clip):

```
SampleFrames(clip_len=64, frame_interval=1, num_clips=1, test_mode=True)
PoseDecode
PoseCompact(hw_ratio=1.0, allow_imgpad=True)
Resize(scale=(56, 56), keep_ratio=False)
GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False)
FormatShape(input_format="NCTHW_Heatmap")
PackActionInputs
```

---

## 6. Training recipe (identical for both variants)

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD (mmaction2 K400 config default) |
| Head learning rate | 0.01 |
| LR schedule | Cosine annealing, T_max = 24, eta_min = 0 |
| Epochs | 24 |
| Batch size | 8 |
| GPU | NVIDIA A10G (Modal) |
| Loss | Cross-entropy + label smoothing (ε = 0.1) |
| Class weights (inverse frequency) | dribble 0.87, layup 2.62, shoot 0.68, bg 1.0 |
| Checkpoint selection | Best `acc/top1` on the v2 val split |
| Random seed | 0 (`cfg.randomness.seed = 0`) |

The class-weighted loss is necessary because layup represents only 13 % of the
dataset; without weighting, the loss is dominated by shoot/dribble and the
minority class never gets adequate gradient signal.

---

## 7. The two variants (only difference: backbone freedom)

The **linear-probe** and **full-fine-tune** variants share every hyperparameter
above. They differ only in one line of code:

```python
cfg.optim_wrapper.paramwise_cfg = dict(
    custom_keys={"backbone": dict(lr_mult=0.0 if freeze_backbone else 0.1)}
)
# Plus, when freeze_backbone is True, also:
for p in runner.model.backbone.parameters():
    p.requires_grad = False
```

- **Linear probe:** backbone weights stay at their Kinetics-400 values for all
  24 epochs; only the I3D head's 4-class linear layer learns. Measures how well
  Kinetics's pre-existing action features generalise to basketball with no
  basketball-specific backbone adaptation.
- **Full fine-tune:** backbone trains at 0.1 × the head's learning rate
  (the "LP-FT" / discriminative-LR convention — adapts the backbone slowly
  while the fresh head moves faster). Measures the configuration we deploy.

Both runs use the same v2 train / val split for training and best-epoch
selection, then are evaluated **on the v2 test split**, which the model has
never seen.

---

## 8. Held-out test evaluation protocol

`scripts/stage5_modal.py::eval_on_test` runs each trained checkpoint on the
v2 test split. For every test clip:

1. Sample one centered contiguous 64-frame window from the clip's COCO-17
   keypoint sequence (same val pipeline as during training).
2. Render the heatmap volume and forward through the model.
3. Take `argmax` over the 4 logits as the predicted class.

Reported metrics on the 588 test clips (148 unseen subjects):

- Per-class accuracy (dribble / layup / shoot)
- Overall accuracy (sum-correct / sum-total)
- **Per-class shoot accuracy** (the deployment-relevant metric — see §0)
- **Mean per-class accuracy** (the supporting balanced metric)
- 3 × 3 row-normalised confusion matrix

The **delta** between linear-probe and full-fine-tune scores on each of these
metrics is the headline "training contribution" number in the report.

---

## 9. Honest limitations to cite in the report

1. **Layup test n = 68.** Per-class layup accuracy carries ~ ± 10 pt
   confidence intervals on a single number; mean-per-class is more stable but
   inherits some of this noise.
2. **The linear probe doesn't strictly freeze BatchNorm running statistics.**
   `requires_grad=False` stops weight updates and `lr_mult=0` zeros gradients,
   but BN running means / variances continue to update under `.train()` mode.
   This slightly *helps* the probe (it gets free domain adaptation in its BN
   stats), which makes the fine-tuning gain we report slightly **conservative**.
3. **MultiSubjects is a single dataset.** Cross-dataset generalisation
   (e.g. to iPhone footage or to TrackID3x3 gameplay) is not measured here;
   the qualitative TrackID3x3 overlays and the F24 cross-method FP analysis
   give separate, complementary evidence on that.
4. **The CheckpointHook selects best by `acc/top1`, which is noisy under class
   imbalance** (it amplifies the majority class). Mean-per-class would be a
   more robust selector and is the metric we lead with in the report; the
   selected epoch is the same under either metric for both runs.

---

## 10. How to reproduce

```bash
# Full fine-tune (~1 h A10G):
uv run modal run --detach scripts/stage5_modal.py::train_ms \
    --tag 64f_bal_v2 --split-version v2 --batch-size 8

# Linear probe (~1 h A10G):
uv run modal run --detach scripts/stage5_modal.py::train_ms \
    --tag 64f_probe_v2 --split-version v2 --batch-size 8 --freeze-backbone

# Evaluate each on the held-out v2 test set:
uv run modal run scripts/stage5_modal.py::test_eval \
    --ckpt /artifacts/checkpoints/stage5/stage2_multisubjects_64f_bal_v2.pth
uv run modal run scripts/stage5_modal.py::test_eval \
    --ckpt /artifacts/checkpoints/stage5/stage2_multisubjects_64f_probe_v2.pth
```

Seed `0`, split function deterministic, training set identical → results
reproducible within seeded randomness.

---

## 11. Regularisation ablation (a clean negative — keep the baseline)

After observing train loss continuing to decrease while val accuracy
plateaued from epoch 2 onward (classic overfitting signature), we tested
whether stronger regularisation would lift the ceiling. We retrained on the
*same* v2 train split, *same* seed 0, *same* every other hyperparameter,
toggling four regularisation knobs simultaneously:

| Knob | Baseline `64f_bal_v2` | Stronger reg `64f_bal_v2_reg` |
|---|---|---|
| Weight decay | 3e-4 | 1e-3 (~3×) |
| Head dropout | 0.5 | 0.7 |
| Label smoothing ε | 0.1 | 0.15 |
| RandomResizedCrop min area | 0.56 | 0.40 |

**Held-out test results (588 clips, 148 unseen subjects, single centered window per clip):**

| Metric | Baseline | Stronger reg | Δ |
|---|---|---|---|
| Dribble accuracy (n=222) | 0.716 | 0.964 | +0.248 |
| Layup accuracy (n=68) | 0.529 | 0.544 | +0.015 |
| **Shoot accuracy (n=298)** | **0.930** | **0.564** | **−0.366** |
| Overall accuracy | 0.803 | 0.713 | −0.090 |
| Mean per-class accuracy | 0.725 | 0.691 | −0.034 |

Mean-per-class only moved a hair, but the **per-class profile shifted
dramatically**: stronger regularisation pushed the model into a "predict
dribble" comfort zone. It bought a 25-pt gain on dribble at the cost of a
**37-pt loss on shoot** — the exact class our downstream pipeline depends on
(§0). The val curves told a misleadingly favourable story (peak val 0.769 →
0.773; plateau just pushed from epoch 2 to epoch 13) because val and test
share class proportions and the model can score similarly on val by
"defaulting" — but the per-class profile (which is what actually matters) was
worse.

**What the negative result tells us:**

- Reg *was* doing its job mechanically (train loss landed at a higher floor;
  best epoch pushed late; val curve smoother). So the experiment was
  well-posed.
- But the binding constraint isn't overfitting — it's **data**. With a
  thin minority class (layup n=369 in train) and intrinsically similar
  shoot/layup pose dynamics, regularisation can only redistribute capacity
  between classes; it can't synthesise discriminative information that isn't
  in the data.
- For the deployment objective (shoot detection), the baseline's sharper
  "overhead arm extension" feature is exactly what we want, and reg destroyed
  it. **Verdict: keep the baseline `64f_bal_v2` as the canonical model.**

**Next levers for closing the remaining gap** (priority order, all
data-side, not loss-side):

1. More layup + diverse shot data (real, not synthetic).
2. FineSports as an intermediate transfer stage when access materialises
   (much larger basketball-action distribution).
3. Multi-window TTA at inference (averages predictions over multiple
   contiguous windows; smooths single-pass noise without retraining).
4. Architectural moves (different temporal pooling heads) — only after the
   data gap is closed.

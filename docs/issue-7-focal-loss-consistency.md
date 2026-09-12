# Issue 7 — Focal-loss and model-output consistency

## Finding

The released SAR-LRA reference implementation has a sigmoid final layer and defines focal loss by calling binary cross-entropy with `from_logits=True`.

This exact combination is present in the repository's earliest retained implementation and remains present in the archived V1 and V2 reference notebooks. The packaged refactor introduced in Issue 3 had silently changed the loss call to `from_logits=False`. Issue 7 removes that silent semantic change.

The repository does **not** contain the complete original training pipeline or a frozen training environment, so the source history establishes the published/released loss implementation but cannot prove every detail of the environment used to train the distributed weights.

## Decision

Released-weight inference is separated from training-loss semantics:

- `build_model()` constructs the released architecture but is uncompiled by default.
- `load_model()` loads weights into the uncompiled model.
- `released_focal_loss()` preserves the historical `from_logits=True` implementation for reproducibility.
- `probability_focal_loss()` exposes the direct probability-space interpretation (`from_logits=False`) only for controlled retraining/experiments.
- `compile_for_training(..., semantics=...)` requires the caller to choose the objective explicitly.

This means correcting or experimenting with the training loss cannot change forward inference from existing weights.

## Why the historical definition is potentially confusing

A sigmoid model output is normally interpreted as a probability, while `from_logits=True` normally means the supplied value is an unbounded pre-sigmoid logit. Modern Keras documentation distinguishes these two input contracts explicitly.

Older TensorFlow/Keras implementations also cached pre-sigmoid logits on tensors produced by a Keras sigmoid activation so cross-entropy could recover the numerically stable logits in graph/model contexts. That implementation detail means the source code cannot be judged solely by treating a sigmoid output as an arbitrary standalone probability tensor. It is one reason the released definition is preserved rather than silently rewritten.

## Compatibility requirements

### Released-weight inference

Inference does not require model compilation. The supported package range remains TensorFlow 2.12 through 2.18, subject to the TensorFlow build supporting the selected Python version. Weight compatibility is tested independently of the loss function.

### Reproducing historical training semantics

For exact training reproduction, use a TensorFlow/Keras environment matching the original experiment if it can be recovered. The repository currently does not contain a lock file or training-environment capture sufficient to identify that exact version.

For new training, do not rely on undocumented Keras internals. Select `released` or `probability` loss semantics explicitly and record TensorFlow, Keras, Python, CUDA, and cuDNN versions with the resulting weights.

## Numerical verification

Tests cover three facts:

1. A known probability and its corresponding logit produce the same BCE/focal value when each is interpreted correctly.
2. Treating a probability value itself as a logit produces a different loss.
3. Compiling identical model weights with the released loss versus probability-space loss does not change forward predictions.

The third test is the critical released-weight guarantee: the loss affects training/evaluation loss calculations, not the deterministic inference graph.

## Released output reproducibility

No layer, activation, tensor shape, or weight-loading operation was changed by this issue. The final layer remains sigmoid. Existing weights are loaded without compilation, so predictions from the released weights are not numerically altered by the focal-loss decision.

A future reference-event regression fixture (Issue 12) is still required to lock end-to-end prediction values across TensorFlow/CPU/GPU environments.

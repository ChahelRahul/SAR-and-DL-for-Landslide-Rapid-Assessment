## Summary

Adds explicit scientific, provenance, input, limitation, and human-review documentation for the released SAR-LRA models.

## Changes

- adds `MODEL_CARD.md`;
- documents training and unseen-event datasets;
- documents ascending and descending architecture differences;
- defines the required four-band order and dB representation;
- documents preprocessing, temporal stacks, and known failure modes;
- requires expert review;
- adds a model-weight source and integrity manifest;
- adds a checksum verification script;
- adds a patch that records model version and weight checksum in raster and vector outputs.

## Licensing blocker

The repository MIT licence does not explicitly state that it covers trained model weights. The model card records this as unconfirmed. A rights holder must approve and commit the proposed explicit weight-licensing statement before that acceptance criterion can be checked.

## Validation

```bash
python scripts/verify_model_weights.py --update-manifest
git diff --check
```

Closes #2 after model-weight licensing is explicitly confirmed and SHA-256 values are committed.

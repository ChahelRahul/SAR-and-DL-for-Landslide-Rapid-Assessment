# SAR-LRA v2.0.0 release checklist

## Blocking legal/provenance gate

- [ ] Obtain explicit maintainer/rightsholder confirmation for redistribution of both trained `.hdf5` weight files.
- [ ] Replace `UNCONFIRMED` in `model/weights-manifest.json` with the exact granted licence.
- [ ] Update the trained-weight licensing section of `MODEL_CARD.md` with matching language.

## Repository checks

- [ ] `pyproject.toml` version is `2.0.0`.
- [ ] `app.__version__` is `2.0.0`.
- [ ] Model identifier remains `sar-lra-v2.0.0-beta.1` unless retrained/reissued weights are deliberately introduced.
- [ ] Full unit/regression suite passes.
- [ ] Required tracked TIFF fixtures are present after a clean Git checkout.
- [ ] CPU/GPU direct dependency locks resolve on Python 3.11.
- [ ] CPU container workflow passes.
- [ ] GPU image build/static workflow passes.
- [ ] Optional real-GPU runner test passes if available.
- [ ] Both released model checksums match `model/weights-manifest.json`.
- [ ] `CHANGELOG.md` and `RELEASE_NOTES_v2.0.0.md` are reviewed.

## Registry/release checks

- [ ] Confirm GitHub release/tag `v2.0.0` does not already exist.
- [ ] Confirm GHCR tags `sar-lra:2.0.0` and `sar-lra-gpu:2.0.0` do not already exist.
- [ ] Push annotated `v2.0.0` tag.
- [ ] Confirm CPU/GPU images are published, scanned, and Cosign-signed.
- [ ] Confirm GitHub Release attaches wheel, sdist, SHA256SUMS, CPU SBOM, and GPU SBOM.
- [ ] Pull `ghcr.io/chahelrahul/sar-lra:2.0.0` on a clean machine.
- [ ] Run the notebook-free quickstart and inspect generated metadata/artifacts.

## Post-release

- [ ] Verify documentation links point to the immutable `2.0.0` image where appropriate.
- [ ] Record any release defect as a new issue; do not overwrite `2.0.0`.
- [ ] Use `2.0.1` or later for corrections.

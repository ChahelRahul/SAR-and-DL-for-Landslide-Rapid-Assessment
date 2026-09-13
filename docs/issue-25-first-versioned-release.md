# Issue 25 — first versioned release

Target software release: **v2.0.0**.

## Why 2.0.0

The refactored package has used the `2.0.0b1` prerelease line throughout development, and the bundled model identifier is `sar-lra-v2.0.0-beta.1`. The first stable software tag therefore advances the package from `2.0.0b1` to `2.0.0`; it does not rename or retrain the model assets.

## Mandatory pre-tag gate

The release must not be tagged while the trained-weight licence is recorded as `UNCONFIRMED`. Obtain explicit rightsholder/maintainer confirmation for the two `.hdf5` files, then update `model/weights-manifest.json` and the licensing section of `MODEL_CARD.md` with the exact granted licence.

## Release sequence

1. Merge the v2.0.0 release-preparation commit into `main`.
2. Confirm all tests and container workflows are green.
3. Confirm the trained-weight distribution licence and update the manifest/model card.
4. Verify no existing `v2.0.0` release/tag or GHCR `2.0.0` image exists.
5. Create and push an annotated tag:

```bash
git switch main
git pull --ff-only
git tag -a v2.0.0 -m "SAR-LRA 2.0.0"
git push origin v2.0.0
```

6. The tag-triggered release workflow:
   - runs the test suite;
   - publishes CPU and GPU images to GHCR;
   - generates/scans/signs the container digests;
   - builds the Python wheel and source distribution;
   - creates `SHA256SUMS`;
   - creates the GitHub Release using `RELEASE_NOTES_v2.0.0.md`;
   - attaches wheel, sdist, checksums, and both SPDX SBOMs.
7. Verify pulls:

```bash
docker pull ghcr.io/chahelrahul/sar-lra:2.0.0
docker pull ghcr.io/chahelrahul/sar-lra-gpu:2.0.0
```

8. Run the notebook-free smoke test from `docs/end-user-deployment.md`.

## Rollback policy

The semantic release tag and image tag are immutable. If a release defect is found after publication, do not overwrite `2.0.0`; publish a corrected patch release such as `2.0.1`.

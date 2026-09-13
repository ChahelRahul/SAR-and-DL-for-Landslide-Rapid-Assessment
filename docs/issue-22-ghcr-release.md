# Issue 22 — Publish images through GitHub Actions and GHCR

SAR-LRA publishes container images to GitHub Container Registry (GHCR) from `.github/workflows/publish-images.yml`.

## Registry layout

- CPU: `ghcr.io/<owner>/sar-lra`
- GPU: `ghcr.io/<owner>/sar-lra-gpu`

A push to `main` runs the test suite and publishes the CPU image. A semantic release tag such as `v2.0.0` runs the tests and publishes both CPU and GPU images.

## Tags

For `v2.0.0`, both release images receive:

- `2.0.0`
- `2.0`
- `1`
- `latest`
- `sha-<short-commit>`

The workflow signs the immutable image digest; the semantic aliases are convenience references to that digest. `sha-*` tags are commit-specific and semantic-version tags must not be reused for a different release.

## Supply-chain checks

Before signing, each pushed image:

1. is built for `linux/amd64` with BuildKit provenance;
2. receives an SPDX JSON SBOM generated with Anchore Syft through `anchore/sbom-action`;
3. is scanned by Trivy and the workflow fails on fixed HIGH or CRITICAL vulnerabilities;
4. is signed by Cosign using GitHub Actions OIDC/keyless signing.

SBOM files are retained as GitHub Actions artifacts. Image signing uses `id-token: write`; no long-lived signing key is stored in repository secrets.

## Release procedure

Create an annotated semantic version tag and push it:

```bash
git tag -a v2.0.0 -m "SAR-LRA 2.0.0"
git push origin v2.0.0
```

After the workflow succeeds:

```bash
docker pull ghcr.io/<owner>/sar-lra:2.0.0
docker pull ghcr.io/<owner>/sar-lra-gpu:2.0.0
```

The package owner may need to make the GHCR package public once, depending on organization/package defaults. Publishing itself uses the workflow-provided `GITHUB_TOKEN` with `packages: write`.

## Verification

Cosign verification should bind the signature to the repository workflow identity. Replace `<owner>/<repo>` with the real GitHub repository:

```bash
cosign verify \
  --certificate-identity-regexp 'https://github.com/<owner>/<repo>/.github/workflows/publish-images.yml@refs/(heads/main|tags/v.*)' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  ghcr.io/<owner>/sar-lra:2.0.0
```

The workflow deliberately does not publish GPU images from ordinary `main` pushes; GPU artifacts are release-tag only. Existing pull-request Docker workflows continue to build and smoke-test images without publishing them.

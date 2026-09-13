# Issue 24 — End-user deployment documentation

The primary end-user guide is [`end-user-deployment.md`](end-user-deployment.md). It consolidates the previously issue-specific deployment material into a single notebook-free workflow.

The guide covers every Issue 24 documentation requirement:

- CLI-only installation and execution;
- Earth Engine credentials and headless/container authentication;
- prepared-raster inference;
- synchronous and asynchronous API deployment;
- Docker and Podman examples;
- CPU and NVIDIA GPU expectations;
- ROI and event-date examples;
- output semantics and interpretation;
- troubleshooting;
- scientific limitations and human-review requirements;
- publication citation and model/checksum provenance.

## Acceptance path

A small synthetic four-band prepared raster and ROI are committed under `examples/quickstart/`. Both CPU and GPU Dockerfiles already copy `examples/` into `/opt/sar-lra/examples`, so a user can pull the CPU image and execute the sample directly from the image without cloning the repository, downloading Earth Engine imagery, or opening a notebook.

The end-user guide uses the released ASCENDING weight path embedded in the image and writes outputs only to a mounted `/output` directory.

The fixture is explicitly described as synthetic and is not presented as a real-event accuracy benchmark.

## Registry caveat

Issue 22 implements the GHCR publication workflow, but repository-side tests cannot prove that a registry owner has actually made a package public or that a specific tag exists. The guide therefore defines the intended canonical image path and recommends immutable version tags for production after a versioned release is published. Issue 25 is responsible for the first release contents.

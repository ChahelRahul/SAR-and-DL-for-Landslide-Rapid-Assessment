# Issue 18 — streamed / tiled inference

## Problem

The previous inference path accumulated every sliding-window patch in a Python
list and converted the complete set into a single NumPy tensor before calling
`model.predict()`. Patch memory therefore grew with ROI size.

## Streaming design

Inference now consumes `iter_window_batches()`. At most `model.batch_size`
`tile_size × tile_size × 4` float32 patches are materialized at one time.
Patch preparation can use `processing.inference_workers`, but workers operate
only on the current bounded batch; they do not introduce an unbounded prefetch
queue.

Production pipeline runs provide temporary memmap paths to `predict_stack()`:

- probability surface: float32 disk-backed memmap;
- binary mask: uint8 disk-backed memmap.

Window probabilities are merged into the probability memmap immediately after
each model batch. Final GeoTIFFs are written in bounded row chunks. Invalid
pixels remain `NaN` internally and become GeoTIFF NoData.

The temporary `.streaming/` workspace is deleted after durable raster/vector
outputs and `result.json` are written.

## Limits

```yaml
model:
  batch_size: 512
processing:
  inference_workers: 1
  nms_diagnostic_limit: 100000
  output_rows_per_chunk: 1024
```

CLI overrides are `--batch-size`, `--inference-workers`,
`--nms-diagnostic-limit`, and `--output-rows-per-chunk`.

NMS is diagnostic only. To stop diagnostic metadata from becoming a separate
unbounded-memory path, no more than `nms_diagnostic_limit` positive windows are
retained for NMS. The total positive-window count is still recorded and
`diagnostics_truncated=true` is emitted if the limit is exceeded.

## Numerical compatibility

`maximum` overlap aggregation is applied in the same window traversal order as
the previous implementation. Tests compare streamed and in-memory outputs at
`atol=1e-7`, including NoData/ROI masking. Binary masks must match exactly.

## Representative memory check

Run:

```bash
python scripts/benchmark_streaming_memory.py
```

The benchmark allocates the input raster before measuring inference overhead,
uses the same batch size for a 512×512 and a 2048×2048 raster, and reports both
process max-RSS delta and the exact maximum patch-tensor bytes. The invariant
used by the automated test is:

`peak_patch_batch_bytes <= batch_size × tile_size × tile_size × 4 bands × 4 bytes`.

The process RSS figure is diagnostic rather than a hard CI threshold because
NumPy allocators, GDAL and operating systems retain memory differently.

Representative run in the Issue 18 development environment (`batch_size=8`):

| Raster | Windows | Peak patch tensor | Max-RSS delta |
|---|---:|---:|---:|
| 512×512 | 225 | 524,288 bytes | 3.250 MiB |
| 2048×2048 | 3,969 | 524,288 bytes | 28.359 MiB |

The exact patch tensor remains constant while the number of windows increases
17.6×. The RSS delta is not constant because the source stack, validity mask,
memory-mapped pages, NumPy/GDAL allocator behavior and OS page cache still scale
with raster dimensions. The acceptance target addressed here is elimination of
the all-window patch tensor; full streaming source-raster I/O is explicitly out
of scope for this issue.

## Remaining memory scaling

The prepared Sentinel-1 input stack itself is still read as one raster array,
and vector polygonization may inspect full raster-sized arrays. Issue 18 removes
the specific all-windows-in-RAM failure mode and disk-backs the probability/mask
outputs; fully windowed source-raster I/O would be a separate acquisition/input
architecture change.

# Issue 21 — Resource, timeout, concurrency, cancellation, and cleanup controls

Issue 21 hardens the asynchronous deployment added in Issue 20. It does not alter the SAR-LRA scientific inference path.

## Execution timeout

Each async job stores an execution timeout. The API default is controlled by:

```text
SAR_LRA_JOB_TIMEOUT_SECONDS=3600
```

The Linux worker uses `SIGALRM`/`setitimer` as a hard process-local deadline and also checks the deadline at pipeline progress boundaries. A timed-out job enters `failed` with the stable public error code `job_timeout`. Partial files under `/output/<job_id>` are removed.

## Cancellation

`DELETE /v1/jobs/{job_id}` changes the job state to `cancelled`. The worker checks the durable record at stage boundaries. If cancellation is observed, processing stops at the next checkpoint and `/output/<job_id>` is removed. The shared Earth Engine cache is not deleted.

Cancellation is cooperative between pipeline stages. A native TensorFlow/GDAL call that does not return to Python immediately may not stop until the call returns; the hard timeout is the upper-bound safeguard in the reference Linux worker.

## Queue backpressure

`SAR_LRA_MAX_QUEUED_JOBS` defaults to 100. New submissions and retries return HTTP 429 with `Retry-After: 30` when the queue has reached the configured limit.

## Retry limit

`SAR_LRA_JOB_MAX_ATTEMPTS` defaults to 3. Once a job has consumed that many worker attempts, the retry endpoint returns HTTP 409 with `attempt_limit_reached`. Duplicate deliveries of completed jobs remain idempotent.

## Disk admission control

Before loading model weights or starting acquisition/inference, the worker checks available space on the output filesystem. `SAR_LRA_MIN_FREE_DISK_MB` defaults to 1024. If the floor is not met the job fails with `insufficient_storage` and no partial per-job result directory is retained.

## Crash recovery

Redis dequeue uses a reservation list (`BRPOPLPUSH`) rather than destructive `BLPOP`. The reference deployment runs one worker process. On worker startup, job IDs left in the reservation list by a previous crashed worker are moved back to the main queue unless their records are already terminal.

This single-worker recovery policy is intentionally conservative. A multi-worker deployment should use an external lease/reaper policy instead of enabling several workers against the reference reservation recovery behavior.

## Concurrency

The reference architecture uses:

- bounded synchronous API concurrency from Issue 19;
- bounded asynchronous queue depth;
- one inference job at a time per `sar-lra-worker` process.

Scale async throughput by adding isolated workers only after implementing an external lease/reaper strategy appropriate for the deployment.

## Container resource limits

`docker-compose.async.yml` provides defaults:

```text
API:    1 CPU, 1 GiB memory, 256 PIDs
worker: 4 CPUs, 8 GiB memory, 512 PIDs
```

These can be overridden with `SAR_LRA_API_*_LIMIT` and `SAR_LRA_WORKER_*_LIMIT` Compose environment variables. Worker shutdown gets a 30-second grace period.

## Cleanup policy

Successful outputs remain until normal job/result retention policy removes metadata or an external storage lifecycle policy removes artifacts. Cancelled, failed, timed-out, and attempt-exhausted executions remove their per-job output tree. Shared caches are preserved.

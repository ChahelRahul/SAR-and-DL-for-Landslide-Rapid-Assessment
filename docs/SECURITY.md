# Security notes

- Never send provider secrets in HTTP request JSON.
- Keep `PC_SDK_SUBSCRIPTION_KEY`, Google credentials, MinIO secrets, and AWS credentials in environment/secret stores or read-only mounts.
- Do not commit `.env` or credential JSON files.
- API input paths are restricted beneath `SAR_LRA_API_INPUT_ROOT`; outputs are restricted to the configured output root.
- Operator-supplied model weights are restricted to the mounted input tree by the API.
- Unexpected exceptions are not reflected verbatim to clients to avoid leaking credential paths/provider details.
- Containers run as a non-root user and production images verify bundled model weights.
- Configure reverse-proxy authentication/TLS when exposing the API outside a trusted network; SAR-LRA itself is not an identity provider.
- Rotate the default MinIO credentials before any shared/server deployment.

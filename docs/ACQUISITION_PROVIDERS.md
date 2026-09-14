# Acquisition providers and credentials

SAR-LRA separates remote Sentinel-1 acquisition from inference. Users can choose a provider per request/CLI invocation or let `auto` select a provider from the credentials available in the runtime environment.

## Provider matrix

| Provider | Selector | Credential | What it supplies | Notes |
|---|---|---|---|---|
| Microsoft Planetary Computer | `planetary-computer` | `PC_SDK_SUBSCRIPTION_KEY` | Sentinel-1 RTC VV/VH assets | Public STAC discovery; RTC asset access is signed. RTC linear intensity is converted to dB before compositing. |
| Google Earth Engine | `earth-engine` | service account/ADC/Earth Engine credential/attached identity | Sentinel-1 GRD processing through Earth Engine | A registered Google Cloud/Earth Engine project may also be required. |
| Automatic | `auto` | whichever supported credential is present | Chooses one of the above | PC key is preferred if present; otherwise conventional EE credential files are detected. For workload identity, select `earth-engine` explicitly. |
| Prepared raster | `/v1/predict-raster` or `sar-lra predict-raster` | none | User-supplied four-band GeoTIFF | Fully provider-independent and can run offline. |

Secrets are **not accepted in API request bodies**. Inject them using environment variables, read-only secret mounts, ADC, or cloud workload identity.

## Automatic selection

`provider: "auto"` is the API default. Resolution order is:

1. Concrete `SAR_LRA_ACQUISITION_PROVIDER` (`planetary-computer` or `earth-engine`).
2. `PC_SDK_SUBSCRIPTION_KEY` if present.
3. Conventional local Earth Engine/ADC credential files if detectable.
4. Otherwise fail with a message directing the user to configure a provider or use prepared-raster inference.

Cloud metadata/workload identity cannot always be detected safely. Set `SAR_LRA_ACQUISITION_PROVIDER=earth-engine` in those deployments.

## Planetary Computer

Required for Sentinel-1 RTC asset reads:

```bash
export PC_SDK_SUBSCRIPTION_KEY='...'
export SAR_LRA_ACQUISITION_PROVIDER=planetary-computer
```

The provider searches `sentinel-1-rtc`, filters for IW and VV/VH, matches the requested ascending/descending pass, selects a common relative orbit across pre/post scenes, converts RTC gamma-nought intensity to dB, creates temporal medians, and writes the model stack in this order:

1. `postVV`
2. `postVH`
3. `diffVV = postVV - preVV`
4. `diffVH = postVH - preVH`

Scientific caveat: Planetary Computer RTC is not asserted to be numerically identical to the Earth Engine preprocessing used by the original model workflow. Validate against reference events before treating providers as interchangeable.

## Earth Engine

Supported non-interactive credential patterns:

- `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`
- Application Default Credentials
- a read-only mounted `~/.config/earthengine/credentials`
- cloud workload identity / attached service account

Example:

```bash
export SAR_LRA_ACQUISITION_PROVIDER=earth-engine
export GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gee.json
```

Pass the Google Cloud project in the request `project` field or CLI `--project` when required.

## Credential-free mode

For deployments that cannot use any external provider credential, prepare the required four-band raster outside SAR-LRA and use:

```text
POST /v1/predict-raster
```

or:

```bash
sar-lra predict-raster --input /input/stack.tif --orbit ASCENDING --output-dir /output
```

After the raster and model weights are available locally, this path does not call Earth Engine, Planetary Computer, or another imagery API.

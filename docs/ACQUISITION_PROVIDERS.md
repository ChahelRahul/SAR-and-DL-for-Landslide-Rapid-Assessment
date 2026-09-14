# Acquisition providers

SAR-LRA separates imagery acquisition from inference. The request selects a provider; credentials are deployment secrets and are never accepted in API request bodies.

| Mode | Request value | User API key required? | Source | Processing |
|---|---|---:|---|---|
| Planetary Computer GRD | `planetary-computer-grd` or alias `planetary-computer` | **No** | `sentinel-1-grd` | Downloads SAFE via an anonymously issued read-only SAS token; local radiometric terrain correction with `sarsen`; temporal composites; dB conversion |
| Planetary Computer RTC | `planetary-computer-rtc` | Yes | `sentinel-1-rtc` | Reads Microsoft's precomputed RTC VV/VH assets, converts linear intensity to dB, then composites |
| Google Earth Engine | `earth-engine` | Google/EE identity required | `COPERNICUS/S1_GRD` | Existing Earth Engine preprocessing pipeline |
| Prepared raster | `/v1/predict-raster` | **No** | user-supplied 4-band GeoTIFF | No remote acquisition |

## Recommended keyless mode

`planetary-computer-grd` is the default for `provider: "auto"`. Planetary Computer STAC metadata is queried without credentials. The implementation then requests a temporary read-only SAS token for the `sentinel1euwest/s1-grd` Azure container using `planetary_computer.sas.get_token(...)`. A user subscription key is not required for this GRD token path.

The downloaded Sentinel-1 GRD SAFE products are processed locally with `sarsen`. SAR-LRA selects IW dual-polarization VV/VH scenes, the requested ascending/descending orbit, and a relative orbit represented in both pre- and post-event windows. The default windows remain 60 days before and 12 days after the event.

A terrain model is required for local correction. By default SAR-LRA searches Planetary Computer's `cop-dem-glo-30` collection and creates a projected ROI DEM. Operators can instead mount a DEM and set:

```bash
export SAR_LRA_PC_DEM_PATH=/input/dem.tif
```

The GRD route is computationally heavier than consuming precomputed RTC because complete SAFE products are downloaded and terrain correction is performed locally. Cache directories are reused across requests.

## Precomputed Planetary Computer RTC

Use this when an account key is available and lower processing cost is preferred:

```bash
export PC_SDK_SUBSCRIPTION_KEY='...'
export SAR_LRA_ACQUISITION_PROVIDER=planetary-computer-rtc
```

or select it per request with `"provider": "planetary-computer-rtc"`. This mode requires the key because the precomputed `sentinel-1-rtc` collection is account-gated.

## Earth Engine

Use `earth-engine` with one of the supported Google credential mechanisms, such as `GOOGLE_APPLICATION_CREDENTIALS`, Application Default Credentials, Earth Engine credentials under the runtime home directory, or an attached workload identity/service account. Interactive authentication is for local development only.

## Auto resolution

`auto` resolves as follows:

1. A concrete `SAR_LRA_ACQUISITION_PROVIDER`, if configured.
2. Otherwise `planetary-computer-grd`.

Explicit request selection always wins. `planetary-computer` is a compatibility alias for `planetary-computer-grd`.

## Scientific compatibility

Provider interchangeability is not assumed. Planetary Computer GRD + `sarsen`, Planetary Computer's precomputed RTC, and Earth Engine can differ in terrain correction, DEM, calibration details, resampling, and edge behavior. The model contract still requires four dB bands in this order: `postVV`, `postVH`, `diffVV`, `diffVH`. Run the reference regression suite before declaring providers scientifically equivalent for production reporting.

## Security

Secrets are **not accepted in API request bodies**. Keep `PC_SDK_SUBSCRIPTION_KEY`, Google credentials, and object-store credentials in an environment secret store or read-only secret mount. The keyless GRD provider still uses short-lived SAS tokens issued by Planetary Computer internally; callers do not supply or manage those tokens.

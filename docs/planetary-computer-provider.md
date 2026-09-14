# Microsoft Planetary Computer acquisition provider

This provider is one of SAR-LRA's selectable remote Sentinel-1 acquisition backends. It is **not** forced as the default; `provider=auto` selects a configured provider from the runtime environment. See [ACQUISITION_PROVIDERS.md](ACQUISITION_PROVIDERS.md) for the complete provider/credential matrix.

The implementation searches the Planetary Computer STAC catalog at `https://planetarycomputer.microsoft.com/api/stac/v1` using `sentinel-1-rtc`, filters IW dual-polarization VV/VH scenes and a common relative orbit, reads signed VV/VH COG assets, converts linear RTC intensity to dB, creates pre/post temporal medians, and writes `postVV`, `postVH`, `diffVV`, `diffVH`.

## Authentication

STAC discovery is public, but the Sentinel-1 RTC notebook/documentation supplied by Microsoft states that local asset access requires an API/subscription key. Supply `PC_SDK_SUBSCRIPTION_KEY` only at runtime; do not copy it into images or repositories.

```bash
export PC_SDK_SUBSCRIPTION_KEY='...'
sar-lra predict --provider planetary-computer --roi roi.geojson --event-date 2025-08-15 --orbit ASCENDING --output-dir results
```

## Scientific compatibility note

Planetary Computer `sentinel-1-rtc` is radiometrically terrain corrected and stored as linear intensity. SAR-LRA converts each RTC scene to dB before temporal median compositing. This is not asserted to be numerically identical to the historical Earth Engine GRD preprocessing chain; reference-event regression is required before treating provider outputs as interchangeable.

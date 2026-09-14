# Architecture

```text
ROI + event date
      |
      +--> provider=planetary-computer --> Sentinel-1 RTC --> dB/composites --+
      |                                                                    |
      +--> provider=earth-engine --------> Sentinel-1 GRD processing -------+--> 4-band stack
      |                                                                    |        |
      +--> prepared raster -------------------------------------------------+        v
                                                                                validation
                                                                                    |
                                                                            sliding-window model
                                                                                    |
                                                                         probability aggregation
                                                                                    |
                                                                       threshold/vector postprocess
                                                                                    |
                                                                          filesystem / S3 results
```

The inference core is independent of the remote provider once the validated four-band stack exists.

The model uses 64×64×4 patches. Sliding-window inference covers raster edges, batches patches incrementally to bound memory, aggregates probabilities, and performs probability-preserving post-processing before optional threshold/vector conversion.

Remote imagery defaults use the configured pre-event and post-event windows (reference defaults: 60 days pre and 12 days post). Orbit direction is not mixed within a model run; separate bundled ASCENDING and DESCENDING weights are available.

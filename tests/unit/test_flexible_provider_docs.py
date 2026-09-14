from pathlib import Path


def test_comprehensive_docs_are_present():
    required = {
        "README.md", "API.md", "ACQUISITION_PROVIDERS.md", "CLI.md",
        "configuration.md", "DEPLOYMENT.md", "OUTPUTS.md", "ARCHITECTURE.md",
        "SECURITY.md", "TROUBLESHOOTING.md", "openapi.json",
    }
    assert required.issubset({p.name for p in Path("docs").iterdir() if p.is_file()})


def test_api_docs_cover_all_application_routes():
    text = Path("docs/API.md").read_text()
    for route in [
        "/healthz", "/readyz", "/v1/predict-raster", "/v1/predict", "/v1/jobs",
        "/v1/jobs/{job_id}", "/v1/jobs/{job_id}/result", "/v1/jobs/{job_id}/retry",
    ]:
        assert route in text
    assert "DELETE /v1/jobs/{job_id}" in text


def test_provider_docs_cover_remote_and_offline_modes():
    text = Path("docs/ACQUISITION_PROVIDERS.md").read_text()
    assert "PC_SDK_SUBSCRIPTION_KEY" in text
    assert "anonymous" in text.lower()
    assert "planetary-computer-grd" in text
    assert "GOOGLE_APPLICATION_CREDENTIALS" in text
    assert "planetary-computer" in text
    assert "earth-engine" in text
    assert "predict-raster" in text
    assert "Secrets are **not accepted in API request bodies**" in text

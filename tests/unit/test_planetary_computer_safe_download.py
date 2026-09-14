from pathlib import Path
from types import SimpleNamespace

import app.acquisition.planetary_computer as mod


class FakeFS:
    def exists(self, path):
        return path.endswith("/manifest.safe")

    def get(self, folder, local, recursive=False):
        # Reproduce fsspec/adlfs behavior where recursive get nests the
        # source basename below the requested local directory.
        product = Path(local) / Path(folder).name
        (product / "annotation").mkdir(parents=True)
        (product / "measurement").mkdir(parents=True)
        (product / "manifest.safe").write_text("manifest")
        (product / "annotation" / "scene.xml").write_text("annotation")
        (product / "measurement" / "scene-vv.tiff").write_bytes(b"TIFF")


class FakeAzure:
    def __init__(self, **kwargs):
        pass
    def exists(self, path):
        return FakeFS().exists(path)
    def get(self, folder, local, recursive=False):
        return FakeFS().get(folder, local, recursive)


def _item():
    return SimpleNamespace(
        assets={"safe-manifest": SimpleNamespace(
            href="https://sentinel1euwest.blob.core.windows.net/s1-grd/GRD/2025/8/14/IW/DV/PRODUCT/manifest.safe"
        )}
    )


def test_download_safe_normalizes_recursive_adlfs_layout(tmp_path, monkeypatch):
    fake_adlfs = SimpleNamespace(AzureBlobFileSystem=FakeAzure)
    fake_pc = SimpleNamespace(sas=SimpleNamespace(
        get_token=lambda *a, **k: SimpleNamespace(token="anonymous")
    ))
    monkeypatch.setattr(mod, "_grd_deps", lambda: (fake_adlfs, fake_pc, None, None, None, None))
    monkeypatch.setattr(mod, "_safe_product_folder", lambda item: (
        "GRD/2025/8/14/IW/DV/PRODUCT", "PRODUCT"
    ))

    product = mod._download_safe(_item(), tmp_path)

    assert product == tmp_path / "PRODUCT"
    assert (product / "manifest.safe").is_file()
    assert (product / "annotation" / "scene.xml").is_file()
    assert (product / "measurement" / "scene-vv.tiff").is_file()
    assert not (product / "PRODUCT").exists()


def test_download_safe_replaces_manifest_only_cache(tmp_path, monkeypatch):
    stale = tmp_path / "PRODUCT"
    stale.mkdir()
    (stale / "manifest.safe").write_text("stale")

    fake_adlfs = SimpleNamespace(AzureBlobFileSystem=FakeAzure)
    fake_pc = SimpleNamespace(sas=SimpleNamespace(
        get_token=lambda *a, **k: SimpleNamespace(token="anonymous")
    ))
    monkeypatch.setattr(mod, "_grd_deps", lambda: (fake_adlfs, fake_pc, None, None, None, None))
    monkeypatch.setattr(mod, "_safe_product_folder", lambda item: (
        "GRD/2025/8/14/IW/DV/PRODUCT", "PRODUCT"
    ))

    product = mod._download_safe(_item(), tmp_path)
    assert (product / "annotation" / "scene.xml").is_file()
    assert (product / "measurement" / "scene-vv.tiff").is_file()

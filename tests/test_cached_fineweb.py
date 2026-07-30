import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "data" / "cached_fineweb10B.py"
SPEC = importlib.util.spec_from_file_location("cached_fineweb_for_test", MODULE_PATH)
downloader = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


class CachedFineWebTest(unittest.TestCase):
    def test_old_huggingface_hub_signature_omits_tqdm_class(self):
        def old_download(repo_id, filename, *, repo_type=None, local_dir=None):
            pass

        with mock.patch.object(downloader, "hf_hub_download", old_download):
            self.assertNotIn("tqdm_class", downloader.download_kwargs())

    def test_new_huggingface_hub_signature_uses_visible_progress(self):
        def new_download(
            repo_id,
            filename,
            *,
            repo_type=None,
            local_dir=None,
            tqdm_class=None,
        ):
            pass

        with mock.patch.object(downloader, "hf_hub_download", new_download):
            self.assertIs(
                downloader.download_kwargs()["tqdm_class"],
                downloader.VisibleDownloadProgress,
            )


if __name__ == "__main__":
    unittest.main()

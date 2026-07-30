import os

from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm


class VisibleDownloadProgress(tqdm):
    """Keep byte progress visible when the downloader is piped through tee."""

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = False
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 0.5)
        super().__init__(*args, **kwargs)


def get(fname, *, file_index, total_files):
    local_dir = os.path.join(os.path.dirname(__file__), "fineweb10B")
    local_path = os.path.join(local_dir, fname)
    prefix = f"[{file_index:02d}/{total_files}]"
    if os.path.exists(local_path):
        size_gib = os.path.getsize(local_path) / 1024**3
        print(f"{prefix} cached:      {fname} ({size_gib:.2f} GiB)", flush=True)
        return

    print(f"{prefix} downloading: {fname}", flush=True)
    hf_hub_download(
        repo_id="kjj0/fineweb10B-gpt2",
        filename=fname,
        repo_type="dataset",
        local_dir=local_dir,
        tqdm_class=VisibleDownloadProgress,
    )
    size_gib = os.path.getsize(local_path) / 1024**3
    print(f"{prefix} ready:       {fname} ({size_gib:.2f} GiB)", flush=True)


def main():
    num_chunks = 50
    total_files = num_chunks + 1
    get("fineweb_val_%06d.bin" % 0, file_index=1, total_files=total_files)
    for i in range(1, num_chunks + 1):
        get(
            "fineweb_train_%06d.bin" % i,
            file_index=i + 1,
            total_files=total_files,
        )


if __name__ == "__main__":
    main()

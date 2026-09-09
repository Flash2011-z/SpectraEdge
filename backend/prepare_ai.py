"""Explicit, photo-free setup: python -m backend.prepare_ai from repo root."""

from backend.ai_cutout import MODEL_MD5, MODEL_NAME, MODEL_PATH, MODEL_URL, _verified_model_path


def download_model(url, output_file, _pooch):
    """Bounded HTTP ranges avoid proxies buffering this entire large asset."""
    import requests

    # Length of the fixed, checksum-pinned upstream ONNX asset.
    expected_size = 972666916
    chunk_size = 8 * 1024 * 1024

    def fetch_chunk(start):
        end = min(start + chunk_size, expected_size) - 1
        with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                          stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{expected_size}":
                raise RuntimeError("Model host did not return the requested byte range. Retry model setup later.")
            content = bytearray()
            for data in response.iter_content(1024 * 1024):
                content.extend(data)
                if len(content) > end - start + 1:
                    raise RuntimeError("Model transfer exceeded the requested size.")
            if len(content) != end - start + 1:
                raise RuntimeError("Model transfer was incomplete. Retry model setup.")
            return content

    # At most eight bounded blocks are in flight, not the whole model in RAM.
    # Writing in range order plus pooch's final checksum verifies reassembly.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool, open(output_file, "wb") as destination:
        received = 0
        for batch in range(0, expected_size, chunk_size * 8):
            starts = range(batch, min(batch + chunk_size * 8, expected_size), chunk_size)
            for content in pool.map(fetch_chunk, starts):
                destination.write(content)
                received += len(content)
                print(f"Model download: {received * 100 // expected_size}%", flush=True)


def main():
    try:
        import pooch
    except ImportError:
        raise SystemExit("Install optional packages first: python -m pip install -r backend/requirements-ai.txt")
    print(f"Preparing {MODEL_NAME}. This downloads pretrained weights, never photographs.", flush=True)
    # pooch downloads to a temporary file and verifies the upstream checksum
    # before promoting it. A failed transfer cannot become a loadable model.
    pooch.retrieve(MODEL_URL, known_hash=f"md5:{MODEL_MD5}",
                   fname=MODEL_PATH.name, path=MODEL_PATH.parent,
                   downloader=download_model, progressbar=False)
    print(f"Verified model ready: {_verified_model_path()}")
    print("The app loads it only when AI-assisted cutout is explicitly requested.")


if __name__ == "__main__":
    main()

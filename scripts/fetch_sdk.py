#!/usr/bin/env python3
"""Install Android SDK packages with parallel range downloads.

dl.google.com throttles per connection (~250 kB/s on some ISPs), so a single
sdkmanager download crawls. This fetches each archive in N concurrent chunks,
verifies sha1 against Google's repository index, and unzips into the SDK.
Re-runnable: finished packages are skipped, finished chunks are reused.
"""
import hashlib
import os
import shutil
import sys
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor

SDK = os.path.expanduser(os.environ.get("ANDROID_HOME", "~/Android/Sdk"))
BASE = "https://dl.google.com/android/repository/"
CONNECTIONS = 16
CHUNK = 2 * 2**20

# (package path, repository index, install dir relative to SDK, top-level dir inside zip)
PACKAGES = [
    ("platform-tools", "repository2-3.xml", "platform-tools", "platform-tools"),
    ("emulator", "repository2-3.xml", "emulator", "emulator"),
    (
        "system-images;android-30;google_apis;x86_64",
        "sys-img/google_apis/sys-img2-4.xml",
        "system-images/android-30/google_apis/x86_64",
        "x86_64",
    ),
]


def local(tag):
    return tag.rsplit("}", 1)[-1]


def find_archive(pkg_path, index):
    """Return (url, size, sha1) of the newest linux/x64 archive for pkg_path."""
    with urllib.request.urlopen(BASE + index, timeout=60) as r:
        root = ET.fromstring(r.read())
    best = None
    for pkg in root.iter():
        if local(pkg.tag) != "remotePackage" or pkg.get("path") != pkg_path:
            continue
        channel = next((e for e in pkg if local(e.tag) == "channelRef"), None)
        if channel is not None and channel.get("ref") != "channel-0":
            continue  # stable only
        rev = tuple(
            int(e.text) for r in pkg if local(r.tag) == "revision"
            for e in r if local(e.tag) in ("major", "minor", "micro")
        )
        for arc in pkg.iter():
            if local(arc.tag) != "archive":
                continue
            fields = {local(e.tag): e for e in arc.iter()}
            if "host-os" in fields and fields["host-os"].text != "linux":
                continue
            if "host-arch" in fields and fields["host-arch"].text != "x64":
                continue
            url = fields["url"].text
            size = int(fields["size"].text)
            sha1 = fields["checksum"].text
            if best is None or rev > best[0]:
                best = (rev, url, size, sha1)
    if best is None:
        sys.exit(f"no linux archive found for {pkg_path}")
    rev, url, size, sha1 = best
    base = BASE + index.rsplit("/", 1)[0] + "/" if "/" in index else BASE
    return base + url, size, sha1


def download(url, size, dest):
    parts_dir = dest + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    ranges = [(i, s, min(s + CHUNK, size) - 1) for i, s in enumerate(range(0, size, CHUNK))]
    done = [0]
    lock = threading.Lock()

    def fetch(item):
        i, start, end = item
        part = os.path.join(parts_dir, f"{i:05d}")
        want = end - start + 1
        if os.path.exists(part) and os.path.getsize(part) == want:
            with lock:
                done[0] += want
            return
        last = None
        for attempt in range(8):
            got = 0
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=30) as r, open(part + ".tmp", "wb") as f:
                    got = 0
                    while chunk := r.read(256 * 1024):
                        f.write(chunk)
                        got += len(chunk)
                        with lock:
                            done[0] += len(chunk)
                if got != want:
                    raise OSError(f"short read {got}/{want}")
                os.replace(part + ".tmp", part)
                return
            except Exception as e:  # retry with backoff, undo partial progress
                with lock:
                    done[0] -= got
                time.sleep(2 * (attempt + 1))
                last = e
        raise RuntimeError(f"chunk {i} failed: {last}")

    t0 = time.time()
    stop = threading.Event()

    def progress():
        while not stop.wait(2):
            el = time.time() - t0
            rate = done[0] / el if el else 0
            eta = (size - done[0]) / rate if rate else 0
            print(f"\r  {done[0]/2**20:7.1f}/{size/2**20:.1f} MB  {rate/2**20:5.2f} MB/s  eta {eta/60:4.1f} min ",
                  end="", flush=True)

    threading.Thread(target=progress, daemon=True).start()
    with ThreadPoolExecutor(CONNECTIONS) as ex:
        list(ex.map(fetch, ranges))
    stop.set()
    print()

    with open(dest, "wb") as out:
        for i, _, _ in ranges:
            with open(os.path.join(parts_dir, f"{i:05d}"), "rb") as f:
                shutil.copyfileobj(f, out)
    shutil.rmtree(parts_dir)


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while b := f.read(2**20):
            h.update(b)
    return h.hexdigest()


def extract(zip_path, top, install_dir):
    """Unzip preserving exec bits; move zip's top dir to install_dir."""
    staging = install_dir + ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            out = z.extract(info, staging)
            mode = (info.external_attr >> 16) & 0o7777
            if (info.external_attr >> 16) & 0o170000 == 0o120000:  # symlink
                target = open(out).read()
                os.remove(out)
                os.symlink(target, out)
            elif mode:
                os.chmod(out, mode)
    shutil.rmtree(install_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(install_dir), exist_ok=True)
    os.replace(os.path.join(staging, top), install_dir)
    shutil.rmtree(staging)


def main():
    cache = os.path.join(SDK, ".downloads")
    os.makedirs(cache, exist_ok=True)
    for pkg_path, index, rel, top in PACKAGES:
        install_dir = os.path.join(SDK, rel)
        marker = os.path.join(install_dir, ".fetch_sdk_ok")
        if os.path.exists(marker):
            print(f"[skip] {pkg_path}")
            continue
        url, size, sha1 = find_archive(pkg_path, index)
        zip_path = os.path.join(cache, os.path.basename(url))
        print(f"[get ] {pkg_path}  ({size/2**20:.0f} MB)  {url}")
        if not (os.path.exists(zip_path) and os.path.getsize(zip_path) == size):
            download(url, size, zip_path)
        if sha1_of(zip_path) != sha1:
            os.remove(zip_path)
            sys.exit(f"sha1 mismatch for {zip_path}, deleted; re-run to retry")
        print(f"[unzip] -> {install_dir}")
        extract(zip_path, top, install_dir)
        open(marker, "w").close()
        os.remove(zip_path)
    print("SDK packages ready.")


if __name__ == "__main__":
    main()

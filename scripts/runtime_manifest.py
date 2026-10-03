"""Resolve pinned x86_64 CPython 3.12 wheel URLs, sizes and upstream hashes."""

import concurrent.futures
import html
import io
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from types import SimpleNamespace

CORE = {
    "einops": "0.8.2",
    "filelock": "3.32.3",
    "fsspec": "2026.7.0",
    "jinja2": "3.1.6",
    "markupsafe": "3.0.3",
    "mpmath": "1.3.0",
    "networkx": "3.6.1",
    "safetensors": "0.8.0",
    "setuptools": "78.1.0",
    "spandrel": "0.4.2",
    "sympy": "1.14.0",
    "typing-extensions": "4.16.0",
}
CUDA = {
    line.split("==")[0]: line.split("==")[1]
    for line in Path("requirements.lock.txt").read_text().splitlines()
    if line.startswith("nvidia-") or line.startswith("triton==")
}


def request(url):
    head = isinstance(url, urllib.request.Request)
    address = url.full_url if head else url
    args = ["curl", "-fsSL", "--connect-timeout", "10", "--max-time", "60"]
    if head:
        args.append("-I")
    result = subprocess.check_output([*args, address])
    if head:
        lengths = re.findall(rb"content-length:\s*(\d+)", result, re.I)
        return SimpleNamespace(headers={"Content-Length": lengths[-1].decode()})
    return io.BytesIO(result)


def wheel(package, version):
    print(f"Resolving {package} {version}", flush=True)
    data = json.load(request(f"https://pypi.org/pypi/{package}/{version}/json"))
    candidates = []
    for item in data["urls"]:
        name = item["filename"]
        if not name.endswith(".whl"):
            continue
        if name.endswith("-none-any.whl"):
            rank = 2
        elif (
            "x86_64" in name
            and "manylinux" in name
            and ("cp312" in name or "abi3" in name or "py3-none" in name)
        ):
            rank = 1 if "cp312" in name else 3
        else:
            continue
        if "abi3" in name:
            match = re.search(r"-cp(\d+)-abi3-", name)
            if match and int(match.group(1)) > 312:
                continue
        candidates.append((rank, name, item))
    if not candidates:
        raise ValueError(f"No compatible wheel for {package}")
    item = sorted(candidates, key=lambda value: (value[0], value[1]))[0][2]
    return {
        "name": package,
        "filename": item["filename"],
        "url": item["url"],
        "sha256": item["digests"]["sha256"],
        "bytes": item["size"],
    }


def torch_wheel(package, version, variant):
    print(f"Resolving {package} {version}+{variant}", flush=True)
    page = request(f"https://download.pytorch.org/whl/{variant}/{package}/").read().decode()
    links = re.findall(r'href="([^"]+)"', page)
    expected = f"{package}-{version}+{variant}-cp312-cp312-manylinux_2_28_x86_64.whl"
    link = next(html.unescape(link) for link in links if expected in urllib.parse.unquote(link))
    url, digest = link.split("#sha256=")
    size = int(request(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])
    return {"name": package, "filename": expected, "url": url, "sha256": digest, "bytes": size}


def main():
    packages = {**CORE, **CUDA, "ncnn": "1.0.20260526"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        resolved = dict(zip(packages, pool.map(lambda pair: wheel(*pair), packages.items())))
    output = {}
    for runtime in ("cuda", "cpu", "vulkan"):
        if runtime == "vulkan":
            wheels = [resolved["ncnn"]]
        else:
            wheels = [resolved[name] for name in CORE]
            if runtime == "cuda":
                wheels += [resolved[name] for name in CUDA]
            variant = "cu128" if runtime == "cuda" else "cpu"
            wheels += [
                torch_wheel("torch", "2.7.1", variant),
                torch_wheel("torchvision", "0.22.1", variant),
            ]
        output[runtime] = {
            "wheels": wheels,
            "download_bytes": sum(item["bytes"] for item in wheels),
        }
    target = Path("src/photo_denoise/runtime-manifest.json")
    target.write_text(json.dumps(output, indent=2) + "\n")
    print({name: round(value["download_bytes"] / 10**6, 1) for name, value in output.items()})


if __name__ == "__main__":
    main()

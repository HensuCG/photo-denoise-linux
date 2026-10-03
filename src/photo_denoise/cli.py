"""Command-line interface and atomic per-photo processing."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import __version__
from .images import EXTENSIONS, family, read_photo, write_photo
from .metadata import copy_metadata, exiftool_command, run_exiftool
from .models import MODELS, Denoiser, cache_directory, download_model, resolve_device


def bounded_float(low: float, high: float):
    def parse(value: str) -> float:
        result = float(value)
        if not low <= result <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return result

    return parse


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="photo-denoise",
        description="Local AI photo denoising with verified metadata preservation",
    )
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("gui", help="Open the desktop application")
    denoise = sub.add_parser(
        "denoise", help="Denoise files or directories; originals are protected"
    )
    denoise.add_argument("inputs", type=Path, nargs="+")
    output = denoise.add_mutually_exclusive_group()
    output.add_argument("-o", "--output", type=Path, help="Output file (one input, same format)")
    output.add_argument(
        "--output-dir", type=Path, help="Batch output directory; preserves directory hierarchy"
    )
    denoise.add_argument("--recursive", action="store_true", help="Include subdirectories")
    denoise.add_argument("--model", choices=MODELS, default="scunet")
    denoise.add_argument(
        "--sigma",
        type=bounded_float(0, 100),
        default=None,
        help="DRUNet noise standard deviation on a 0..255 scale (default: 15)",
    )
    denoise.add_argument(
        "--amount",
        type=bounded_float(0, 1),
        default=1,
        help="Blend amount: 0=original pixels, 1=full denoising (default: 1)",
    )
    denoise.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    denoise.add_argument(
        "--runtime",
        choices=("cuda", "vulkan", "cpu", "auto"),
        default=None,
        help="Inference runtime; --device remains supported for the original CLI",
    )
    denoise.add_argument(
        "--gpu-index", type=int, default=0, help="GPU index within the selected runtime"
    )
    denoise.add_argument("--json-events", action="store_true", help=argparse.SUPPRESS)
    denoise.add_argument(
        "--tile-size", type=int, default=512, help="Maximum tile side in pixels (default: 512)"
    )
    denoise.add_argument(
        "--overlap", type=int, default=32, help="Context around each tile core (default: 32)"
    )
    denoise.add_argument(
        "--jpeg-quality", type=int, choices=range(1, 101), metavar="1..100", default=95
    )
    denoise.add_argument("--suffix", default="-denoised", help="Suffix for generated filenames")
    denoise.add_argument(
        "--overwrite", action="store_true", help="Replace existing outputs; never input files"
    )
    denoise.add_argument(
        "--offline", action="store_true", help="Require cached model; do not download"
    )
    denoise.add_argument("--cache-dir", type=Path, default=cache_directory())
    denoise.add_argument("--quiet", action="store_true")
    fetch = sub.add_parser("download", help="Download and verify official model weights")
    fetch.add_argument("model", choices=(*MODELS, "all"), nargs="?", default="scunet")
    fetch.add_argument("--cache-dir", type=Path, default=cache_directory())
    sub.add_parser("doctor", help="Show runtime, CUDA, metadata tool, and model diagnostics")
    return root


def jobs(args) -> list[tuple[Path, Path]]:
    found: dict[Path, Path] = {}
    for item in args.inputs:
        item = item.expanduser().resolve()
        if not item.exists():
            raise ValueError(f"Input does not exist: {item}")
        if item.is_dir():
            files = sorted(item.rglob("*") if args.recursive else item.iterdir())
            for source in files:
                if source.is_file() and source.suffix.lower() in EXTENSIONS:
                    # Avoid feeding previously generated outputs back into the model.
                    if args.suffix and source.stem.endswith(args.suffix):
                        continue
                    if (
                        args.output_dir
                        and args.output_dir.resolve() != item
                        and (
                            args.output_dir.resolve() == source
                            or args.output_dir.resolve() in source.parents
                        )
                    ):
                        continue
                    relative = source.relative_to(item)
                    if len(args.inputs) > 1:
                        relative = Path(item.name) / relative
                    found.setdefault(source.resolve(), relative)
        elif item.suffix.lower() in EXTENSIONS:
            found.setdefault(item, Path(item.name))
        else:
            raise ValueError(f"Unsupported input format: {item.name}")
    if not found:
        raise ValueError("No supported photos found (JPEG, PNG, TIFF)")
    if args.output and len(found) != 1:
        raise ValueError("--output requires exactly one photo; use --output-dir for batches")
    result = []
    destinations = set()
    sources = set(found)
    for source, relative in found.items():
        if args.output:
            destination = args.output.expanduser().absolute()
        elif args.output_dir:
            destination = args.output_dir.expanduser().absolute() / relative.with_name(
                relative.stem + args.suffix + relative.suffix
            )
        else:
            destination = source.with_name(source.stem + args.suffix + source.suffix)
        if destination.suffix.lower() not in EXTENSIONS or family(source) != family(destination):
            raise ValueError("Output must use the same format as the input to preserve metadata")
        if destination.is_symlink():
            raise ValueError(f"Refusing a symlink output: {destination}")
        if destination.resolve() in sources:
            raise ValueError(f"Refusing to overwrite an input: {destination}")
        if destination.exists() and any(
            os.path.samefile(destination, original) for original in sources
        ):
            raise ValueError(f"Output is a hard link to an input: {destination}")
        if destination.resolve() in destinations:
            raise ValueError(f"Multiple inputs map to the same output: {destination}")
        if destination.exists() and not args.overwrite:
            raise ValueError(
                f"Output already exists: {destination}; use --overwrite to replace outputs"
            )
        destinations.add(destination.resolve())
        result.append((source, destination))
    return result


def process_photo(source: Path, destination: Path, denoiser, args, progress=None) -> None:
    photo = read_photo(source)
    result = (
        photo.rgb.copy()
        if args.amount == 0
        else denoiser.denoise(
            photo.rgb,
            sigma=15 if args.sigma is None else args.sigma,
            amount=args.amount,
            tile_size=args.tile_size,
            overlap=args.overlap,
            progress=progress,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=".photo-denoise-", suffix=destination.suffix, dir=destination.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        write_photo(temporary, photo, result, jpeg_quality=args.jpeg_quality)
        copy_metadata(source, temporary)
        # Flush bytes before publishing. link() provides atomic no-clobber behavior.
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if args.overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def doctor() -> int:
    report = {
        "python": sys.version.split()[0],
        "model_cache": str(cache_directory()),
    }
    from .hardware import detect_gpus

    report["vulkan_devices"] = detect_gpus()
    try:
        import torch

        report.update(
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
            spandrel=importlib.metadata.version("spandrel"),
        )
        if report["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            report.update(gpu=props.name, vram_gib=round(props.total_memory / 2**30, 1))
    except ImportError:
        report["cuda_available"] = False
    try:
        report["exiftool"] = run_exiftool("-ver").stdout.strip()
    except ValueError as error:
        report["exiftool_error"] = str(error)
    report["cached_models"] = [
        name for name, (filename, _) in MODELS.items() if (cache_directory() / filename).is_file()
    ]
    print(json.dumps(report, indent=2))
    return 0 if "exiftool" in report else 1


def main(argv=None) -> int:
    root = parser()
    args = root.parse_args(argv)
    if args.command == "denoise" and threading.current_thread() is threading.main_thread():

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "gui":
            from .gui import main as gui_main

            return gui_main()
        if args.command == "doctor":
            return doctor()
        if args.command == "download":
            for name in MODELS if args.model == "all" else (args.model,):
                print(f"{name}: {download_model(name, args.cache_dir.expanduser())}")
            return 0
        if args.sigma is not None and args.model != "drunet":
            root.error("--sigma is only supported with --model drunet; use --amount with SCUNet")
        if args.tile_size < 64 or args.overlap < 0 or args.overlap * 2 >= args.tile_size:
            root.error(
                "--tile-size must be >=64; --overlap must be >=0 and less than half the tile size"
            )
        if "/" in args.suffix or "\\" in args.suffix:
            root.error("--suffix cannot contain path separators")
        planned = jobs(args)
        exiftool_command()
        denoiser = None

        def event(kind, **values):
            if args.json_events:
                print(json.dumps({"event": kind, **values}), flush=True)

        if args.amount:
            runtime = args.runtime
            if runtime == "auto":
                from .hardware import detect_gpus, recommended_runtime

                runtime = recommended_runtime(detect_gpus())
            device = "cpu" if runtime == "cpu" else args.device
            if runtime != "vulkan":
                device = resolve_device(device)
            if not args.quiet:
                print(f"Loading {args.model} on {device}...", file=sys.stderr)
            denoiser = Denoiser(
                args.model,
                args.cache_dir.expanduser(),
                str(device),
                offline=args.offline,
                runtime=runtime,
                gpu_index=args.gpu_index,
            )
        failures = 0
        for index, (source, destination) in enumerate(planned, 1):
            started = time.monotonic()
            if not args.quiet:
                print(f"[{index}/{len(planned)}] {source.name}...", file=sys.stderr)
            try:
                event("file", index=index, total=len(planned), source=str(source))
                process_photo(
                    source,
                    destination,
                    denoiser,
                    args,
                    progress=lambda done, total: event(
                        "progress", index=index, files=len(planned), done=done, total=total
                    ),
                )
                event(
                    "saved",
                    source=str(source),
                    output=str(destination),
                    seconds=time.monotonic() - started,
                )
                if not args.quiet:
                    print(
                        f"Saved {destination} ({time.monotonic() - started:.1f}s; metadata verified)",
                        file=sys.stderr,
                    )
            except Exception as error:
                failures += 1
                event("error", source=str(source), message=str(error))
                print(f"Error: {source}: {error}", file=sys.stderr)
        return 1 if failures else 0
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

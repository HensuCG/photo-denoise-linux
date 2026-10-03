"""GUI-first launcher with the original CLI and isolated runtime selection."""

import os
import subprocess
import sys
from pathlib import Path


def option_value(args, name):
    value = None
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "gui" or all(Path(arg).exists() for arg in args):
        from .gui import main as gui_main

        return gui_main(args[1:] if args and args[0] == "gui" else args)
    from .runtime_setup import runtime_environment, runtime_installed
    from .settings import load_settings

    if args[0] in ("denoise", "doctor") and not os.environ.get("PHOTO_DENOISE_ACTIVE_RUNTIME"):
        settings = load_settings()
        runtime = settings.get("runtime")
        explicit_runtime = option_value(args, "--runtime")
        explicit_device = option_value(args, "--device")
        if explicit_runtime:
            runtime = explicit_runtime
            if runtime == "auto":
                from .hardware import detect_gpus, recommended_runtime

                runtime = settings.get("runtime") or recommended_runtime(detect_gpus())
                args += ["--runtime", runtime]
        elif explicit_device:
            runtime = "cpu" if explicit_device == "cpu" else "cuda"
            if explicit_device == "auto" and not runtime_installed("cuda"):
                runtime = "cpu"
        elif runtime and args[0] == "denoise":
            args += ["--runtime", runtime]
            if option_value(args, "--gpu-index") is None:
                args += ["--gpu-index", str(settings.get("gpu_index", 0))]
        if runtime in ("cuda", "vulkan", "cpu") and runtime_installed(runtime):
            return subprocess.call(
                [sys.executable, "-m", "photo_denoise.cli", *args], env=runtime_environment(runtime)
            )
    from .cli import main as cli_main

    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

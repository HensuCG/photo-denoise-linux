from photo_denoise import launch, runtime_setup, settings


def test_auto_uses_selected_isolated_runtime(monkeypatch):
    monkeypatch.delenv("PHOTO_DENOISE_ACTIVE_RUNTIME", raising=False)
    monkeypatch.setattr(settings, "load_settings", lambda: {"runtime": "vulkan"})
    monkeypatch.setattr(runtime_setup, "runtime_installed", lambda runtime: runtime == "vulkan")
    monkeypatch.setattr(runtime_setup, "runtime_environment", lambda runtime: {"chosen": runtime})
    calls = []
    monkeypatch.setattr(
        launch.subprocess, "call", lambda command, env: calls.append((command, env)) or 0
    )
    assert launch.main(["denoise", "photo.png", "--runtime", "auto"]) == 0
    assert calls[0][0][-1] == "vulkan"
    assert calls[0][1] == {"chosen": "vulkan"}


def test_equals_option_selects_isolated_cpu(monkeypatch):
    monkeypatch.delenv("PHOTO_DENOISE_ACTIVE_RUNTIME", raising=False)
    monkeypatch.setattr(settings, "load_settings", lambda: {"runtime": "cuda"})
    monkeypatch.setattr(runtime_setup, "runtime_installed", lambda runtime: runtime == "cpu")
    monkeypatch.setattr(runtime_setup, "runtime_environment", lambda runtime: {"chosen": runtime})
    calls = []
    monkeypatch.setattr(
        launch.subprocess, "call", lambda command, env: calls.append((command, env)) or 0
    )
    assert launch.main(["denoise", "photo.png", "--runtime=cpu"]) == 0
    assert calls[0][1] == {"chosen": "cpu"}

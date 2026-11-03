from src.config import settings
from src.core.inference_engine import FaceInferenceEngine


def test_configure_gpu_preloads_runtime_libraries(monkeypatch):
    engine = FaceInferenceEngine()
    preload = []
    monkeypatch.setattr(settings, "gpu_device_id", 0)
    monkeypatch.setattr(
        "src.core.inference_engine.ort.get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        "src.core.inference_engine.ort.preload_dlls",
        lambda **kwargs: preload.append(kwargs),
    )

    providers = engine._configure_providers()

    assert providers[0][0] == "CUDAExecutionProvider"
    assert providers[-1] == "CPUExecutionProvider"
    assert preload == [{}]


def test_configure_cpu_does_not_preload_gpu_libraries(monkeypatch):
    engine = FaceInferenceEngine()
    monkeypatch.setattr(settings, "gpu_device_id", -1)
    monkeypatch.setattr(
        "src.core.inference_engine.ort.preload_dlls",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected GPU preload")),
    )

    assert engine._configure_providers() == ["CPUExecutionProvider"]

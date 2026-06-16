import sys
import types
from pathlib import Path

import numpy as np


fake_sherpa = types.ModuleType("sherpa_onnx")
sys.modules.setdefault("sherpa_onnx", fake_sherpa)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.transcriber import Transcriber


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_pcm_signal_detector_ignores_silence():
    silence = np.zeros(16_000, dtype=np.float32)

    assert_true(
        Transcriber._has_pcm_signal(silence) is False,
        "silent PCM should not be sent to fallback ASR",
    )


def test_pcm_signal_detector_accepts_audible_samples():
    tone = np.sin(np.linspace(0, np.pi * 20, 16_000)).astype(np.float32) * 0.02

    assert_true(
        Transcriber._has_pcm_signal(tone) is True,
        "audible PCM should be eligible for fallback ASR",
    )


if __name__ == "__main__":
    test_pcm_signal_detector_ignores_silence()
    test_pcm_signal_detector_accepts_audible_samples()
    print("transcriber fallback checks ok")

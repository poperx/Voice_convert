"""Separate a mix, then retune vocal timbre, pitch, and random micro-pauses."""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

SR = 44100
N_FFT = 7680
HOP = 1024
DIM_F = 3072
DIM_T = 256
OVERLAP = 0.25
COMPENSATE = 1.021
TRIM = N_FFT // 2
CHUNK = HOP * (DIM_T - 1)
MODEL_NAME = "UVR-MDX-NET-Voc_FT.onnx"
MODEL_URL = (
    "https://huggingface.co/Blane187/all_public_uvr_models/resolve/main/" + MODEL_NAME
)
MODEL_MIN_BYTES = 60_000_000
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

def _runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _bundle_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", _runtime_dir()))
    return Path(__file__).resolve().parent


APP_DIR = _runtime_dir()
BUNDLE_DIR = _bundle_dir()
CACHE_DIR = APP_DIR / ".cache"
CACHE_VERSION = "v3"
_MODEL_CANDIDATES = (
    BUNDLE_DIR / "models" / MODEL_NAME,
    APP_DIR / "models" / MODEL_NAME,
)
MODEL_PATH = next((path for path in _MODEL_CANDIDATES if path.exists()), _MODEL_CANDIDATES[-1])

_WIN = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float64)


@dataclass
class VoiceSettings:
    formant_ratio: float = 0.86
    semitones: float = 0.0
    pause_count: int = 9
    pause_ms: int = 50
    seed: int = 1
    denoise: float = 40
    noise: float = 0
    jitter_cents: float = 0
    jitter_probability: float = 0
    smoothness: float = 45

    def clamped(self) -> "VoiceSettings":
        return VoiceSettings(
            formant_ratio=float(np.clip(self.formant_ratio, 0.55, 1.7)),
            semitones=float(np.clip(self.semitones, -12, 12)),
            pause_count=int(np.clip(self.pause_count, 0, 80)),
            pause_ms=int(np.clip(self.pause_ms, 15, 200)),
            seed=int(np.clip(self.seed, 0, 99999)),
            denoise=float(np.clip(self.denoise, 0, 100)),
            noise=float(np.clip(self.noise, 0, 100)),
            jitter_cents=float(np.clip(self.jitter_cents, 0, 100)),
            jitter_probability=float(np.clip(self.jitter_probability, 0, 100)),
            smoothness=float(np.clip(self.smoothness, 0, 100)),
        )


@dataclass
class Stems:
    source: np.ndarray
    vocals: np.ndarray
    instrumental: np.ndarray
    sample_rate: int = SR

    @property
    def samples(self) -> int:
        return int(self.source.shape[-1])


@dataclass
class RenderResult:
    audio: np.ndarray
    pause_count: int
    sample_rate: int = SR


def _run(cmd: list[str]) -> None:
    subprocess.run(
        cmd,
        check=True,
        creationflags=CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def find_ffmpeg() -> Path:
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(root.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"))
    if not matches:
        raise FileNotFoundError("找不到 ffmpeg。请先安装 FFmpeg，或重新运行安装脚本。")
    return matches[-1]


def find_ffplay() -> Path:
    ffmpeg = find_ffmpeg()
    sibling = ffmpeg.with_name("ffplay.exe")
    if sibling.exists():
        return sibling
    found = shutil.which("ffplay")
    if found:
        return Path(found)
    raise FileNotFoundError("找不到 ffplay，无法在程序内播放预览。")


def cache_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_folder(media: Path) -> Path:
    return CACHE_DIR / f"{cache_key(media)}_{CACHE_VERSION}"


def stems_ready(media: Path) -> bool:
    folder = cache_folder(media)
    return (folder / "vocals.npy").exists() and (folder / "inst.npy").exists() and (folder / "source.wav").exists()


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 2 or handle.getframerate() != SR or handle.getsampwidth() != 2:
            raise ValueError(f"需要 44100 Hz 立体声 16-bit WAV：{path}")
        frames = handle.getnframes()
        pcm = np.frombuffer(handle.readframes(frames), dtype=np.int16)
    audio = pcm.reshape(-1, 2).T.astype(np.float32) / 32768.0
    return np.ascontiguousarray(audio)


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped.T * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def _frame_s16(frame) -> np.ndarray:
    array = np.asarray(frame.to_ndarray())
    if array.dtype != np.int16:
        array = np.clip(array, -1.0, 1.0)
        array = (array * 32767.0).astype(np.int16)
    if array.ndim == 2 and array.shape[0] == 2:
        array = array.T.reshape(-1)
    else:
        array = array.reshape(-1)
    return np.ascontiguousarray(array)


def _extract_wav_av(media: Path, dest: Path) -> None:
    import av

    container = av.open(str(media))
    try:
        if not container.streams.audio:
            raise RuntimeError("文件里没有音频轨。")
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=SR)
        with wave.open(str(dest), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(SR)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    handle.writeframes(_frame_s16(out).tobytes())
            for out in resampler.resample(None):
                handle.writeframes(_frame_s16(out).tobytes())
    finally:
        container.close()
    if not dest.exists() or dest.stat().st_size < SR:
        raise RuntimeError("提取到的音频过短。")


def _extract_wav_ffmpeg(media: Path, dest: Path) -> None:
    ffmpeg = find_ffmpeg()
    _run(
        [
            str(ffmpeg),
            "-y",
            "-i",
            str(media),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(SR),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    )


def extract_wav(media: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    try:
        _extract_wav_av(media, dest)
        return
    except Exception as exc:
        errors.append(f"内置解码失败：{exc}")
        if dest.exists():
            dest.unlink()
    try:
        _extract_wav_ffmpeg(media, dest)
    except Exception as exc:
        errors.append(f"ffmpeg 失败：{exc}")
        raise RuntimeError("无法提取音频。" + " ".join(errors)) from exc


def ensure_model(progress=None) -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MODEL_MIN_BYTES:
        return MODEL_PATH
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.0, "正在下载人声分离模型（约 64 MB）…")
    curl = shutil.which("curl") or "curl.exe"
    subprocess.run(
        [
            curl,
            "-L",
            "--retry",
            "8",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "-o",
            str(MODEL_PATH),
            MODEL_URL,
        ],
        check=True,
        creationflags=CREATE_NO_WINDOW,
    )
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < MODEL_MIN_BYTES:
        raise RuntimeError("人声分离模型下载不完整。")
    return MODEL_PATH


def _stft(x: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), (pad, pad), mode="reflect")
    frames = 1 + (len(padded) - N_FFT) // HOP
    view = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP][:frames]
    return np.fft.rfft(view * _WIN, axis=-1).T


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    count = spec.shape[1]
    time_frames = np.fft.irfft(spec.T, n=N_FFT) * _WIN
    out_len = N_FFT + HOP * (count - 1)
    mixed = np.zeros(out_len, dtype=np.float64)
    weight = np.zeros(out_len, dtype=np.float64)
    for index in range(count):
        start = index * HOP
        mixed[start : start + N_FFT] += time_frames[index]
        weight[start : start + N_FFT] += _WIN ** 2
    usable = weight > 1e-8
    mixed[usable] /= weight[usable]
    pad = N_FFT // 2
    mixed = mixed[pad : pad + length]
    if len(mixed) < length:
        mixed = np.pad(mixed, (0, length - len(mixed)))
    return mixed[:length]


def _wave_to_model(wave: np.ndarray) -> np.ndarray:
    left = _stft(wave[0])
    right = _stft(wave[1])
    packed = np.stack([left.real, left.imag, right.real, right.imag], axis=0).astype(np.float32)
    if packed.shape[-1] < DIM_T:
        packed = np.pad(packed, ((0, 0), (0, 0), (0, DIM_T - packed.shape[-1])))
    elif packed.shape[-1] > DIM_T:
        packed = packed[:, :, :DIM_T]
    return packed[None, :, :DIM_F, :]


def _model_to_wave(spec: np.ndarray, length: int) -> np.ndarray:
    if spec.ndim == 4:
        spec = spec[0]
    bins = N_FFT // 2 + 1
    full = np.zeros((4, bins, spec.shape[-1]), dtype=np.float32)
    use = min(DIM_F, spec.shape[1], bins)
    full[:, :use, :] = spec[:, :use, :]
    left = _istft(full[0] + 1j * full[1], length)
    right = _istft(full[2] + 1j * full[3], length)
    return np.stack([left, right], axis=0).astype(np.float32)


def separate_mix(mix: np.ndarray, model_path: Path, progress=None) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 2)
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    peak = float(np.max(np.abs(mix)))
    scale = 0.9 / peak if peak > 1e-8 else 1.0
    proc = (mix * scale).astype(np.float32)
    gen_size = CHUNK - 2 * TRIM
    tail = gen_size + TRIM - (proc.shape[-1] % gen_size)
    mixture = np.concatenate(
        [np.zeros((2, TRIM), np.float32), proc, np.zeros((2, tail), np.float32)],
        axis=1,
    )
    step = int((1 - OVERLAP) * CHUNK)
    result = np.zeros((2, mixture.shape[-1]), np.float32)
    divider = np.zeros(mixture.shape[-1], np.float32)
    total = max(1, (mixture.shape[-1] + step - 1) // step)
    done = 0
    for start in range(0, mixture.shape[-1], step):
        end = min(start + CHUNK, mixture.shape[-1])
        actual = end - start
        part = mixture[:, start:end]
        if part.shape[-1] < CHUNK:
            part = np.pad(part, ((0, 0), (0, CHUNK - part.shape[-1])))
        spec = _wave_to_model(part)
        spec[:, :, :3, :] = 0
        # One forward pass. A second inverted pass costs twice the CPU and RAM
        # and was a source of brittle high-frequency residue.
        pred = session.run(None, {input_name: spec})[0]
        wav = _model_to_wave(pred, CHUNK)
        window = np.hanning(actual).astype(np.float32)
        wav[:, :actual] *= window
        result[:, start:end] += wav[:, :actual]
        divider[start:end] += window
        done += 1
        if progress and (done == 1 or done % 2 == 0 or done == total):
            progress(done / total, f"正在分离人声 {done}/{total}")

    vocals = result[:, TRIM : TRIM + proc.shape[-1]] / np.maximum(divider[TRIM : TRIM + proc.shape[-1]], 1e-8)
    vocals = (vocals / scale).astype(np.float32)
    instrumental = (mix - vocals * COMPENSATE).astype(np.float32)
    del session
    return vocals, instrumental


def prepare_stems(media: Path, progress=None) -> Stems:
    media = media.resolve()
    if not media.exists():
        raise FileNotFoundError(f"找不到输入文件：{media}")
    folder = cache_folder(media)
    source_wav = folder / "source.wav"
    vocals_path = folder / "vocals.npy"
    inst_path = folder / "inst.npy"
    if vocals_path.exists() and inst_path.exists() and source_wav.exists():
        if progress:
            progress(1.0, "已使用缓存的人声和伴奏")
        return Stems(read_wav(source_wav), np.load(vocals_path), np.load(inst_path))

    if progress:
        progress(0.02, "正在从文件提取声音…")
    extract_wav(media, source_wav)
    mix = read_wav(source_wav)
    model = ensure_model(progress)
    if progress:
        progress(0.08, "开始分离人声…")

    def stem_progress(frac: float, message: str) -> None:
        if progress:
            progress(0.08 + 0.9 * frac, message)

    vocals, instrumental = separate_mix(mix, model, stem_progress)
    n = min(mix.shape[-1], vocals.shape[-1], instrumental.shape[-1])
    mix = mix[:, :n]
    vocals = vocals[:, :n]
    instrumental = instrumental[:, :n]
    np.save(vocals_path, vocals)
    np.save(inst_path, instrumental)
    if mix.shape[-1] != read_wav(source_wav).shape[-1]:
        write_wav(source_wav, mix)
    if progress:
        progress(1.0, "人声分离完成")
    return Stems(mix, vocals, instrumental)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) + 1e-12)


def estimate_f0(mono: np.ndarray, sr: int = SR) -> float:
    frame = int(0.04 * sr)
    hop = int(0.02 * sr)
    min_lag = int(sr / 900)
    max_lag = int(sr / 70)
    window = np.hanning(frame)
    values = []
    for start in range(0, max(0, len(mono) - frame), hop * 2):
        seg = mono[start : start + frame]
        if rms(seg) < 0.015:
            continue
        seg = seg * window
        corr = np.correlate(seg, seg, mode="full")[frame - 1 :]
        if corr[0] <= 1e-8:
            continue
        zone = corr[min_lag:max_lag]
        if zone.size == 0:
            continue
        lag = int(np.argmax(zone)) + min_lag
        if corr[lag] / corr[0] < 0.3:
            continue
        values.append(sr / lag)
    if not values:
        return 0.0
    return float(np.median(values))


def spectral_centroid(mono: np.ndarray, sr: int = SR) -> float:
    if len(mono) < 4096:
        return 0.0
    window = np.hanning(4096)
    freqs = np.fft.rfftfreq(4096, 1 / sr)
    step = 4096 * 8
    cents = []
    for start in range(0, len(mono) - 4096, step):
        frame = mono[start : start + 4096]
        if rms(frame) < 0.01:
            continue
        mag = np.abs(np.fft.rfft(frame * window))
        cents.append(float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12)))
    if not cents:
        return 0.0
    return float(np.median(cents))


def _biquad_peaking(sr: int, freq: float, q: float, gain_db: float) -> np.ndarray:
    amp = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w = np.cos(w0)
    b0 = 1 + alpha * amp
    b1 = -2 * cos_w
    b2 = 1 - alpha * amp
    a0 = 1 + alpha / amp
    a1 = -2 * cos_w
    a2 = 1 - alpha / amp
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0], dtype=np.float64)


def _biquad_lowshelf(sr: int, freq: float, gain_db: float, q: float = 0.7) -> np.ndarray:
    amp = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w = np.cos(w0)
    two_sqrt = 2 * np.sqrt(amp)
    b0 = amp * ((amp + 1) - (amp - 1) * cos_w + two_sqrt * alpha)
    b1 = 2 * amp * ((amp - 1) - (amp + 1) * cos_w)
    b2 = amp * ((amp + 1) - (amp - 1) * cos_w - two_sqrt * alpha)
    a0 = (amp + 1) + (amp - 1) * cos_w + two_sqrt * alpha
    a1 = -2 * ((amp - 1) + (amp + 1) * cos_w)
    a2 = (amp + 1) + (amp - 1) * cos_w - two_sqrt * alpha
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0], dtype=np.float64)


def color_eq(channel: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    # ratio < 1 warms the voice; ratio > 1 brightens it. 1 leaves the EQ flat.
    amount = float(np.clip((1.0 - ratio) / 0.35, -1.0, 1.0))
    if abs(amount) < 0.04:
        return channel.astype(np.float32)
    sections = np.stack(
        [
            _biquad_lowshelf(sr, 280, 4.5 * amount),
            _biquad_peaking(sr, 1400, 1.0, 2.4 * amount),
            _biquad_peaking(sr, 3600, 1.0, -5.0 * amount),
        ]
    )
    return signal.sosfiltfilt(sections, channel).astype(np.float32)


def _praat_median(sound) -> float | None:
    from parselmouth.praat import call

    try:
        pitch = call(sound, "To Pitch", 0.01, 70, 1100)
        value = float(call(pitch, "Get quantile", 0, 0, 0.5, "Hertz"))
    except Exception:
        return None
    if not np.isfinite(value) or value < 60 or value > 1200:
        return None
    return value


def _praat_segment(segment: np.ndarray, sr: int, ratio: float, semitones: float) -> np.ndarray:
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(np.asarray(segment, dtype=np.float64), sampling_frequency=sr)
    new_median = 0.0
    ceiling = 1100.0
    if abs(semitones) >= 0.05:
        median = _praat_median(sound)
        if median is None:
            median = estimate_f0(segment, sr)
        if median > 0:
            new_median = median * (2.0 ** (semitones / 12.0))
            ceiling = max(1100.0, new_median * 1.9)
    # A new pitch median of 0 keeps the original melody.
    changed = call(sound, "Change gender", 60, ceiling, float(ratio), float(new_median), 1.0, 1.0)
    values = np.asarray(changed.values, dtype=np.float64).reshape(-1)
    if len(values) < len(segment):
        values = np.pad(values, (0, len(segment) - len(values)))
    return values[: len(segment)]


def _mostly_unvoiced(segment: np.ndarray, sr: int) -> bool:
    if rms(segment) < 0.008:
        return True
    # Praat turns noise-like stretches into electrical buzz, and they are slow.
    return float(np.mean(_voicing_mask(np.asarray(segment, dtype=np.float32), sr))) < 0.05


def convert_channel(channel: np.ndarray, sr: int, ratio: float, semitones: float) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float64)
    if abs(ratio - 1.0) < 1e-3 and abs(semitones) < 0.05:
        return channel.astype(np.float32)
    chunk = int(22 * sr)
    overlap = int(0.4 * sr)
    try:
        if len(channel) <= chunk:
            if _mostly_unvoiced(channel, sr):
                changed = channel
            else:
                changed = _praat_segment(channel, sr, ratio, semitones)
        else:
            mixed = np.zeros(len(channel), dtype=np.float64)
            weight = np.zeros(len(channel), dtype=np.float64)
            pos = 0
            while pos < len(channel):
                end = min(len(channel), pos + chunk)
                segment = channel[pos:end]
                if _mostly_unvoiced(segment, sr):
                    piece = segment
                else:
                    piece = _praat_segment(segment, sr, ratio, semitones)
                fade = min(overlap, max(1, len(piece) // 5))
                envelope = np.ones(len(piece), dtype=np.float64)
                if pos > 0:
                    envelope[:fade] = np.linspace(0, 1, fade)
                if end < len(channel):
                    envelope[-fade:] = np.linspace(1, 0, fade)
                mixed[pos:end] += piece * envelope
                weight[pos:end] += envelope
                if end >= len(channel):
                    break
                pos = end - overlap
            changed = mixed / np.maximum(weight, 1e-8)
    except Exception:
        changed = _cepstral_formant(channel, ratio)
        if abs(semitones) >= 0.05:
            changed = _resample_pitch(changed, sr, semitones)
    return color_eq(changed.astype(np.float32), sr, ratio)


def _cepstral_formant(channel: np.ndarray, ratio: float) -> np.ndarray:
    n_fft = 2048
    hop = 256
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float64)
    pad = n_fft // 2
    padded = np.pad(channel.astype(np.float64), (pad, pad), mode="reflect")
    count = 1 + (len(padded) - n_fft) // hop
    view = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:count]
    spec = np.fft.rfft(view * window, axis=-1).T
    mag = np.abs(spec) + 1e-8
    cep = np.fft.irfft(np.log(mag), n=n_fft, axis=0)
    lift = np.zeros_like(cep)
    cutoff = 36
    lift[:cutoff] = cep[:cutoff]
    lift[-cutoff + 1 :] = cep[-cutoff + 1 :]
    env = np.maximum(np.exp(np.fft.rfft(lift, n=n_fft, axis=0).real), 1e-8)
    bins = np.arange(mag.shape[0])
    src = np.clip(bins / ratio, 0, mag.shape[0] - 1)
    shifted = np.vstack([np.interp(src, bins, env[:, index]) for index in range(env.shape[1])]).T
    new_spec = (mag / env) * shifted * np.exp(1j * np.angle(spec))
    time_frames = np.fft.irfft(new_spec.T, n=n_fft) * window
    out_len = n_fft + hop * (count - 1)
    mixed = np.zeros(out_len, dtype=np.float64)
    weight = np.zeros(out_len, dtype=np.float64)
    for index in range(count):
        start = index * hop
        mixed[start : start + n_fft] += time_frames[index]
        weight[start : start + n_fft] += window ** 2
    usable = weight > 1e-8
    mixed[usable] /= weight[usable]
    mixed = mixed[pad : pad + len(channel)]
    if len(mixed) < len(channel):
        mixed = np.pad(mixed, (0, len(channel) - len(mixed)))
    return mixed[: len(channel)]


def _resample_pitch(channel: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Pitch shift used only if Praat is unavailable. Resample changes pitch, atempo restores length."""
    import tempfile

    rate = 2.0 ** (semitones / 12.0)
    pitched_len = max(8, int(round(len(channel) / rate)))
    pitched = signal.resample(channel, pitched_len).astype(np.float32)
    folder = Path(tempfile.mkdtemp(prefix="pitch_"))
    src = folder / "in.wav"
    dst = folder / "out.wav"
    write_wav(src, np.stack([pitched, pitched]))
    factors: list[float] = []
    remaining = rate
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    filter_chain = ",".join(f"atempo={factor:.6f}" for factor in factors)
    _run([str(find_ffmpeg()), "-y", "-i", str(src), "-af", filter_chain, "-ar", str(sr), str(dst)])
    restored = read_wav(dst)[0]
    if len(restored) < len(channel):
        restored = np.pad(restored, (0, len(channel) - len(restored)))
    return restored[: len(channel)]


def _stft_mag(channel: np.ndarray, n_fft: int = 1024, hop: int = 512):
    window = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    padded = np.pad(channel.astype(np.float32), (pad, pad))
    count = 1 + max(0, (len(padded) - n_fft) // hop)
    view = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop][:count]
    spec = np.fft.rfft(view * window, axis=-1)
    return spec, window, pad, hop


def _istft_mag(spec: np.ndarray, window: np.ndarray, pad: int, hop: int, length: int) -> np.ndarray:
    n_fft = len(window)
    frames = np.fft.irfft(spec, n=n_fft).astype(np.float32) * window
    out_len = n_fft + hop * (spec.shape[0] - 1)
    mixed = np.zeros(out_len, dtype=np.float32)
    weight = np.zeros(out_len, dtype=np.float32)
    for index in range(spec.shape[0]):
        start = index * hop
        mixed[start : start + n_fft] += frames[index]
        weight[start : start + n_fft] += window ** 2
    usable = weight > 1e-8
    mixed[usable] /= weight[usable]
    mixed = mixed[pad : pad + length]
    if len(mixed) < length:
        mixed = np.pad(mixed, (0, length - len(mixed)))
    return mixed[:length]


def spectral_denoise(channel: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Subtract a noise profile taken from the quietest frames. amount is 0–100."""
    if amount <= 0.5 or len(channel) < 2048:
        return np.asarray(channel, dtype=np.float32)
    strength = float(np.clip(amount, 0, 100)) / 100.0
    spec, window, pad, hop = _stft_mag(channel)
    mag = np.abs(spec).astype(np.float32)
    energy = np.mean(mag, axis=1)
    quiet_cut = np.percentile(energy, 18)
    quiet = mag[energy <= quiet_cut]
    if quiet.shape[0] < 4:
        quiet = mag
    noise = np.median(quiet, axis=0)
    # Keep a floor so vowels are not turned into underwater artifacts.
    floor = 1.0 - 0.85 * strength
    gain = (mag - (strength * 1.15) * noise) / (mag + 1e-8)
    gain = np.clip(gain, floor, 1.0).astype(np.float32)
    if gain.shape[0] >= 3:
        padded = np.pad(gain, ((1, 1), (0, 0)), mode="edge")
        gain = padded[:-2] * 0.25 + padded[1:-1] * 0.5 + padded[2:] * 0.25
    if gain.shape[1] >= 3:
        padded = np.pad(gain, ((0, 0), (1, 1)), mode="edge")
        gain = padded[:, :-2] * 0.2 + padded[:, 1:-1] * 0.6 + padded[:, 2:] * 0.2
    cleaned = _istft_mag(spec * gain, window, pad, hop, len(channel))
    return cleaned.astype(np.float32)


def _voicing_mask(channel: np.ndarray, sr: int) -> np.ndarray:
    """1 where the frame is loud and tonal. Praat turns the rest into electrical buzz."""
    frame = 1024
    hop = 256
    if len(channel) < frame:
        return np.ones(len(channel), dtype=np.float32)
    window = np.hanning(frame).astype(np.float32)
    view = np.lib.stride_tricks.sliding_window_view(channel.astype(np.float32), frame)[::hop]
    windowed = view * window
    mag = np.abs(np.fft.rfft(windowed, axis=-1)) + 1e-8
    flat = np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)
    energy = np.sqrt(np.mean(windowed * windowed, axis=1) + 1e-12)
    reference = float(np.percentile(energy, 80)) + 1e-8
    loud = np.clip(energy / reference, 0, 1)
    tonal = np.clip((0.62 - flat) / 0.42, 0, 1)
    frames = (loud * tonal).astype(np.float32)
    positions = np.clip(np.arange(len(channel)) / hop, 0, len(frames) - 1)
    left = positions.astype(np.int32)
    right = np.minimum(left + 1, len(frames) - 1)
    frac = positions - left
    mask = frames[left] * (1 - frac) + frames[right] * frac
    kernel = max(1, int(0.02 * sr))
    mask = np.convolve(mask, np.ones(kernel) / kernel, mode="same")
    return np.clip(mask, 0, 1).astype(np.float32)


def _protect_unvoiced(dry: np.ndarray, wet: np.ndarray, sr: int) -> np.ndarray:
    mask = _voicing_mask(dry, sr)
    n = min(len(dry), len(wet), len(mask))
    mixed = dry[:n] * (1.0 - mask[:n]) + wet[:n] * mask[:n]
    if len(mixed) < len(dry):
        mixed = np.pad(mixed, (0, len(dry) - len(mixed)))
    return mixed[: len(dry)].astype(np.float32)


def _highpass(channel: np.ndarray, sr: int, freq: float = 75.0) -> np.ndarray:
    sos = signal.butter(2, freq / (sr / 2), btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, channel).astype(np.float32)


def apply_smoothness(dry: np.ndarray, wet: np.ndarray, amount: float) -> np.ndarray:
    """Soften the brittle top end Praat adds, then blend a little of the dry vocal back."""
    if amount <= 0.5:
        return np.asarray(wet, dtype=np.float32)
    level = float(np.clip(amount, 0, 100)) / 100.0
    n = min(len(dry), len(wet))
    dry = dry[:n]
    wet = wet[:n]
    cutoff = 4200 - 1800 * level
    sos = signal.butter(2, max(800.0, cutoff) / (SR / 2), btype="lowpass", output="sos")
    low = signal.sosfiltfilt(sos, wet)
    high = wet - low
    softened = low + high * (1.0 - 0.8 * level)
    mixed = softened * (1.0 - 0.28 * level) + dry * (0.28 * level)
    return mixed.astype(np.float32)


def apply_jitter(channel: np.ndarray, sr: int, cents: float, probability: float, seed: int) -> np.ndarray:
    if cents < 1 or probability <= 0.5 or len(channel) < sr // 5:
        return np.asarray(channel, dtype=np.float32)
    grain = int(0.046 * sr)
    hop = grain // 2
    window = np.hanning(grain).astype(np.float32)
    rng = np.random.default_rng(seed + 17)
    mixed = np.zeros(len(channel) + grain, dtype=np.float32)
    weight = np.zeros(len(channel) + grain, dtype=np.float32)
    chance = float(np.clip(probability, 0, 100)) / 100.0
    span = float(np.clip(cents, 0, 100))
    for pos in range(0, len(channel) - grain, hop):
        piece = channel[pos : pos + grain].astype(np.float32)
        if rng.random() < chance:
            rate = 2.0 ** (float(rng.uniform(-span, span)) / 1200.0)
            idx = np.minimum(np.arange(grain) * rate, grain - 1.001)
            piece = np.interp(idx, np.arange(grain), piece).astype(np.float32)
        mixed[pos : pos + grain] += piece * window
        weight[pos : pos + grain] += window
    usable = weight[: len(channel)] > 1e-6
    out = channel.astype(np.float32).copy()
    out[usable] = mixed[: len(channel)][usable] / weight[: len(channel)][usable]
    return out


def add_breath(channel: np.ndarray, sr: int, amount: float, seed: int) -> np.ndarray:
    """Band-limited breath noise that follows the vocal. Not mains hum."""
    if amount <= 0.5:
        return np.asarray(channel, dtype=np.float32)
    rng = np.random.default_rng(seed + 99)
    white = rng.standard_normal(len(channel)).astype(np.float32)
    breath = signal.lfilter([0.04], [1.0, -0.96], white)
    sos = signal.butter(2, [500 / (sr / 2), 5500 / (sr / 2)], btype="band", output="sos")
    breath = signal.sosfilt(sos, breath).astype(np.float32)
    hop = 1024
    kernel = np.ones(hop, dtype=np.float32) / hop
    env = np.sqrt(np.convolve(channel.astype(np.float32) ** 2, kernel, mode="same") + 1e-8)
    peak = float(np.max(env)) + 1e-8
    env /= peak
    gain = (float(np.clip(amount, 0, 100)) / 100.0) * 0.7
    return (channel + breath * env * gain).astype(np.float32)


def convert_vocals(vocals: np.ndarray, sr: int, settings: VoiceSettings, progress=None) -> np.ndarray:
    settings = settings.clamped()
    if vocals.shape[0] == 1:
        original = vocals[0].astype(np.float32)
        side = np.zeros_like(original)
    else:
        original = ((vocals[0] + vocals[1]) * 0.5).astype(np.float32)
        side = ((vocals[0] - vocals[1]) * 0.5).astype(np.float32)
    if progress:
        progress(0.05, "正在降噪…")
    mid = spectral_denoise(original, sr, settings.denoise)
    if settings.denoise > 0.5 and rms(side) > 1e-5:
        side = spectral_denoise(side, sr, settings.denoise * 0.5)
    needs_retune = abs(settings.formant_ratio - 1.0) >= 1e-3 or abs(settings.semitones) >= 0.05
    if progress:
        progress(0.25, "正在调整音色和音高…" if needs_retune else "音色和音高保持原样")
    if needs_retune:
        wet = convert_channel(mid, sr, settings.formant_ratio, settings.semitones)
        wet = _protect_unvoiced(mid, wet, sr)
    else:
        wet = mid
    if progress:
        progress(0.72, "正在平滑、抖动并混合噪声…")
    wet = apply_smoothness(mid, wet, settings.smoothness)
    wet = apply_jitter(wet, sr, settings.jitter_cents, settings.jitter_probability, settings.seed)
    wet = _highpass(wet, sr, 75)
    before = rms(original)
    after = rms(wet)
    if after > 1e-6:
        wet *= before / after
    # Breath is added after loudness match so the noise slider is not scaled away.
    wet = add_breath(wet, sr, settings.noise, settings.seed)
    left = wet + side
    right = wet - side if vocals.shape[0] > 1 else wet
    return np.stack([left, right], axis=0).astype(np.float32)


def choose_pauses(samples: int, sr: int, settings: VoiceSettings, vocal: np.ndarray) -> list[int]:
    settings = settings.clamped()
    if settings.pause_count <= 0 or samples < sr:
        return []
    pause = int(settings.pause_ms * sr / 1000)
    margin = int(0.35 * sr)
    min_gap = pause + int(0.28 * sr)
    hop = int(0.05 * sr)
    mono = np.mean(vocal, axis=0)
    level = rms(mono) * 0.35
    candidates = []
    for start in range(margin, max(margin + 1, samples - margin - pause), hop):
        if rms(mono[start : start + hop]) >= level:
            candidates.append(start)
    if len(candidates) < settings.pause_count:
        candidates = list(range(margin, max(margin + 1, samples - margin - pause), hop))
    if not candidates:
        return []
    order = np.random.default_rng(settings.seed).permutation(len(candidates))
    chosen: list[int] = []
    for index in order:
        pos = int(candidates[int(index)])
        if all(abs(pos - prev) >= min_gap for prev in chosen):
            chosen.append(pos)
        if len(chosen) >= settings.pause_count:
            break
    chosen.sort()
    return chosen


def apply_pauses(vocal: np.ndarray, sr: int, positions: list[int], pause_ms: int) -> np.ndarray:
    if not positions:
        return vocal
    out = vocal.copy()
    pause = int(pause_ms * sr / 1000)
    fade = max(1, int(0.01 * sr))
    for start in positions:
        end = min(out.shape[-1], start + pause)
        region = end - start
        if region <= fade * 2:
            continue
        envelope = np.zeros(region, dtype=np.float32)
        envelope[:fade] = np.linspace(1, 0, fade, dtype=np.float32)
        envelope[-fade:] = np.linspace(0, 1, fade, dtype=np.float32)
        out[:, start:end] *= envelope
    return out


def match_loudness(audio: np.ndarray, reference: np.ndarray) -> np.ndarray:
    target = rms(reference)
    current = rms(audio)
    scaled = audio * (target / current)
    threshold = 0.72
    ceiling = 0.98
    magnitude = np.abs(scaled)
    sign = np.sign(scaled)
    over = np.maximum(magnitude - threshold, 0)
    compressed = np.minimum(threshold + over / 6.0, ceiling)
    limited = sign * np.where(magnitude > threshold, compressed, magnitude)
    peak = float(np.max(np.abs(limited)))
    if peak > ceiling:
        limited *= ceiling / peak
    return limited.astype(np.float32)


def render(
    stems: Stems,
    settings: VoiceSettings,
    start: int | None = None,
    end: int | None = None,
    progress=None,
) -> RenderResult:
    settings = settings.clamped()
    total = stems.samples
    start = 0 if start is None else max(0, start)
    end = total if end is None else min(total, end)
    if end <= start:
        raise ValueError("预览范围为空")
    margin = int(0.45 * SR)
    region_start = max(0, start - margin)
    region_end = min(total, end + margin)
    changed = convert_vocals(stems.vocals[:, region_start:region_end], SR, settings, progress)
    positions = choose_pauses(total, SR, settings, stems.vocals)
    local_positions = [pos - region_start for pos in positions if region_start <= pos < region_end]
    paused = apply_pauses(changed, SR, local_positions, settings.pause_ms)
    vocal = paused[:, start - region_start : end - region_start]
    instrumental = stems.instrumental[:, start:end]
    reference = stems.source[:, start:end]
    n = min(vocal.shape[-1], instrumental.shape[-1], reference.shape[-1])
    mixed = match_loudness(instrumental[:, :n] + vocal[:, :n], reference[:, :n])
    inside = sum(1 for pos in positions if start <= pos < end)
    return RenderResult(mixed, inside if (start, end) != (0, total) else len(positions))


def preview_range(stems: Stems, settings: VoiceSettings, seconds: float = 20.0) -> tuple[int, int]:
    window = int(seconds * SR)
    total = stems.samples
    if total <= window:
        return 0, total
    positions = choose_pauses(total, SR, settings.clamped(), stems.vocals)
    if positions:
        center = positions[len(positions) // 2]
        start = max(0, min(total - window, center - window // 3))
        return start, start + window
    mono = np.mean(np.abs(stems.vocals), axis=0)
    cumulative = np.cumsum(mono * mono)
    hop = int(0.5 * SR)
    best_start = 0
    best_energy = -1.0
    for start in range(0, total - window, hop):
        energy = float(cumulative[start + window] - cumulative[start])
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return best_start, best_start + window


def stop_playback() -> None:
    if os.name != "nt":
        return
    winmm = ctypes.windll.winmm
    winmm.mciSendStringW("stop voiceprev", None, 0, None)
    winmm.mciSendStringW("close voiceprev", None, 0, None)


def play_wav(path: Path) -> None:
    stop_playback()
    if os.name != "nt":
        raise RuntimeError("当前系统不能在程序内播放预览。")
    winmm = ctypes.windll.winmm
    error = winmm.mciSendStringW(f'open "{path}" type waveaudio alias voiceprev', None, 0, None)
    if error:
        raise RuntimeError(f"无法播放预览（错误 {error}）。")
    winmm.mciSendStringW("play voiceprev", None, 0, None)


def _export_mp3_av(audio: np.ndarray, dest: Path, sr: int) -> None:
    import av

    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    audio = np.ascontiguousarray(audio[:2])
    container = av.open(str(dest), mode="w", format="mp3")
    try:
        stream = container.add_stream("libmp3lame", rate=sr)
        stream.bit_rate = 320_000
        frame_size = 1152
        pts = 0
        total = int(audio.shape[1])
        for start in range(0, max(total, 1), frame_size):
            chunk = audio[:, start : start + frame_size]
            if chunk.shape[1] < frame_size:
                chunk = np.pad(chunk, ((0, 0), (0, frame_size - chunk.shape[1])))
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk), format="fltp", layout="stereo")
            frame.sample_rate = sr
            frame.pts = pts
            pts += frame_size
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    finally:
        container.close()
    if not dest.exists() or dest.stat().st_size < 256:
        raise RuntimeError("MP3 编码结果为空。")


def _export_mp3_ffmpeg(audio: np.ndarray, dest: Path, sr: int) -> None:
    temp = dest.with_suffix(".tmp.wav")
    write_wav(temp, audio, sr)
    try:
        _run(
            [
                str(find_ffmpeg()),
                "-y",
                "-i",
                str(temp),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "320k",
                str(dest),
            ]
        )
    finally:
        if temp.exists():
            temp.unlink()


def export_mp3(audio: np.ndarray, dest: Path, sr: int = SR) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _export_mp3_av(audio, dest, sr)
        return
    except Exception:
        if dest.exists():
            dest.unlink()
    _export_mp3_ffmpeg(audio, dest, sr)


def describe(settings: VoiceSettings) -> str:
    settings = settings.clamped()
    if settings.formant_ratio < 0.94:
        tone = "偏暖"
    elif settings.formant_ratio > 1.06:
        tone = "偏亮"
    else:
        tone = "接近原声"
    sign = "+" if settings.semitones > 0 else ""
    return (
        f"音色 {settings.formant_ratio:.2f}（{tone}），音高 {sign}{settings.semitones:.0f} 半音，平滑 {settings.smoothness:.0f}\n"
        f"降噪 {settings.denoise:.0f}，模拟噪声 {settings.noise:.0f}，"
        f"抖动 ±{settings.jitter_cents:.0f} 音分 / {settings.jitter_probability:.0f}%\n"
        f"停顿 {settings.pause_count} 次 × {settings.pause_ms} 毫秒，种子 {settings.seed}"
    )


def _make_vowel(sr: int = SR, seconds: float = 2.0, f0: float = 180.0, formant: float = 750.0) -> np.ndarray:
    count = int(sr * seconds)
    excitation = np.zeros(count, dtype=np.float64)
    period = max(1, int(sr / f0))
    excitation[::period] = 1.0
    radius = np.exp(-np.pi * 90 / sr)
    theta = 2 * np.pi * formant / sr
    coef1 = 2 * radius * np.cos(theta)
    coef2 = -(radius ** 2)
    formed = np.zeros(count, dtype=np.float64)
    for index in range(2, count):
        formed[index] = excitation[index] + coef1 * formed[index - 1] + coef2 * formed[index - 2]
    peak = np.max(np.abs(formed)) or 1.0
    return (formed / peak * 0.4).astype(np.float32)


def _formant_peak(mono: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    band = (freqs >= 280) & (freqs <= 1800)
    smooth = np.convolve(spec[band], np.ones(31) / 31, mode="same")
    return float(freqs[band][int(np.argmax(smooth))])


def self_test() -> None:
    tone = 0.2 * np.sin(2 * np.pi * 440 * np.arange(CHUNK) / SR)
    restored = _istft(_stft(tone), CHUNK)
    noise = np.sum((tone - restored) ** 2)
    signal_power = np.sum(tone ** 2)
    snr = 10 * np.log10(signal_power / max(noise, 1e-20))
    if snr < 80:
        raise AssertionError(f"STFT roundtrip SNR too low: {snr:.1f} dB")

    plain = dict(pause_count=0, pause_ms=40, seed=1, denoise=0, noise=0, jitter_cents=0, jitter_probability=0, smoothness=0)
    vowel = _make_vowel()
    stereo = np.stack([vowel, vowel])
    warm = convert_vocals(stereo, SR, VoiceSettings(formant_ratio=0.78, semitones=0, **plain))
    bright = convert_vocals(stereo, SR, VoiceSettings(formant_ratio=1.28, semitones=0, **plain))
    warm_peak = _formant_peak(warm.mean(axis=0), SR)
    bright_peak = _formant_peak(bright.mean(axis=0), SR)
    if not bright_peak > warm_peak * 1.12:
        raise AssertionError(f"timbre slider did not move the formant: warm {warm_peak:.0f} bright {bright_peak:.0f}")

    original_f0 = estimate_f0(vowel, SR)
    pitched = convert_vocals(stereo, SR, VoiceSettings(formant_ratio=1.0, semitones=6, **plain))
    shifted_f0 = estimate_f0(pitched.mean(axis=0), SR)
    if shifted_f0 < original_f0 * 1.25:
        raise AssertionError(f"pitch slider did not raise F0: {original_f0:.1f} -> {shifted_f0:.1f}")

    long = np.tile(stereo, (1, 6))
    none_paused = choose_pauses(long.shape[-1], SR, VoiceSettings(1, 0, 0, 60, 3), long)
    some = choose_pauses(long.shape[-1], SR, VoiceSettings(1, 0, 6, 60, 3), long)
    other = choose_pauses(long.shape[-1], SR, VoiceSettings(1, 0, 6, 60, 9), long)
    if none_paused or len(some) != 6 or some == other:
        raise AssertionError(f"pause placement failed: {len(none_paused)} {some} {other}")
    silenced = apply_pauses(long, SR, some, 60)
    if rms(silenced) >= rms(long):
        raise AssertionError("pauses did not reduce the vocal")

    rng = np.random.default_rng(0)
    hiss = (0.15 * np.sin(2 * np.pi * 220 * np.arange(SR * 2) / SR) + 0.03 * rng.standard_normal(SR * 2)).astype(np.float32)
    cleaned = spectral_denoise(hiss, SR, 80)

    def high_band(samples: np.ndarray) -> float:
        sos = signal.butter(2, 6000 / (SR / 2), btype="highpass", output="sos")
        return rms(signal.sosfiltfilt(sos, samples))

    if high_band(cleaned) > high_band(hiss) * 0.8:
        raise AssertionError("denoise did not lower the high-frequency noise")
    breathy = add_breath(vowel, SR, 80, 2)
    untouched = add_breath(vowel, SR, 0, 2)
    if rms(breathy) <= rms(vowel) * 1.01 or not np.allclose(untouched, vowel):
        raise AssertionError("simulated noise control failed")
    steady = apply_jitter(vowel, SR, 0, 100, 1)
    wobbled = apply_jitter(vowel, SR, 70, 100, 1)
    if not np.allclose(steady, vowel) or np.allclose(wobbled, vowel):
        raise AssertionError("jitter control failed")
    rough = vowel + 0.04 * rng.standard_normal(len(vowel)).astype(np.float32)
    softened = apply_smoothness(vowel, rough, 100)
    if np.corrcoef(softened, vowel)[0, 1] <= np.corrcoef(rough, vowel)[0, 1]:
        raise AssertionError("smoothness did not move the vocal toward the dry signal")
    noise_only = (0.04 * rng.standard_normal(SR)).astype(np.float32)
    protected = convert_vocals(
        np.stack([noise_only, noise_only]),
        SR,
        VoiceSettings(formant_ratio=0.8, semitones=0, **plain),
    )
    noise_corr = float(np.corrcoef(noise_only, protected[0])[0, 1])
    if noise_corr < 0.8:
        raise AssertionError(f"unvoiced noise was resynthesized into buzz, correlation {noise_corr:.2f}")
    print(
        f"self-test ok  snr {snr:.0f} dB  formant {warm_peak:.0f}->{bright_peak:.0f} Hz  "
        f"f0 {original_f0:.0f}->{shifted_f0:.0f} Hz  pauses {len(some)}  noise-corr {noise_corr:.2f}"
    )


if __name__ == "__main__":
    self_test()

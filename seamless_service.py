"""
SeamlessM4T v2 Large FastAPI Service for ASR
Uses facebook/seamless-m4t-v2-large for multilingual speech recognition.

Exposes an OpenAI-compatible transcription endpoint so it can be swapped with
the Whisper vLLM backend without changing the caller.

Run in the 'seamless' conda environment on port 8004:
    conda activate seamless
    pip install fastapi uvicorn python-multipart transformers sentencepiece protobuf soundfile torchaudio
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    uvicorn seamless_service:app --host 0.0.0.0 --port 8004
"""

import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

app = FastAPI(title="SeamlessM4T v2 ASR Service")

# Map common 2-letter codes to SeamlessM4T's ISO 639-3 codes
LANG_MAP = {
    "fa": "pes",   # Persian (Western / Iranian)
    "en": "eng",
    "ar": "arb",
    "tr": "tur",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "ru": "rus",
    "zh": "cmn",
}

MODEL_ID = os.environ.get("SEAMLESS_MODEL", "facebook/seamless-m4t-v2-large")

# Cached model / processor
_model = None
_processor = None
_device = "cpu"
_dtype = None


def get_model():
    """Load the Seamless M4T v2 model and processor on first use."""
    global _model, _processor, _device, _dtype
    if _model is not None:
        return _model, _processor, _device

    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2Model

    if torch.cuda.is_available():
        _device = "cuda"
        _dtype = torch.bfloat16  # Lower memory footprint
    else:
        _device = "cpu"
        _dtype = torch.float32

    print(f"Loading {MODEL_ID} on {_device.upper()} ({_dtype})...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = SeamlessM4Tv2Model.from_pretrained(MODEL_ID, torch_dtype=_dtype).to(_device)
    _model.eval()
    print("Seamless model loaded.")
    return _model, _processor, _device


def read_audio_to_16k_mono(audio_bytes: bytes):
    """Decode arbitrary audio bytes into 16kHz mono float32 numpy array."""
    import numpy as np
    import soundfile as sf

    # Try reading directly with soundfile
    try:
        buf = io.BytesIO(audio_bytes)
        waveform, sr = sf.read(buf, dtype="float32", always_2d=True)
    except Exception:
        # Fall back to ffmpeg conversion
        tmp = Path(tempfile.mkdtemp(prefix="seamless_"))
        try:
            inp = tmp / "input"
            inp.write_bytes(audio_bytes)
            outp = tmp / "out.wav"
            r = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(inp),
                    "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    str(outp),
                ],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise HTTPException(400, f"ffmpeg decode failed: {r.stderr[:300]}")
            waveform, sr = sf.read(str(outp), dtype="float32", always_2d=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # Mono
    if waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1)
    else:
        waveform = waveform[:, 0]

    # Resample to 16k
    if sr != 16000:
        import torch
        import torchaudio
        w = torch.from_numpy(waveform).unsqueeze(0)
        w = torchaudio.functional.resample(w, sr, 16000)
        waveform = w.squeeze(0).numpy()
        sr = 16000

    return waveform, sr


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": _device,
        "model": MODEL_ID,
        "model_loaded": _model is not None,
    }


@app.get("/v1/models")
def list_models():
    """OpenAI-compatible models list (so health checks look identical to vLLM)."""
    return {"data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(MODEL_ID),
    language: str = Form("fa"),
):
    """
    OpenAI-compatible transcription endpoint.
    `language` accepts 2-letter codes (fa, en, ar, ...) or Seamless ISO-639-3 codes directly.
    """
    import torch

    tgt_lang = LANG_MAP.get(language.lower(), language)

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    waveform, sr = read_audio_to_16k_mono(audio_bytes)

    # Guard: Seamless has a context limit; warn for very long inputs
    duration_s = len(waveform) / sr
    if duration_s > 60:
        # Seamless handles up to ~30-60s well; beyond that output quality degrades.
        # The caller (main.py) should send already-chunked audio.
        print(f"Warning: long audio chunk ({duration_s:.1f}s), output may be truncated")

    mdl, proc, dev = get_model()

    inputs = proc(audios=waveform, return_tensors="pt", sampling_rate=16000)
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.inference_mode():
        output_tokens = mdl.generate(
            **inputs,
            tgt_lang=tgt_lang,
            generate_speech=False,
        )

    # SeamlessM4Tv2Model.generate returns a tuple when generate_speech=False;
    # the first element is the text token tensor.
    if isinstance(output_tokens, tuple):
        token_tensor = output_tokens[0]
    else:
        token_tensor = output_tokens

    tokens = token_tensor[0].tolist()
    text = proc.decode(tokens, skip_special_tokens=True)

    return {"text": text.strip()}


@app.on_event("startup")
async def startup():
    """Pre-load the model so first request isn't slow."""
    try:
        get_model()
    except Exception as e:
        print(f"Warning: could not pre-load Seamless model: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

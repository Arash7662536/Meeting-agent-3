"""
Speaker Diarization FastAPI Service
Uses pyannote/speaker-diarization-3.1 to answer "who spoke when?"

Run in the 'annot' conda environment on port 8003:
    conda activate annot
    pip install fastapi uvicorn python-multipart pyannote.audio soundfile
    uvicorn diarization_service:app --host 0.0.0.0 --port 8003
"""

import os
import subprocess
import sys
import shutil
import tempfile
import uuid
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query

app = FastAPI(title="Speaker Diarization Service")

HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HUGGINGFACE_TOKEN_HERE")

# Cache the pipeline so it's loaded only once
_pipeline = None
_device = "cpu"


def get_pipeline():
    """Load and cache the pyannote diarization pipeline."""
    global _pipeline, _device
    if _pipeline is not None:
        return _pipeline

    import torch
    from pyannote.audio import Pipeline

    print("Loading pyannote/speaker-diarization-3.1 model...")
    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )

    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            _pipeline.to(torch.device("cuda"))
            _device = "cuda"
    except RuntimeError:
        print("GPU unavailable, falling back to CPU.")

    print(f"Pipeline loaded on {_device.upper()}")
    return _pipeline


def load_audio(audio_path: Path) -> dict:
    """Load audio as a waveform dict (bypasses torchcodec)."""
    import soundfile as sf
    import torch

    waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T)  # (channels, time)

    if sample_rate != 16000:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return {"waveform": waveform, "sample_rate": sample_rate}


def iter_turns(diarization):
    """Yield (start, end, speaker) from diarization result."""
    if hasattr(diarization, "speaker_diarization"):
        for turn, speaker in diarization.speaker_diarization:
            yield turn.start, turn.end, speaker
    elif hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            yield turn.start, turn.end, speaker
    else:
        raise RuntimeError(f"Unknown diarization output type: {type(diarization)}")


@app.get("/health")
def health():
    return {"status": "ok", "device": _device, "model_loaded": _pipeline is not None}


@app.post("/diarize")
async def diarize(
    file: UploadFile = File(...),
    num_speakers: int = Query(None, description="Exact number of speakers"),
    min_speakers: int = Query(None, description="Minimum number of speakers"),
    max_speakers: int = Query(None, description="Maximum number of speakers"),
):
    """
    Accept an audio file, run speaker diarization, return segments as JSON.
    """
    if HF_TOKEN == "YOUR_HUGGINGFACE_TOKEN_HERE":
        raise HTTPException(500, "HF_TOKEN not configured. Set the HF_TOKEN env variable.")

    job_id = str(uuid.uuid4())[:8]
    work_dir = Path(tempfile.mkdtemp(prefix=f"diarize_{job_id}_"))

    try:
        # Save uploaded file
        suffix = Path(file.filename).suffix.lower() if file.filename else ".wav"
        input_path = work_dir / f"input{suffix}"
        content = await file.read()
        input_path.write_bytes(content)

        # Convert to WAV if needed
        if suffix != ".wav":
            wav_path = work_dir / "audio.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                str(wav_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise HTTPException(500, f"ffmpeg error: {result.stderr[:500]}")
            audio_path = wav_path
        else:
            audio_path = input_path

        # Load audio
        audio_input = load_audio(audio_path)

        # Build kwargs
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers

        # Run diarization
        pipeline = get_pipeline()
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            diarization = pipeline(audio_input, hook=hook, **kwargs)

        # Collect segments
        segments = []
        speaker_times = defaultdict(float)
        for start, end, speaker in iter_turns(diarization):
            segments.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
            })
            speaker_times[speaker] += end - start

        # Sort by start time
        segments.sort(key=lambda s: s["start"])

        duration = audio_input["waveform"].shape[1] / audio_input["sample_rate"]

        return {
            "segments": segments,
            "num_speakers": len(speaker_times),
            "duration": round(duration, 3),
            "speaker_stats": {
                spk: round(dur, 3) for spk, dur in sorted(speaker_times.items())
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Diarization error: {str(e)}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.on_event("startup")
async def startup():
    """Pre-load the pipeline on startup for faster first request."""
    try:
        get_pipeline()
    except Exception as e:
        print(f"Warning: Could not pre-load pipeline: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

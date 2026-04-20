"""
ASR Pipeline - Gradio App
Orchestrates: Voice Extraction -> Speaker Diarization -> Segment Merging -> Whisper Transcription

Usage:
    pip install gradio requests soundfile numpy yt-dlp
    python main.py

Services required (each in its own conda env):
    - extract_voice_service.py  on port 8002 (conda: dm)
    - diarization_service.py    on port 8003 (conda: annot)
    - vLLM Whisper              on port 8001 (conda: vllm)
"""

import io
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import numpy as np
import requests
import soundfile as sf

# ---------------------------------------------------------------------------
# Default service URLs (override via environment variables)
# ---------------------------------------------------------------------------
EXTRACT_SERVICE_URL = os.environ.get("EXTRACT_SERVICE_URL", "http://localhost:8002")
DIARIZE_SERVICE_URL = os.environ.get("DIARIZE_SERVICE_URL", "http://localhost:8003")
WHISPER_URL = os.environ.get("WHISPER_URL", "http://localhost:8001")
SEAMLESS_URL = os.environ.get("SEAMLESS_URL", "http://localhost:8004")

REQUEST_TIMEOUT = 3600  # 1 hour max per service call

# Available ASR backends: (label, service_url_var, model_name)
ASR_BACKENDS = {
    "Whisper (vLLM)": ("whisper", "openai/whisper-large-v3"),
    "SeamlessM4T v2 Large": ("seamless", "facebook/seamless-m4t-v2-large"),
}


# ===========================================================================
# URL validation & download
# ===========================================================================

def is_valid_url(url: str) -> bool:
    """Check if a string is a valid HTTP(S) URL."""
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def is_direct_media_url(url: str) -> tuple[bool, str | None]:
    """
    HEAD-check a URL. Returns (is_media, content_type).
    """
    try:
        resp = requests.head(url, timeout=15, allow_redirects=True)
        ct = resp.headers.get("Content-Type", "").lower()
        media_types = ("audio/", "video/", "application/octet-stream")
        return any(ct.startswith(t) for t in media_types), ct
    except Exception:
        return False, None


def download_direct(url: str, dest_dir: Path) -> Path:
    """Download a direct media URL with requests."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    # Try to get filename from Content-Disposition or URL
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        fname = cd.split("filename=")[-1].strip('" ')
    else:
        fname = Path(urlparse(url).path).name or "download"
    if not Path(fname).suffix:
        fname += ".mp4"

    dest = dest_dir / fname
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def download_with_ytdlp(url: str, dest_dir: Path) -> Path:
    """Download using yt-dlp (handles YouTube, etc.)."""
    import subprocess
    output_template = str(dest_dir / "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestaudio/best",
        "-o", output_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")

    # Find the downloaded file (most recent in dest_dir)
    files = sorted(dest_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError("yt-dlp did not produce any output file")
    return files[0]


def download_from_url(url: str, dest_dir: Path) -> Path:
    """
    Download media from a URL. Tries direct download first, then yt-dlp.
    """
    is_media, ct = is_direct_media_url(url)
    if is_media:
        return download_direct(url, dest_dir)

    # Try yt-dlp for platform URLs (YouTube, etc.)
    try:
        return download_with_ytdlp(url, dest_dir)
    except Exception as e:
        raise RuntimeError(
            f"Could not download from URL.\n"
            f"Direct download content-type: {ct}\n"
            f"yt-dlp error: {e}"
        )


# ===========================================================================
# Service calls
# ===========================================================================

def call_extract_service(file_path: Path, model: str = "htdemucs") -> bytes:
    """Send file to voice extraction service, return vocals WAV bytes."""
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{EXTRACT_SERVICE_URL}/extract",
            files={"file": (file_path.name, f, "application/octet-stream")},
            params={"model": model},
            timeout=REQUEST_TIMEOUT,
        )
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        raise RuntimeError(f"Extract service error ({resp.status_code}): {detail}")
    return resp.content


def call_diarize_service(
    file_path: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """Send audio to diarization service, return segments JSON."""
    params = {}
    if num_speakers:
        params["num_speakers"] = num_speakers
    else:
        if min_speakers:
            params["min_speakers"] = min_speakers
        if max_speakers:
            params["max_speakers"] = max_speakers

    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{DIARIZE_SERVICE_URL}/diarize",
            files={"file": (file_path.name, f, "audio/wav")},
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        raise RuntimeError(f"Diarization service error ({resp.status_code}): {detail}")
    return resp.json()


# ===========================================================================
# Segment merging
# ===========================================================================

def merge_segments(segments: list[dict], max_gap: float = 1.5) -> list[dict]:
    """
    Merge consecutive segments from the same speaker when the gap between
    them is smaller than max_gap seconds.
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segs = sorted(segments, key=lambda s: s["start"])

    merged = [sorted_segs[0].copy()]
    for seg in sorted_segs[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and (seg["start"] - last["end"]) <= max_gap:
            last["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    # Assign chunk IDs: SPEAKER_XX_N
    speaker_counts: dict[str, int] = {}
    for seg in merged:
        spk = seg["speaker"]
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
        seg["chunk_id"] = f"{spk}_{speaker_counts[spk]}"

    return merged


# ===========================================================================
# Audio chunk cutting
# ===========================================================================

def cut_audio_chunk(audio_data: np.ndarray, sr: int, start: float, end: float) -> bytes:
    """Cut a chunk from audio array and return as WAV bytes."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    # Clamp
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)

    chunk = audio_data[start_sample:end_sample]
    if len(chunk) == 0:
        return b""

    buf = io.BytesIO()
    sf.write(buf, chunk, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# ===========================================================================
# Whisper transcription via vLLM
# ===========================================================================

def transcribe_chunk(
    audio_bytes: bytes,
    asr_url: str,
    asr_model_name: str,
    language: str = "fa",
    chunk_id: str = "chunk",
) -> str:
    """Send audio bytes to an ASR backend (Whisper vLLM or Seamless) for transcription."""
    if not audio_bytes:
        return ""

    try:
        resp = requests.post(
            f"{asr_url}/v1/audio/transcriptions",
            files={"file": (f"{chunk_id}.wav", audio_bytes, "audio/wav")},
            data={
                "model": asr_model_name,
                "language": language,
            },
            timeout=300,
        )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("text", "").strip()
        else:
            return f"[Transcription error: {resp.status_code} - {resp.text[:200]}]"
    except requests.exceptions.ConnectionError:
        return "[ASR service unavailable]"
    except Exception as e:
        return f"[Transcription error: {e}]"


# ===========================================================================
# Main pipeline
# ===========================================================================

def check_services(services: dict[str, str]) -> dict[str, bool]:
    """
    Check which services are reachable.
    `services` maps logical names to URLs, e.g. {"extract": "http://...", "whisper": "..."}.
    ASR backends fall back to /v1/models since vLLM doesn't expose /health.
    """
    status = {}
    for name, url in services.items():
        try:
            r = requests.get(f"{url}/health", timeout=5)
            status[name] = r.status_code == 200
        except Exception:
            if name in ("whisper", "seamless"):
                try:
                    r = requests.get(f"{url}/v1/models", timeout=5)
                    status[name] = r.status_code == 200
                except Exception:
                    status[name] = False
            else:
                status[name] = False
    return status


def run_pipeline(
    input_file,
    url_input: str,
    demucs_model: str,
    num_speakers: int,
    min_speakers: int,
    max_speakers: int,
    merge_gap: float,
    asr_choice: str,
    extract_url: str,
    diarize_url: str,
    whisper_url: str,
    seamless_url: str,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Full pipeline: download/upload -> extract vocals -> diarize -> merge -> transcribe -> JSON
    """
    global EXTRACT_SERVICE_URL, DIARIZE_SERVICE_URL, WHISPER_URL, SEAMLESS_URL
    EXTRACT_SERVICE_URL = extract_url.rstrip("/")
    DIARIZE_SERVICE_URL = diarize_url.rstrip("/")
    WHISPER_URL = whisper_url.rstrip("/")
    SEAMLESS_URL = seamless_url.rstrip("/")

    # Resolve ASR backend based on user choice
    if asr_choice not in ASR_BACKENDS:
        return None, None, f"Unknown ASR backend: {asr_choice}"
    asr_key, asr_model_name = ASR_BACKENDS[asr_choice]
    asr_url = WHISPER_URL if asr_key == "whisper" else SEAMLESS_URL

    work_dir = Path(tempfile.mkdtemp(prefix="asr_pipeline_"))
    logs = []

    def log(msg):
        logs.append(msg)
        print(msg)

    try:
        # ----- Step 0: Get input file -----
        progress(0.0, desc="Preparing input...")
        if url_input and url_input.strip():
            url = url_input.strip()
            if not is_valid_url(url):
                return None, None, "Invalid URL. Please provide a valid HTTP/HTTPS link."
            log(f"Downloading from URL: {url}")
            try:
                input_path = download_from_url(url, work_dir)
            except Exception as e:
                return None, None, f"Download failed: {e}"
            log(f"Downloaded: {input_path.name}")
        elif input_file is not None:
            input_path = Path(input_file)
            if not input_path.exists():
                return None, None, "Uploaded file not found."
            log(f"Using uploaded file: {input_path.name}")
        else:
            return None, None, "Please upload a file or provide a URL."

        original_name = input_path.stem

        # ----- Step 1: Check services -----
        progress(0.05, desc="Checking services...")
        services_to_check = {
            "extract": EXTRACT_SERVICE_URL,
            "diarize": DIARIZE_SERVICE_URL,
            asr_key: asr_url,
        }
        svc_status = check_services(services_to_check)
        for name, ok in svc_status.items():
            status_str = "OK" if ok else "UNREACHABLE"
            log(f"  {name} service: {status_str}")
        if not svc_status.get("extract"):
            return None, None, f"Voice extraction service is unreachable at {EXTRACT_SERVICE_URL}"
        if not svc_status.get("diarize"):
            return None, None, f"Diarization service is unreachable at {DIARIZE_SERVICE_URL}"
        log(f"  Using ASR backend: {asr_choice} ({asr_model_name}) at {asr_url}")

        # ----- Step 2: Extract vocals -----
        progress(0.1, desc="Extracting vocals (Demucs)... This may take a while.")
        log("Step 1/4: Extracting vocals with Demucs...")
        t0 = time.time()
        vocals_bytes = call_extract_service(input_path, model=demucs_model)
        log(f"  Vocals extracted in {time.time() - t0:.1f}s ({len(vocals_bytes) / 1024 / 1024:.1f} MB)")

        # Save vocals locally
        vocals_path = work_dir / "vocals.wav"
        vocals_path.write_bytes(vocals_bytes)

        # ----- Step 3: Speaker diarization -----
        progress(0.4, desc="Running speaker diarization...")
        log("Step 2/4: Running speaker diarization...")
        t0 = time.time()

        ns = num_speakers if num_speakers > 0 else None
        mins = min_speakers if min_speakers > 0 else None
        maxs = max_speakers if max_speakers > 0 else None

        diarize_result = call_diarize_service(vocals_path, ns, mins, maxs)
        raw_segments = diarize_result["segments"]
        n_speakers = diarize_result["num_speakers"]
        duration = diarize_result["duration"]
        log(f"  Diarization done in {time.time() - t0:.1f}s")
        log(f"  Found {n_speakers} speakers, {len(raw_segments)} raw segments, duration: {duration:.1f}s")

        # ----- Step 4: Merge segments -----
        progress(0.6, desc="Merging contiguous segments...")
        log("Step 3/4: Merging contiguous segments...")
        merged = merge_segments(raw_segments, max_gap=merge_gap)
        log(f"  {len(raw_segments)} segments merged into {len(merged)} chunks")

        # ----- Step 5: Transcribe each chunk -----
        progress(0.65, desc=f"Transcribing with {asr_choice}...")
        log(f"Step 4/4: Transcribing chunks with {asr_choice}...")

        # Load vocals audio for cutting
        audio_data, sr = sf.read(str(vocals_path), dtype="float32")

        asr_available = svc_status.get(asr_key, False)
        if not asr_available:
            log(f"  WARNING: {asr_choice} service not available. Transcription will be skipped.")

        results = []
        for i, seg in enumerate(merged):
            progress(0.65 + 0.3 * (i / max(len(merged), 1)), desc=f"Transcribing chunk {i+1}/{len(merged)}...")

            chunk_bytes = cut_audio_chunk(audio_data, sr, seg["start"], seg["end"])

            if asr_available and chunk_bytes:
                text = transcribe_chunk(chunk_bytes, asr_url, asr_model_name, language="fa", chunk_id=seg["chunk_id"])
            else:
                text = f"[{asr_choice} unavailable]" if not asr_available else ""

            results.append({
                "chunk_id": seg["chunk_id"],
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "duration": round(seg["end"] - seg["start"], 3),
                "text": text,
            })
            log(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['chunk_id']}: {text[:80]}...")

        # ----- Build output JSON -----
        progress(0.95, desc="Building output...")

        # Build full transcript with speaker labels
        full_transcript_lines = []
        for r in results:
            full_transcript_lines.append(f"[{r['speaker']}] {r['text']}")

        output = {
            "file": original_name,
            "duration": duration,
            "num_speakers": n_speakers,
            "speaker_stats": diarize_result.get("speaker_stats", {}),
            "merge_gap_threshold": merge_gap,
            "asr_backend": asr_choice,
            "asr_model": asr_model_name,
            "total_chunks": len(results),
            "segments": results,
            "full_transcript": "\n".join(full_transcript_lines),
        }

        # Save JSON
        json_path = work_dir / f"{original_name}_transcript.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        progress(1.0, desc="Done!")
        log(f"\nPipeline complete! {len(results)} transcribed segments.")

        formatted_json = json.dumps(output, ensure_ascii=False, indent=2)
        log_text = "\n".join(logs)

        return str(json_path), formatted_json, log_text

    except Exception as e:
        log_text = "\n".join(logs)
        return None, None, f"{log_text}\n\nERROR: {e}"


# ===========================================================================
# Gradio UI
# ===========================================================================

def build_ui():
    with gr.Blocks(
        title="ASR Pipeline - Voice Extraction, Diarization & Transcription",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # ASR Pipeline
            **Voice Extraction** (Demucs) -> **Speaker Diarization** (pyannote) -> **Transcription** (Whisper vLLM)

            Upload an audio/video file or provide a URL. The pipeline will extract clean vocals,
            identify who spoke when, merge contiguous segments, and transcribe each chunk to Persian.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Upload Audio/Video File",
                    file_types=[".mp4", ".mkv", ".avi", ".mov", ".webm", ".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                )
                url_input = gr.Textbox(
                    label="Or Enter URL (YouTube, direct link, etc.)",
                    placeholder="https://www.youtube.com/watch?v=... or https://example.com/audio.mp3",
                )

            with gr.Column(scale=1):
                asr_choice = gr.Radio(
                    choices=list(ASR_BACKENDS.keys()),
                    value="Whisper (vLLM)",
                    label="ASR Model",
                    info="Whisper is fast via vLLM; Seamless M4T v2 is multilingual and runs with transformers.",
                )

                with gr.Accordion("Settings", open=False):
                    demucs_model = gr.Dropdown(
                        choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
                        value="htdemucs",
                        label="Demucs Model",
                    )
                    num_speakers = gr.Number(value=0, label="Exact Number of Speakers (0 = auto)", precision=0)
                    min_speakers = gr.Number(value=0, label="Min Speakers (0 = auto)", precision=0)
                    max_speakers = gr.Number(value=0, label="Max Speakers (0 = auto)", precision=0)
                    merge_gap = gr.Slider(
                        minimum=0.5, maximum=5.0, value=1.5, step=0.1,
                        label="Merge Gap Threshold (seconds)",
                    )

                with gr.Accordion("Service URLs", open=False):
                    extract_url = gr.Textbox(value=EXTRACT_SERVICE_URL, label="Extract Service URL")
                    diarize_url = gr.Textbox(value=DIARIZE_SERVICE_URL, label="Diarization Service URL")
                    whisper_url = gr.Textbox(value=WHISPER_URL, label="Whisper vLLM URL")
                    seamless_url = gr.Textbox(value=SEAMLESS_URL, label="Seamless M4T v2 URL")

        run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                output_json = gr.Code(label="Output JSON", language="json", lines=25)
            with gr.Column():
                log_output = gr.Textbox(label="Pipeline Log", lines=25, interactive=False)

        output_file = gr.File(label="Download JSON Result")

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                file_input, url_input,
                demucs_model, num_speakers, min_speakers, max_speakers,
                merge_gap,
                asr_choice,
                extract_url, diarize_url, whisper_url, seamless_url,
            ],
            outputs=[output_file, output_json, log_output],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)

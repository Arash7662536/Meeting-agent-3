# ASR Pipeline - Voice Extraction, Diarization & Transcription

A modular ASR pipeline that extracts clean vocals from audio/video, identifies speakers, and transcribes speech to Persian using Whisper.

## Architecture

```
                         +-------------------+
  User (Gradio UI) ----->|   main.py :7860   |
                         +--------+----------+
                                  |
             +--------------------+--------------------+-----------------+
             |                    |                    |                 |
             v                    v                    v                 v
  +--------------------+  +-------------------+  +-----------------+  +-------------------+
  | extract_voice_svc  |  | diarization_svc   |  | Whisper vLLM    |  | Seamless M4T v2   |
  | (dm env) :8002     |  | (annot env) :8003 |  | (vllm env):8001 |  | (seamless) :8004  |
  +--------------------+  +-------------------+  +-----------------+  +-------------------+
                                                         \_________ASR (user choice)_______/
```

Each service runs in its own conda environment. The Gradio app orchestrates the pipeline.
Whisper and Seamless M4T v2 are **interchangeable ASR backends** — the user picks one in the UI.

## Pipeline Flow

1. **Input** - Upload audio/video file or provide a URL (YouTube, direct link)
2. **Vocal Extraction** - Demucs separates vocals from background music/noise
3. **Speaker Diarization** - pyannote identifies who spoke when
4. **Segment Merging** - Contiguous segments from the same speaker are merged
5. **Transcription** - Each merged chunk is transcribed to Persian via the selected ASR backend (Whisper or Seamless M4T v2)
6. **Output** - Structured JSON with per-speaker segments and full transcript

## Prerequisites

- **Python 3.10+**
- **ffmpeg** installed and in PATH
- **4 conda environments**: `dm`, `annot`, `vllm`, `seamless` (Seamless is optional — only needed if you want to use it as an ASR backend)
- **CUDA GPU** (recommended for all services)
- **Hugging Face token** with access to pyannote models

### ffmpeg Installation

```bash
# Ubuntu
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows - download from https://ffmpeg.org/download.html
```

## Setup

### 1. Voice Extraction Service (conda: dm)

```bash
conda activate dm
pip install fastapi uvicorn python-multipart demucs soundfile
pip uninstall torchaudio torchcodec -y
pip install torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Diarization Service (conda: annot)

```bash
conda activate annot
pip install fastapi uvicorn python-multipart pyannote.audio soundfile
```

You need a Hugging Face token with access to:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

Set it via environment variable (recommended):
```bash
export HF_TOKEN="hf_your_token_here"
```

### 3. Whisper vLLM (conda: vllm)

```bash
conda activate vllm
# Install vLLM per https://docs.vllm.ai/en/latest/getting_started/installation.html
```

### 4. SeamlessM4T v2 Service (conda: seamless) — optional ASR backend

Create a dedicated environment and install dependencies:

```bash
conda create -n seamless python=3.10 -y
conda activate seamless

# PyTorch with CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Transformers + friends
pip install transformers sentencepiece protobuf soundfile
pip install fastapi uvicorn python-multipart
```

Notes:
- The model `facebook/seamless-m4t-v2-large` is ~9 GB and downloads from Hugging Face on first run
- Needs roughly ~20 GB VRAM at bfloat16 (loaded automatically if CUDA is available)
- Supports Persian via the ISO-639-3 code `pes` (mapped automatically from `fa`)
- Uses the exact same OpenAI-compatible endpoint shape as Whisper (`/v1/audio/transcriptions`), so swapping is seamless

### 5. Gradio App (any Python 3.10+ environment)

```bash
pip install gradio requests soundfile numpy yt-dlp
```

## Running

You need **4-5 terminals** (or use tmux/screen). Start services in order. You only need to launch the ASR backend(s) you plan to use — Whisper, Seamless, or both.

### Terminal 1 - Whisper vLLM

```bash
conda activate vllm
vllm serve openai/whisper-large-v3 \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype bfloat16 \
    --max-model-len 448 \
    --gpu-memory-utilization 0.40
```

### Terminal 2 - Voice Extraction Service

```bash
conda activate dm
cd /path/to/Agent_3
uvicorn extract_voice_service:app --host 0.0.0.0 --port 8002
```

### Terminal 3 - Diarization Service

```bash
conda activate annot
cd /path/to/Agent_3
export HF_TOKEN="hf_your_token_here"
uvicorn diarization_service:app --host 0.0.0.0 --port 8003
```

### Terminal 4 - SeamlessM4T v2 Service (optional)

```bash
conda activate seamless
cd /path/to/Agent_3
uvicorn seamless_service:app --host 0.0.0.0 --port 8004
```

The service pre-loads the model on startup, so the first boot takes a minute.

### Terminal 5 - Gradio App

```bash
cd /path/to/Agent_3
python main.py
```

The Gradio app launches at `http://localhost:7860` with a **public shareable link** printed in the terminal.

## Running on Remote Server

If your services run on a remote machine (e.g., Ubuntu GPU server), set the URLs when launching the Gradio app:

```bash
export EXTRACT_SERVICE_URL="http://your-server:8002"
export DIARIZE_SERVICE_URL="http://your-server:8003"
export WHISPER_URL="http://your-server:8001"
export SEAMLESS_URL="http://your-server:8004"
python main.py
```

Or configure them in the Gradio UI under "Service URLs".

## Output Format

The pipeline produces a JSON file:

```json
{
  "file": "my_meeting",
  "duration": 120.5,
  "num_speakers": 2,
  "speaker_stats": {
    "SPEAKER_00": 65.3,
    "SPEAKER_01": 55.2
  },
  "merge_gap_threshold": 1.5,
  "asr_backend": "Whisper (vLLM)",
  "asr_model": "openai/whisper-large-v3",
  "total_chunks": 12,
  "segments": [
    {
      "chunk_id": "SPEAKER_00_1",
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 10.5,
      "duration": 10.5,
      "text": "متن فارسی ..."
    }
  ],
  "full_transcript": "[SPEAKER_00] متن فارسی ...\n[SPEAKER_01] ..."
}
```

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| ASR Model | Whisper (vLLM) | Choose `Whisper (vLLM)` or `SeamlessM4T v2 Large` |
| Demucs Model | `htdemucs` | `htdemucs`, `htdemucs_ft`, or `mdx_extra` |
| Num Speakers | 0 (auto) | Set exact number if known |
| Min/Max Speakers | 0 (auto) | Constrain speaker detection range |
| Merge Gap | 1.5s | Max gap (seconds) to merge same-speaker segments |

## Troubleshooting

- **"Service unreachable"**: Make sure the FastAPI services are running and the URLs are correct
- **Demucs slow**: First run downloads the model (~1GB). Subsequent runs are faster. Use GPU for speed.
- **Diarization fails**: Check your HF_TOKEN has accepted the pyannote model agreements
- **Whisper errors**: Verify vLLM is running and the model is loaded (`curl http://server:8001/v1/models`)
- **Seamless OOM**: SeamlessM4T v2 Large needs ~20 GB VRAM. Share the GPU with other services by lowering `--gpu-memory-utilization` for vLLM, or run Seamless on a separate GPU
- **Seamless slow first request**: The model pre-loads on startup, but first download is ~9 GB
- **URL download fails**: Install `yt-dlp` (`pip install yt-dlp`) for YouTube/platform URLs

## Choosing Between Whisper and Seamless

| Aspect | Whisper (vLLM) | SeamlessM4T v2 Large |
|--------|----------------|----------------------|
| Speed | Fast (vLLM batching) | Slower (HuggingFace generate) |
| VRAM | ~3 GB at bf16 | ~20 GB at bf16 |
| Persian quality | Very good | Very good, multilingual-trained |
| Best for | Quick transcription, throughput | Cross-lingual consistency, translation-ready |

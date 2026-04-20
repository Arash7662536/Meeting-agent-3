# Agent_3 - ASR Pipeline

## Project Overview

A microservices-based ASR (Automatic Speech Recognition) pipeline for Persian audio/video transcription with speaker diarization. The system separates vocals, identifies speakers, and transcribes speech using three independent services orchestrated by a Gradio frontend.

## Architecture

- **extract_voice_service.py** (port 8002, conda `dm`) - FastAPI service wrapping Meta's Demucs for vocal separation from audio/video
- **diarization_service.py** (port 8003, conda `annot`) - FastAPI service wrapping pyannote/speaker-diarization-3.1 for speaker identification
- **main.py** (port 7860) - Gradio app that orchestrates the full pipeline: input handling -> vocal extraction -> diarization -> segment merging -> ASR transcription -> JSON output
- **Whisper vLLM** (port 8001, conda `vllm`) - External vLLM server serving openai/whisper-large-v3 for Persian transcription (ASR backend option 1)
- **seamless_service.py** (port 8004, conda `seamless`) - FastAPI service wrapping facebook/seamless-m4t-v2-large via HuggingFace transformers (ASR backend option 2)

## Key Design Decisions

- Services are separate because each requires a different conda environment with conflicting dependencies
- Communication between services is HTTP/REST (file uploads via multipart form data)
- The diarization pipeline caches the pyannote model in memory after first load
- The Seamless pipeline pre-loads the model on startup so first request isn't slow
- Segment merging uses a configurable gap threshold (default 1.5s) to combine consecutive same-speaker segments
- Audio chunks are cut from the extracted vocals (not original audio) for cleaner transcription
- The Gradio app generates a public share link for remote access
- Whisper and Seamless M4T v2 are **interchangeable** ASR backends - both expose an OpenAI-compatible `/v1/audio/transcriptions` endpoint so `main.py` routes to either based on a Gradio radio button
- `main.py` defines `ASR_BACKENDS` dict mapping UI labels -> (service_key, model_name); adding a new backend only requires a new entry + service URL

## Conda Environments

- `dm` - Demucs, torchaudio 2.4.0 (CUDA 12.1), soundfile, fastapi
- `annot` - pyannote.audio, soundfile, fastapi
- `vllm` - vLLM with whisper-large-v3 model
- `seamless` - torch (CUDA 12.1), transformers, sentencepiece, protobuf, torchaudio, soundfile, fastapi. Loads `facebook/seamless-m4t-v2-large` in bfloat16 on GPU (~20 GB VRAM)

## Environment Variables

- `HF_TOKEN` - Hugging Face token for pyannote models (used by diarization_service.py)
- `EXTRACT_SERVICE_URL` - Override extract service URL (default: http://localhost:8002)
- `DIARIZE_SERVICE_URL` - Override diarization service URL (default: http://localhost:8003)
- `WHISPER_URL` - Override Whisper vLLM URL (default: http://localhost:8001)
- `SEAMLESS_URL` - Override Seamless M4T v2 service URL (default: http://localhost:8004)
- `SEAMLESS_MODEL` - Override Seamless model ID (default: facebook/seamless-m4t-v2-large)

## Original Standalone Scripts

- `extract_voice_mp4.py` - Original CLI script for voice extraction (kept for reference)
- `speaker_diarization.py` - Original CLI script for diarization (kept for reference)

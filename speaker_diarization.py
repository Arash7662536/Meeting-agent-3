"""
Speaker Diarization using pyannote/speaker-diarization-3.1
Answers: "Who spoke when?" in your audio/video file.

Requirements:
    pip install pyannote.audio soundfile
    

You also need a FREE Hugging Face token:
    1. Go to https://huggingface.co/settings/tokens
    2. Create a token (read access is enough)
    3. Accept model conditions at:
       - https://huggingface.co/pyannote/speaker-diarization-3.1
       - https://huggingface.co/pyannote/segmentation-3.0

    

"""

import argparse
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"


def install_dependencies():
    try:
        import pyannote.audio
    except ImportError:
        print("Installing pyannote.audio...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyannote.audio"])
        print("Done.\n")


def extract_audio_if_video(input_path: Path, temp_dir: Path) -> Path:
    """If input is a video file, extract audio using ffmpeg."""
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    if input_path.suffix.lower() not in video_extensions:
        return input_path

    print("Video detected. Extracting audio track...")
    audio_path = temp_dir / (input_path.stem + "_audio.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:", result.stderr)
        sys.exit(1)
    print(f"Audio extracted -> {audio_path}\n")
    return audio_path


def load_audio(audio_path: Path):
    """
    Preload audio as a waveform dict to bypass torchcodec entirely.
    Uses soundfile + torch.
    """
    try:
        import soundfile as sf
        import torch

        print(f"Loading audio from: {audio_path}")
        waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform.T)  # (channels, time)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        print(f"Audio loaded: {waveform.shape[1] / sample_rate:.1f}s @ {sample_rate}Hz\n")
        return {"waveform": waveform, "sample_rate": sample_rate}

    except ImportError:
        print("soundfile not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
        return load_audio(audio_path)


def iter_turns(diarization):
    """
    Unified iterator over diarization results.
    Handles both:
      - Old pyannote: Annotation object with .itertracks()
      - New pyannote: DiarizeOutput object with .speaker_diarization
    Yields (start, end, speaker) tuples.
    """
    if hasattr(diarization, "speaker_diarization"):
        # New pyannote >= 3.3: DiarizeOutput
        for turn, speaker in diarization.speaker_diarization:
            yield turn.start, turn.end, speaker
    elif hasattr(diarization, "itertracks"):
        # Old pyannote: Annotation
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            yield turn.start, turn.end, speaker
    else:
        raise RuntimeError(f"Unknown diarization output type: {type(diarization)}")


def run_diarization(audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    print("Loading pyannote/speaker-diarization-3.1 model...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )

    # Safe GPU check — avoids crash on driver/CUDA version mismatch
    device = "cpu"
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            pipeline.to(torch.device("cuda"))
            device = "cuda"
    except RuntimeError:
        print("GPU unavailable, falling back to CPU.")
    print(f"Running on: {device.upper()}\n")

    # Preload audio to bypass torchcodec
    audio_input = load_audio(audio_path)

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    print("Running diarization (this may take a while for long files)...")
    with ProgressHook() as hook:
        diarization = pipeline(audio_input, hook=hook, **kwargs)

    return diarization


def print_summary(diarization):
    speaker_times = defaultdict(float)
    for start, end, speaker in iter_turns(diarization):
        speaker_times[speaker] += end - start

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total speakers detected: {len(speaker_times)}\n")
    for speaker, duration in sorted(speaker_times.items()):
        minutes, seconds = divmod(duration, 60)
        print(f"  {speaker}: {int(minutes)}m {seconds:.1f}s total speaking time")
    print()


def save_results(diarization, audio_path: Path, output_dir: Path):
    stem = audio_path.stem

    # Plain text timeline
    txt_path = output_dir / f"{stem}_diarization.txt"
    with open(txt_path, "w") as f:
        f.write("Speaker Diarization Results\n")
        f.write(f"File: {audio_path.name}\n")
        f.write("=" * 50 + "\n\n")
        for start, end, speaker in iter_turns(diarization):
            f.write(f"[{start:07.3f}s - {end:07.3f}s]  {speaker}\n")

    # RTTM format
    rttm_path = output_dir / f"{stem}_diarization.rttm"
    try:
        annotation = diarization if hasattr(diarization, "write_rttm") else diarization.speaker_diarization
        with open(rttm_path, "w") as f:
            annotation.write_rttm(f)
    except Exception as e:
        print(f"  (RTTM export skipped: {e})")
        rttm_path = None

    return txt_path, rttm_path


def main():
    parser = argparse.ArgumentParser(description="Speaker diarization using pyannote 3.1")
    parser.add_argument("input", help="Path to audio or video file")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--token", help="Hugging Face token (overrides HF_TOKEN in script)")
    parser.add_argument("--num-speakers", type=int, help="Exact number of speakers (if known)")
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers")
    args = parser.parse_args()

    global HF_TOKEN
    if args.token:
        HF_TOKEN = args.token

    if HF_TOKEN == "YOUR_HUGGINGFACE_TOKEN_HERE":
        print("ERROR: Please set your Hugging Face token.")
        print("  Edit HF_TOKEN in the script, or pass --token YOUR_TOKEN")
        sys.exit(1)

    install_dependencies()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: File not found - {input_path}")
        sys.exit(1)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    print(f"\nInput  : {input_path}")
    print(f"Output : {output_dir}\n")

    audio_path = extract_audio_if_video(input_path, temp_dir)

    diarization = run_diarization(
        audio_path,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    print("\n--- DIARIZATION TIMELINE ---")
    for start, end, speaker in iter_turns(diarization):
        print(f"[{start:07.3f}s -> {end:07.3f}s]  {speaker}")

    print_summary(diarization)

    txt_path, rttm_path = save_results(diarization, audio_path, output_dir)
    print("Results saved:")
    print(f"  Text : {txt_path}")
    if rttm_path:
        print(f"  RTTM : {rttm_path}")

    if audio_path != input_path:
        audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

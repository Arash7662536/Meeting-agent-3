"""
Voice Extractor from MP4
Extracts audio from a video file, then separates vocals using Meta's Demucs.

Install dependencies first:
    pip install demucs
    # Also requires ffmpeg installed on your system:
    # macOS:   brew install ffmpeg
    # Ubuntu:  sudo apt update && sudo apt install ffmpeg
    # Windows: https://ffmpeg.org/download.html

    pip uninstall torchaudio torchcodec -y
    pip install torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install soundfile

"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def check_ffmpeg():
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH.")
        print("  macOS  : brew install ffmpeg")
        print("  Ubuntu : sudo apt install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
        sys.exit(1)


def install_demucs():
    """Install demucs if not already installed."""
    try:
        import demucs
    except ImportError:
        print("Installing demucs...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "demucs"])
        print("Demucs installed successfully.\n")


def extract_audio_from_video(video_path: Path, temp_dir: Path) -> Path:
    """
    Use ffmpeg to extract audio track from MP4 as a WAV file.

    Args:
        video_path: Path to the input .mp4 file.
        temp_dir:   Directory to write the temporary WAV file.

    Returns:
        Path to the extracted WAV file.
    """
    audio_path = temp_dir / (video_path.stem + "_audio.wav")

    print(f"Step 1/2: Extracting audio from video...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                   # No video
        "-acodec", "pcm_s16le",  # Standard WAV format
        "-ar", "44100",          # 44.1 kHz sample rate
        "-ac", "2",              # Stereo
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ffmpeg error:")
        print(result.stderr)
        sys.exit(1)

    print(f"   Audio extracted → {audio_path}\n")
    return audio_path


def extract_voice(input_path: str, output_dir: str = "output", model: str = "htdemucs"):
    """
    Extract vocals from an MP4 video file.

    Args:
        input_path: Path to the input MP4 file.
        output_dir: Directory where results will be saved.
        model:      Demucs model to use.
                    - 'htdemucs'    : Best quality, default
                    - 'htdemucs_ft' : Fine-tuned, slightly better on some tracks
                    - 'mdx_extra'   : Alternative high-quality model
    """
    input_path = Path(input_path).resolve()

    if not input_path.exists():
        print(f"Error: File not found — {input_path}")
        sys.exit(1)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input file  : {input_path}")
    print(f"Output dir  : {output_dir}")
    print(f"Model       : {model}\n")

    # Step 1: Extract audio from MP4
    audio_path = extract_audio_from_video(input_path, temp_dir)

    # Step 2: Run Demucs for vocal separation
    print("Step 2/2: Separating vocals from music...")
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",   # Only separate vocals vs. accompaniment
        "-n", model,
        "-o", str(output_dir),
        str(audio_path),
    ]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nDemucs encountered an error. Try a different model with -m flag.")
        sys.exit(result.returncode)

    # Locate output files
    stem_name      = audio_path.stem
    vocals_path    = output_dir / model / stem_name / "vocals.wav"
    no_vocals_path = output_dir / model / stem_name / "no_vocals.wav"

    print("\n✅ Done! Output files:")
    if vocals_path.exists():
        print(f"   🎤 Vocals only : {vocals_path}")
    if no_vocals_path.exists():
        print(f"   🎵 Music only  : {no_vocals_path}")

    # Clean up temp audio file
    audio_path.unlink(missing_ok=True)

    return str(vocals_path), str(no_vocals_path)


def main():
    parser = argparse.ArgumentParser(description="Extract voice from an MP4 video file.")
    parser.add_argument("input",  help="Path to the input MP4 file")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: ./output)")
    parser.add_argument(
        "-m", "--model",
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
        help="Demucs model to use (default: htdemucs)"
    )
    args = parser.parse_args()

    check_ffmpeg()
    install_demucs()
    extract_voice(args.input, args.output, args.model)


if __name__ == "__main__":
    main()

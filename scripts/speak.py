#!/usr/bin/env -S uv run

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import sounddevice as sd

from app.models.kokoro_tts import KokoroTTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Speak text using Kokoro-ONNX TTS")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("--voice", default="af_sarah", help="Voice to use")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed of speech")
    parser.add_argument("--device", default="cpu", help="Device to run model on")
    args = parser.parse_args()

    print(f"Loading model on {args.device}...")
    tts = KokoroTTS("kokoro-82m-onnx", device=args.device)

    print(f"Generating speech for: '{args.text}'")
    samples, sample_rate = tts.generate_speech(
        args.text, voice=args.voice, speed=args.speed
    )

    print("Playing audio...")
    try:
        sd.play(samples, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")
        print("Make sure you have PortAudio installed")
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()

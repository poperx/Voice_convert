# Voice Convert

Voice Convert is a desktop tool for changing the character of vocals in audio and video files. It separates vocals from the backing track, applies adjustable acoustic effects, then mixes the processed vocals back into the original audio.

Use it to experiment with vocal timbre and pitch, smoothness, noise, jitter, and randomly placed short pauses—all through a graphical interface.

## Features

- Separate vocals from the backing track, with the separated audio cached for later previews and exports.
- Adjust vocal timbre (warmer or brighter) and pitch (up to ±12 semitones).
- Control smoothness and denoise strength, or add simulated breath-like noise.
- Add randomized pitch jitter and short pauses. A random seed lets you reproduce a pause pattern.
- Preview a 20-second section and export the processed mix as an MP3.
- Process common audio and video inputs: MP4, MKV, MOV, WAV, MP3, FLAC, and M4A.

> The output is a mix of the processed vocals and the original backing track; it is not a vocals-only export. Export is currently MP3 only.

## Requirements

- Windows is recommended. The graphical interface is built with Tkinter, and in-app audio preview is currently supported on Windows.
- Python 3.12 is recommended.
- FFmpeg available on your `PATH` for media decoding/encoding fallback. The app can also use its bundled PyAV support for many media operations.
- Internet access on first use to download the vocal-separation model (approximately 64 MB), unless the model is already present locally.

## Run from source

1. Install Python 3.12 and FFmpeg. Make sure `ffmpeg` is available from a terminal.
2. Clone or download this repository, then open a terminal in the project directory.
3. (Optional but recommended) Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Launch the app:

   ```powershell
   python voice_gui.py
   ```

The app downloads the separation model the first time it is needed and stores it under `models/`. If `curl` is not available, install it or place the model at `models/UVR-MDX-NET-Voc_FT.onnx` yourself.

## Use the app

1. Select an audio or video file.
2. Choose **Separate Vocals** to extract and cache the vocal and instrumental stems. Separation is only needed again if the source file changes.
3. Adjust the sliders for timbre, pitch, smoothness, denoising, simulated noise, jitter, and pauses.
4. Choose **Preview 20 seconds** to hear a sample, or **Export MP3** to render the full mix.

The interface currently displays Chinese labels. The controls correspond to the features listed above.

## Build a Windows executable

The repository includes `build_exe.bat` for packaging with PyInstaller. The script expects the bundled Python runtime, dependency wheels, and model resources configured by the project’s build environment. Run it from a prepared project checkout; it produces the executable in `dist/`.

## Dependencies

Python package versions are listed in [`requirements.txt`](requirements.txt). The project uses NumPy, SciPy, ONNX Runtime, Praat-Parselmouth, and PyAV.

## License

This project is distributed under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

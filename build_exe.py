import subprocess
import sys


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--console",
            "--onedir",
            "--name",
            "VoiceToText",
            "--collect-all",
            "PIL",
            "--collect-all",
            "transformers",
            "--collect-all",
            "torch",
            "--exclude-module",
            "matplotlib",
            "--exclude-module",
            "scipy",
            "--exclude-module",
            "pandas",
            "--exclude-module",
            "pytest",
            "--exclude-module",
            "yt_dlp",
            "--exclude-module",
            "numba",
            "--exclude-module",
            "IPython",
            "--exclude-module",
            "notebook",
            "--exclude-module",
            "jupyter",
            "--noconfirm",
            "voicetotext/__main__.py",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

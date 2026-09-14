"""Launcher kept for backwards compatibility: ``python main.py`` opens the GUI,
``python main.py video.mp4 [options]`` runs the command line interface."""

from auto_video_editor.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

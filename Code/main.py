import os
import sys
import platform
import tkinter as tk
from matplotlib import font_manager
from src.gui import LoadingScreen, start_main_gui
from src.utils import get_data_path


def set_gdal_proj_paths():
    """
    Ensures GDAL and PROJ data directories are correctly set for Windows, macOS, and PyInstaller packages.
    - PyInstaller: Extracts paths from bundled environment
    """

    if getattr(sys, 'frozen', False):  # Running as a PyInstaller-packed application
        base_path = sys._MEIPASS
        os.environ["GDAL_DATA"] = os.path.join(base_path, "gdal")
        os.environ["PROJ_LIB"] = os.path.join(base_path, "pyproj/proj_dir")
    else:
        if platform.system() == "Darwin":  # macOS
            os.environ["GDAL_DATA"] = "/opt/homebrew/share/gdal"
            os.environ["PROJ_LIB"] = "/opt/homebrew/share/proj"
        elif platform.system() == "Windows":  # Windows
            os.environ["GDAL_DATA"] = r"C:\Users\leo\AppData\Local\Programs\Python\Python311\Lib\site-packages\osgeo\data\gdal"
            os.environ["PROJ_LIB"] = r"C:\Users\leo\AppData\Local\Programs\Python\Python311\Lib\site-packages\pyproj\proj_dir\share\proj"

    print(f"Using GDAL_DATA: {os.environ['GDAL_DATA']}")
    print(f"Using PROJ_LIB: {os.environ['PROJ_LIB']}")

def ensure_matplotlib_font_cache():
    """
    Trigger font discovery and cache build in the simplest, safest way.
    """
    try:
        # This call alone triggers font loading and cache building if needed
        font_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        _ = font_manager.get_font(font_paths[0]) if font_paths else None

        print("Matplotlib font cache triggered.")
    except Exception as e:
        print(f"Warning: Could not ensure font cache: {e}")

def main():
    """
    Entry point for launching the Data Cleaning and Fusion Tool.
    - Sets the `DATA_PATH` environment variable to ensure proper access to `data/`.
    - Ensures Matplotlib font cache is built before launching GUI.
    - Initializes the Tkinter application with a loading screen.
    """

    # Ensure Matplotlib font cache is ready
    ensure_matplotlib_font_cache()

    # Set PROJ data path
    set_gdal_proj_paths()

    # Set environment variable for `data/` directory
    os.environ["DATA_PATH"] = get_data_path()

    # Initialize Tkinter root window
    root = tk.Tk()

    # Launch the loading screen, then transition to the main GUI
    LoadingScreen(root, lambda: start_main_gui(root))

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()
import os
import sys


def get_data_path():
    """
    Ensures `data/` is always inside the `Software/` project directory, NOT inside `dist/`.
    """
    if getattr(sys, 'frozen', False):  # Running in a packaged app
        base_path = os.path.dirname(os.path.abspath(sys.executable))  # Default: dist/[AppName].app/Contents/MacOS/

        if sys.platform == "darwin":  # macOS (.app bundle)
            software_root = os.path.abspath(os.path.join(base_path, "../../../../"))  # Move up to Software/
        else:  # Windows/Linux
            software_root = os.path.abspath(os.path.join(base_path, ".."))  # Move up to Software/
    else:
        # Running from source, assume script is inside `Software/src/`
        software_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Move up from `src/` to `Software/`

    # Return the `data/` directory inside `Software/`
    data_path = os.path.join(software_root, "data")
    return data_path

def get_asset_path(filename):
    """
    Get the absolute path of an asset file (e.g., images) inside the `assets/` directory.
    - If running from source, looks in `Software/src/assets/`.
    - If running from a PyInstaller package, looks inside `sys._MEIPASS/assets/`.
    """
    if getattr(sys, 'frozen', False):  # Running in a packaged app
        base_path = os.path.join(sys._MEIPASS, "assets")  # Locate assets inside PyInstaller package
    else:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

    asset_path = os.path.join(base_path, filename)
    return asset_path

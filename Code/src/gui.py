import os
import sys
import time
import tkinter as tk
import subprocess
import threading
import platform
from tkinter import Canvas, Entry, Button, Label, Text, filedialog
from PIL import Image, ImageTk
from src.utils import get_data_path, get_asset_path
from src.basic_data_cleaning.unified_database import CSVToSQLiteProcessor
from src.basic_data_cleaning.time_standardization import TimeStandardizationProcessor
from src.basic_data_cleaning.basic_data_cleaning import BasicDataCleaner
from src.map_matching.mapmatching import MapMatchingProcessor, MapMatchingAnalyzer
from src.trend_analysis_cbi.trend_analysis_cbi import TrendAnalyzer, CBIAnalyzer
from src.trend_analysis_cbi.od_analysis_wp import ODMatrixAnalyzerWP, ODTravelTimeAnalyzerWP
from src.trend_analysis_cbi.od_analysis_tp import ODMatrixAnalyzerTP, ODTravelTimeAnalyzerTP
from src.basic_data_fusion.speed_fusion_wp import SpeedFusionAggregatorWP
from src.basic_data_fusion.speed_fusion_tp import SpeedFusionAggregatorTP
from src.advanced_data_fusion.penetration_rate_wp import PenetrationRateAnalyzerWP
from src.advanced_data_fusion.penetration_rate_tp import PenetrationRateAnalyzerTP
from src.advanced_data_fusion.od_fusion_wp import ODFusionAggregatorWP
from src.advanced_data_fusion.od_fusion_tp import ODFusionAggregatorTP
from src.odme_simulation.time_dependent_od_wp import TimeDependentODAggregatorWP
from src.odme_simulation.time_dependent_od_tp import TimeDependentODAggregatorTP
from src.odme_simulation.fd_wp import FDCalibratorWP
from src.odme_simulation.fd_tp import FDCalibratorTP


def get_base_path():
    """
    Returns the correct base directory path:
    - If running in source mode, use `os.path.dirname(__file__)`.
    - If running from a PyInstaller `.app` or `.exe`, use `sys._MEIPASS`.
    """
    if getattr(sys, 'frozen', False):  # Running from a packaged app
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))  # Running from source


def get_font(size, weight="normal"):
    """
    Dynamically adjusts font size based on the operating system.

    - macOS: Uses `Helvetica` and increases the size slightly for better readability.
    - Windows: Uses `Segoe UI` with the given size.
    - Other systems: Defaults to `Arial`.
    """
    if platform.system() == "Darwin":  # macOS
        return ("Helvetica", size + 2, weight)  # Slightly larger for macOS
    elif platform.system() == "Windows":
        return ("Segoe UI", size - 2, weight)  # Standard size for Windows
    else:
        return ("Arial", size, weight)  # Default font for Linux and others


class LoadingScreen:
    def __init__(self, master, main_app_callback):
        """Displays a loading screen with an image, then launches the main GUI."""
        self.master = master
        self.main_app_callback = main_app_callback  # Function to start the main GUI

        self.master.title("Loading...")
        self.master.geometry("960x540")
        self.master.configure(bg="black")

        self.image_path = get_asset_path("loading_image.png")

        # Load and display the image
        try:
            image = Image.open(self.image_path)
            image = image.resize((960, 540), Image.LANCZOS)
            self.img = ImageTk.PhotoImage(image)

            self.label = Label(self.master, image=self.img)
            self.label.image = self.img
            self.label.pack(expand=True, fill="both")
        except Exception as e:
            print(f"Error loading image: {e}")
            self.label = Label(self.master, text="Loading...", fg="white", bg="black", font=("Arial", 24))
            self.label.pack(expand=True)

        # Automatically transition to the main GUI after 5 seconds
        self.master.after(5000, self.close_and_start_main_gui)

    def close_and_start_main_gui(self):
        """Closes the loading screen and starts the main GUI."""
        start_main_gui(self.master)  # Use the same Tk window


class TextRedirector:
    """ Redirects print output to the GUI and automatically logs it to a file. """

    def __init__(self, widget, output_folder_callback):
        self.widget = widget
        self.output_folder_callback = output_folder_callback

    def get_log_file_path(self):
        output_folder = self.output_folder_callback()
        log_filename = f"log_{time.strftime('%Y%m%d')}.txt"
        return os.path.join(output_folder, log_filename)

    def write(self, string):
        """ Inserts text into the GUI and writes to the log file. """
        self.widget.insert(tk.END, string)  # Show in GUI log
        self.widget.see(tk.END)
        self.widget.update_idletasks()

        # Write to the log file
        log_file_path = self.get_log_file_path()
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(string)

    def flush(self):
        pass  # Required for stdout compatibility


class StdoutRedirector:
    """ Context manager to redirect stdout to both the GUI and a log file. """

    def __init__(self, gui_widget, output_folder_callback, enable=True):
        self.gui_widget = gui_widget
        self.output_folder_callback = output_folder_callback
        self.enable = enable
        self.original_stdout = None

    def __enter__(self):
        """ Start redirection if enabled. """
        if self.enable and not isinstance(sys.stdout, TextRedirector):
            self.original_stdout = sys.stdout
            sys.stdout = TextRedirector(self.gui_widget, self.output_folder_callback)

    def __exit__(self, exc_type, exc_value, traceback):
        """ Restore original stdout. """
        if self.original_stdout:
            sys.stdout = self.original_stdout


class TqdmRedirector:
    """ Redirect tqdm progress bar to a Tkinter Text widget, ensuring only one line updates dynamically. """
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.progress_line_index = None

    def write(self, message):
        if not message:
            return

        if "\r" in message:
            message = message.split("\r")[-1].rstrip("\n")

            if self.progress_line_index is None:
                self.progress_line_index = self.text_widget.index("end-1l")
            else:
                self.text_widget.delete(self.progress_line_index, self.progress_line_index + " lineend")
                self.text_widget.insert(self.progress_line_index, message)
        else:
            self.text_widget.insert(tk.END, message)

        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass  # Required for tqdm

# class TqdmRedirector:
#     """ Redirect tqdm progress bar to a Tkinter Text widget, ensuring only one line updates dynamically. """
#     def __init__(self, text_widget):
#         self.text_widget = text_widget
#         self.progress_line_index = None
#
#     def write(self, message):
#         if not message or not self.text_widget:
#             return
#         try:
#             if "\r" in message:
#                 message = message.split("\r")[-1].rstrip("\n")
#
#                 if self.progress_line_index is None:
#                     self.progress_line_index = self.text_widget.index("end-1l")
#                 else:
#                     self.text_widget.delete(self.progress_line_index, self.progress_line_index + " lineend")
#                     self.text_widget.insert(self.progress_line_index, message)
#             else:
#                 self.text_widget.insert(tk.END, message)
#
#             self.text_widget.see(tk.END)
#             self.text_widget.update_idletasks()
#
#         except Exception as e:
#             # fallback to stdout on GUI failure
#             print(message)
#
#     def flush(self):
#         pass  # Required for tqdm

class DataProcessingGUI:
    def __init__(self, window):
        """ Initializes the GUI components """
        self.window = window
        self.window.title("Data Cleaning and Fusion Tool")
        self.window.geometry("960x540")
        self.window.minsize(960, 540)
        self.window.configure(bg="#1D3B6B")

        # Define project paths
        # Retrieve the `data/` directory path
        data_dir = get_data_path()
        self.input_folder_default = os.path.join(data_dir, "input")
        self.output_folder_default = os.path.join(data_dir, "output")
        # self.input_folder_default = os.path.dirname("/data/input/")
        # self.output_folder_default = os.path.dirname("/data/output/")

        self.create_widgets()

        self.output_folder_callback = lambda: self.output_entry.get()


        self.is_running = {  # Track whether each process is running
            "Initial_Data_Quality_Check_and_Formatting": False,
            "Remove_Spatial_Outliers_Using_Map_Matching": False,
            "Trend_Analysis_and_CBI": False,
            "Basic_Data_Fusion": False,
            "Advanced_Data_Fusion": False,
            "ODME_and_Calibrate_Initial_Simulation_Parameters": False
        }
        self.stop_events = {name: threading.Event() for name in self.is_running}  # Create stop event for each process
        self.threads = {}  # Store running threads

        # Define log file path inside `data/` output folder
        log_filename = f"log_{time.strftime('%Y%m%d')}.txt"
        self.log_file_path = os.path.join(self.output_folder_callback(), log_filename)

    def create_widgets(self):
        """Creates the widgets for the GUI."""
        # Create a canvas for background elements
        self.canvas = Canvas(self.window, bg="#1D3B6B", bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Store references to elements for dynamic resizing
        self.rect_id = self.canvas.create_rectangle(480, 0, 960, 720, fill="#F5F5F5", outline="")
        self.divider1_id = self.canvas.create_line(480, 340, 960, 340, fill="black", width=2.5)
        # self.divider2_id = self.canvas.create_line(480, 520, 960, 520, fill="black", width=2.5)

        # Title
        Label(self.window, text="Data Cleaning and Fusion Tool", fg="white", bg="#1D3B6B",
              font=get_font(22, "bold")).place(relx=0.05, rely=0.05, relwidth=0.4, relheight=0.07)

        # Input Folder
        Label(self.window, text="Data Input Folder", fg="white", bg="#1D3B6B",
              font=get_font(16), anchor="w").place(relx=0.05, rely=0.15, relwidth=0.25, relheight=0.05)

        self.input_entry = Entry(self.window, bd=1, font=get_font(12), bg="#F5F5F5", fg="black")
        self.input_entry.place(relx=0.05, rely=0.2, relwidth=0.4, relheight=0.06)
        self.input_entry.insert(0, os.path.abspath(self.input_folder_default))

        Button(self.window, text="Browse", font=get_font(12), bg="white", relief="ridge",
               command=lambda: self.select_folder(self.input_entry)).place(relx=0.38, rely=0.2, relwidth=0.07,
                                                                           relheight=0.06)

        # Output Folder
        Label(self.window, text="Data Output Folder", fg="white", bg="#1D3B6B",
              font=get_font(16), anchor="w").place(relx=0.05, rely=0.3, relwidth=0.25, relheight=0.05)

        self.output_entry = Entry(self.window, bd=1, font=get_font(12), bg="#F5F5F5", fg="black")
        self.output_entry.place(relx=0.05, rely=0.35, relwidth=0.4, relheight=0.06)
        self.output_entry.insert(0, os.path.abspath(self.output_folder_default))

        Button(self.window, text="Browse", font=get_font(12), bg="white", relief="ridge",
               command=lambda: self.select_folder(self.output_entry)).place(relx=0.38, rely=0.35, relwidth=0.07,
                                                                            relheight=0.06)

        # Packages Selection
        self.selected_packages = tk.StringVar(value="1")

        frame_1 = tk.Frame(self.window, bg="#F5F5F5", relief="ridge", borderwidth=3)
        frame_1.place(relx=0.05, rely=0.45, relwidth=0.18, relheight=0.06)

        self.package_1 = tk.Radiobutton(self.window, text="Package 1",
                                        variable=self.selected_packages, value="1",
                                        font=get_font(14), bg="#F5F5F5", fg="black")
        self.package_1.place(relx=0.06, rely=0.46, relwidth=0.16, relheight=0.04)

        frame_2 = tk.Frame(self.window, bg="#F5F5F5", relief="ridge", borderwidth=3)
        frame_2.place(relx=0.27, rely=0.45, relwidth=0.18, relheight=0.06)

        self.package_2 = tk.Radiobutton(self.window, text="Package 2",
                                        variable=self.selected_packages, value="2",
                                        font=get_font(14), bg="#F5F5F5", fg="black")
        self.package_2.place(relx=0.28, rely=0.46, relwidth=0.16, relheight=0.04)

        # Results Log
        Label(self.window, text="Results", fg="white", bg="#1D3B6B",
              font=get_font(16), anchor="w").place(relx=0.05, rely=0.53, relwidth=0.25, relheight=0.05)

        self.extra_entry = Text(self.window, bd=1, font=get_font(12), bg="#F5F5F5", fg="black")
        self.extra_entry.place(relx=0.05, rely=0.58, relwidth=0.4, relheight=0.25)

        Button(self.window, text="DB Browser", font=get_font(14),
               bg="white", relief="ridge", command=self.open_database).place(relx=0.05, rely=0.85, relwidth=0.15, relheight=0.08)

        Button(self.window, text="QGIS", font=get_font(14),
               bg="white", relief="ridge", command=self.open_qgis).place(relx=0.30, rely=0.85, relwidth=0.15, relheight=0.08)

        # Data Cleaning Section
        Label(self.window, text="Data Cleaning", fg="black", bg="#F5F5F5",
              font=get_font(16, "bold"), anchor="w").place(relx=0.52, rely=0.02, relwidth=0.4, relheight=0.07)

        self.basic_cleaning_button = Button(self.window, text="1. Initial Data Quality Check and Formatting",
                                            font=get_font(14), bg="white", relief="ridge",
                                            command=lambda: self.toggle_task("Initial_Data_Quality_Check_and_Formatting",
                                                                             self.basic_cleaning_button,
                                                                             self.run_basic_data_cleaning_task))
        self.basic_cleaning_button.original_text = "1. Initial Data Quality Check and Formatting"
        self.basic_cleaning_button.place(relx=0.55, rely=0.1, relwidth=0.4, relheight=0.08)

        self.map_matching_button = Button(self.window, text="2. Remove Spatial Outliers Using Map Matching",
                                          font=get_font(14), bg="white", relief="ridge", state="disabled",
                                          command=lambda: self.toggle_task("Remove_Spatial_Outliers_Using_Map_Matching",
                                                                           self.map_matching_button,
                                                                           self.run_mapmatching_task))
        self.map_matching_button.original_text = "2. Remove Spatial Outliers Using Map Matching"
        self.map_matching_button.place(relx=0.55, rely=0.22, relwidth=0.4, relheight=0.08)

        self.trend_analysis_button = Button(self.window, text="3. Trend Analysis and CBI",
                                            font=get_font(14), bg="white", relief="ridge", state="disabled",
                                            command=lambda: self.toggle_task("Trend_Analysis_and_CBI",
                                                                             self.trend_analysis_button,
                                                                             self.run_trend_analysis_cbi_task))
        self.trend_analysis_button.original_text = "3. Trend Analysis and CBI"
        self.trend_analysis_button.place(relx=0.55, rely=0.34, relwidth=0.4, relheight=0.08)

        # Data Fusion Section
        Label(self.window, text="Data Fusion", fg="black", bg="#F5F5F5",
              font=get_font(16, "bold"), anchor="w").place(relx=0.52, rely=0.52, relwidth=0.4, relheight=0.07)

        self.basic_fusion_button = Button(self.window, text="4. Basic Data Fusion",
                                          font=get_font(14), bg="white", relief="ridge", state="disabled",
                                          command=lambda: self.toggle_task("Basic_Data_Fusion",
                                                                           self.basic_fusion_button,
                                                                           self.run_basic_data_fusion_task))
        self.basic_fusion_button.original_text = "4. Basic Data Fusion"
        self.basic_fusion_button.place(relx=0.55, rely=0.60, relwidth=0.4, relheight=0.08)

        self.advanced_fusion_button = Button(self.window, text="5. Advanced Data Fusion", state="disabled",
                                             font=get_font(14), bg="white", relief="ridge",
                                             command=lambda: self.toggle_task("Advanced_Data_Fusion",
                                                                              self.advanced_fusion_button,
                                                                              self.run_advanced_data_fusion_task))
        self.advanced_fusion_button.original_text = "5. Advanced Data Fusion"
        self.advanced_fusion_button.place(relx=0.55, rely=0.72, relwidth=0.4, relheight=0.08)

        self.odme_simulation_button = Button(self.window, text="6. ODME and Calibrate Initial Simulation Parameters",
                                             font=get_font(14), bg="white", relief="ridge", state="disabled",
                                             command=lambda: self.toggle_task("ODME_and_Calibrate_Initial_Simulation_Parameters",
                                                                              self.odme_simulation_button,
                                                                              self.run_odme_simulation_task))
        self.odme_simulation_button.original_text = "6. ODME and Calibrate Initial Simulation Parameters"
        self.odme_simulation_button.place(relx=0.55, rely=0.84, relwidth=0.4, relheight=0.08)

        self.next_task_mapping = {
            "Initial_Data_Quality_Check_and_Formatting": self.map_matching_button,
            "Remove_Spatial_Outliers_Using_Map_Matching": self.trend_analysis_button,
            "Trend_Analysis_and_CBI": self.basic_fusion_button,
            "Basic_Data_Fusion": self.advanced_fusion_button,
            "Advanced_Data_Fusion": self.odme_simulation_button,
            "ODME_and_Calibrate_Initial_Simulation_Parameters": None
        }

        # Bind window resize event
        self.window.bind("<Configure>", self.resize_elements)

    def resize_elements(self, event):
        """Dynamically resizes elements when the window is resized."""
        if self.canvas.winfo_exists():  # Ensure canvas exists before updating
            new_width, new_height = self.window.winfo_width(), self.window.winfo_height()

            # Resize the right panel rectangle
            self.canvas.coords(self.rect_id, new_width / 2, 0, new_width, new_height)

            # Adjust the black divider lines
            mid_x = new_width / 2
            y1 = new_height * 0.50
            # y2 = new_height * 0.81
            self.canvas.coords(self.divider1_id, mid_x, y1, new_width, y1)
            # self.canvas.coords(self.divider2_id, mid_x, y2, new_width, y2)

    def select_folder(self, entry_field):
        """ Opens a folder selection dialog and updates the entry field """
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            entry_field.delete(0, tk.END)
            entry_field.insert(0, folder_selected)

    def update_status(self, message, duration=None):
        """
        Updates the results log with a timestamped message.
        If a duration (seconds) is provided, it formats it as minutes and seconds.
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # Format duration if provided
        if duration is not None:
            minutes, seconds = divmod(duration, 60)  # Convert seconds to minutes + seconds
            if minutes >= 1:
                duration_str = f" in {int(minutes)} min {seconds:.2f} sec"
            else:
                duration_str = f" in {seconds:.2f} sec"
            message += duration_str  # Append duration to the message

        log_message = f"{timestamp} - {message}\n"

        # Insert the formatted message into the results log
        self.extra_entry.insert(tk.END, log_message)
        self.extra_entry.see(tk.END)
        self.window.update_idletasks()

        log_file_path = os.path.join(self.output_folder_callback(), f"log_{time.strftime('%Y%m%d')}.txt")

        # Ensure log directory exists before writing
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except Exception as e:
                print(f"Error creating log directory: {e}")
                return

        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(log_message)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def open_database(self):
        output_folder = self.output_entry.get()
        database_path = os.path.join(output_folder, "database", "unified_database.db")

        if os.path.exists(database_path):
            if os.name == "nt":  # Windows
                os.startfile(database_path)
            elif os.uname().sysname == "Darwin":  # macOS
                subprocess.Popen(["open", database_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", database_path])
        else:
            print("Database file not found:", database_path) # Linux

    def open_qgis(self):
        input_folder = self.input_entry.get()
        qgis_path = os.path.join(input_folder, "network", "qgis.qgz")

        if os.path.exists(qgis_path):
            if os.name == "nt":  # Windows
                os.startfile(qgis_path)
            elif os.uname().sysname == "Darwin":  # macOS
                subprocess.Popen(["open", qgis_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", qgis_path])
        else:
            print("QGIS file not found:", qgis_path)  # Linux

    def toggle_task(self, task_name, button, task_function):
        """ Starts or stops a data processing task based on its current state. """
        if self.is_running[task_name]:
            # If running, stop it
            self.stop_events[task_name].set()
            self.update_status(f"Stopping {task_name.replace('_', ' ')}...")
            self.is_running[task_name] = False
            self.window.after(100, lambda: button.config(text=button.original_text))
        else:
            # If not running, start it
            self.stop_events[task_name].clear()  # Reset stop event

            input_folder = self.input_entry.get()
            output_folder = self.output_entry.get()
            database_path = os.path.join(output_folder, "database", "unified_database.db")

            if not input_folder or not output_folder:
                self.update_status("ERROR: Please select input and output folders")
                return

            # Define a wrapper function to run the task and enable the next button when done
            def task_wrapper():
                try:
                    task_function(input_folder, output_folder, database_path, self.stop_events[task_name])
                except Exception as e:
                    self.update_status(f"ERROR: {str(e)}")
                finally:
                    self.is_running[task_name] = False

                    # If the task was not stopped manually, enable the next button
                    if not self.stop_events[task_name].is_set():
                        next_button = self.next_task_mapping.get(task_name)
                        if next_button:
                            self.window.after(100, lambda: next_button.config(state="normal"))

                    # Restore button state
                    self.is_running[task_name] = False
                    self.window.after(100, lambda: button.config(text=button.original_text))

            # Ensure only one task runs at a time
            for key in self.is_running:
                if self.is_running[key]:
                    self.update_status("ERROR: Another task is already running!")
                    return

            # Start the task in a separate thread
            self.threads[task_name] = threading.Thread(target=task_wrapper)
            self.threads[task_name].start()

            # Update button text and state
            self.is_running[task_name] = True
            button.config(text="Stop")

    def enable_next_task(self, task_name):
        """Enable the next task button based on the task sequence"""
        next_button = self.next_task_mapping.get(task_name)
        if next_button:
            self.window.after(100, lambda: next_button.config(state="normal"))

    def restore_button_state(self, task_name):
        """Restore the button text and update running state"""
        self.is_running[task_name] = False
        task_button = self.get_task_button_by_name(task_name)
        if task_button:
            self.window.after(100, lambda: task_button.config(text=task_button.original_text))

    def get_task_button_by_name(self, task_name):
        """Returns the button widget based on task name"""
        task_buttons = {
            "Initial_Data_Quality_Check_and_Formatting": self.basic_cleaning_button,
            "Remove_Spatial_Outliers_Using_Map_Matching": self.map_matching_button,
            "Trend_Analysis_and_CBI": self.trend_analysis_button,
            "Basic_Data_Fusion": self.basic_fusion_button,
            "Advanced_Data_Fusion": self.advanced_fusion_button,
            "ODME_and_Calibrate_Initial_Simulation_Parameters": self.odme_simulation_button
        }
        return task_buttons.get(task_name)

    def run_basic_data_cleaning_task(self, input_folder, output_folder, database_path, stop_event):
        """ Runs the entire data cleaning pipeline """
        self.extra_entry.delete("1.0", tk.END)

        try:
            start_time = time.time()
            with ((StdoutRedirector(self.extra_entry, self.output_folder_callback, enable=True))):  # Redirect only inside this block
                self.update_status("Starting Initial Data Quality Check and Formatting...")

                if stop_event.is_set(): return
                self.update_status("Running Unified Database Processing...")
                CSVToSQLiteProcessor(input_folder, output_folder, stop_event).run()

                if stop_event.is_set(): return
                self.update_status("Running Time Standardization...")
                TimeStandardizationProcessor(database_path, output_folder, stop_event).run()

                if stop_event.is_set(): return
                self.update_status("Running Basic Data Cleaning...")
                BasicDataCleaner(database_path, stop_event).run()

            total_time = time.time() - start_time
            self.update_status("Initial Data Quality Check and Formatting Completed", duration=total_time)

            # If task was completed normally, enable the next button
            if not stop_event.is_set():
                self.window.after(100, lambda: self.enable_next_task("Initial_Data_Quality_Check_and_Formatting"))

        except Exception as e:
            self.update_status(f"ERROR: {e}")

        # Restore button state (ensuring it's on the main thread)
        self.window.after(100, lambda: self.restore_button_state("Initial_Data_Quality_Check_and_Formatting"))

    def run_mapmatching_task(self, input_folder, output_folder, database_path, stop_event):
        """ Runs the map matching process """
        try:
            selected_task = self.selected_packages.get()
            start_time = time.time()

            tqdm_output = TqdmRedirector(self.extra_entry)
            from tqdm import tqdm
            tqdm._instances.clear()

            if selected_task == "1":

                self.update_status("Starting Remove Spatial Outliers Using Map Matching...")

                if stop_event.is_set(): return

                MapMatchingProcessor(input_folder, output_folder, stop_event=stop_event,
                                     progress_callback=self.update_status, progress_output=tqdm_output).run()

                # with StdoutRedirector(self.extra_entry, self.output_folder_callback, enable=True):
                #     if stop_event.is_set(): return
                #     MapMatchingAnalyzer(database_path).run()

                total_time = time.time() - start_time
                self.update_status("Remove Spatial Outliers Using Map Matching Completed", duration=total_time)

            elif selected_task == "2":

                self.update_status("Remove Spatial Outliers Using Map Matching is not needed for Package 2...")

                total_time = time.time() - start_time
                self.update_status("Remove Spatial Outliers Using Map Matching Skipped", duration=total_time)

            # If task was completed normally, enable the next button
            if not stop_event.is_set():
                self.window.after(100, lambda: self.enable_next_task("Remove_Spatial_Outliers_Using_Map_Matching"))

        except Exception as e:
            self.update_status(f"ERROR: {e}")

        # Restore button state (ensuring it's on the main thread)
        self.window.after(100, lambda: self.restore_button_state("Remove_Spatial_Outliers_Using_Map_Matching"))

    def run_trend_analysis_cbi_task(self, input_folder, output_folder, database_path, stop_event):
        """ Runs the map matching process """
        try:
            selected_task = self.selected_packages.get()
            start_time = time.time()
            with StdoutRedirector(self.extra_entry, self.output_folder_callback, enable=True):  # Redirect only inside this block
                self.update_status("Starting Trend Analysis and CBI...")

                if stop_event.is_set(): return
                self.update_status("Running Trend Analysis...")
                TrendAnalyzer(database_path, output_folder, stop_event=stop_event).run()

                if stop_event.is_set(): return
                self.update_status("Running CBI...")
                CBIAnalyzer(database_path, output_folder, stop_event=stop_event).run()

                if selected_task == "1":

                    if stop_event.is_set(): return
                    self.update_status("Running OD Matrix Analysis...")
                    ODMatrixAnalyzerWP(database_path, output_folder, stop_event=stop_event).run()

                    if stop_event.is_set(): return
                    self.update_status("Running OD Travel Time Analysis...")
                    ODTravelTimeAnalyzerWP(database_path, output_folder, stop_event=stop_event).run()

                elif selected_task == "2":

                    if stop_event.is_set(): return
                    self.update_status("Running OD Matrix Analysis...")
                    ODMatrixAnalyzerTP(database_path, output_folder, stop_event=stop_event).run()

                    if stop_event.is_set(): return
                    self.update_status("Running OD Travel Time Analysis...")
                    ODTravelTimeAnalyzerTP(database_path, output_folder, stop_event=stop_event).run()

            total_time = time.time() - start_time
            self.update_status("Trend Analysis and CBI Completed", duration=total_time)

            # If task was completed normally, enable the next button
            if not stop_event.is_set():
                self.window.after(100, lambda: self.enable_next_task("Trend_Analysis_and_CBI"))

        except Exception as e:
            self.update_status(f"ERROR: {e}")

        # Restore button state (ensuring it's on the main thread)
        self.window.after(100, lambda: self.restore_button_state("Trend_Analysis_and_CBI"))

    def run_basic_data_fusion_task(self, input_folder, output_folder, database_path, stop_event):
        """ Runs the map matching process """
        if not input_folder or not output_folder:
            self.update_status("ERROR: Please select input and output folders")
            return

        try:
            selected_task = self.selected_packages.get()
            start_time = time.time()
            with StdoutRedirector(self.extra_entry, self.output_folder_callback, enable=True):  # Redirect only inside this block
                self.update_status("Starting Basic Data Fusion...")

                if selected_task == "1":

                    if stop_event.is_set(): return
                    self.update_status("Running Speed Fusion...")
                    SpeedFusionAggregatorWP(database_path, output_folder, stop_event=stop_event).run()

                elif selected_task == "2":

                    if stop_event.is_set(): return
                    self.update_status("Running Speed Fusion...")
                    SpeedFusionAggregatorTP(database_path, output_folder, stop_event=stop_event).run()

            total_time = time.time() - start_time
            self.update_status("Basic Data Fusion Completed", duration=total_time)

            # If task was completed normally, enable the next button
            if not stop_event.is_set():
                self.window.after(100, lambda: self.enable_next_task("Basic_Data_Fusion"))

        except Exception as e:
            self.update_status(f"ERROR: {e}")

        # Restore button state (ensuring it's on the main thread)
        self.window.after(100, lambda: self.restore_button_state("Basic_Data_Fusion"))

    def run_advanced_data_fusion_task(self, input_folder, output_folder, database_path, stop_event):
        """ Runs the map matching process """
        try:
            selected_task = self.selected_packages.get()
            start_time = time.time()
            with StdoutRedirector(self.extra_entry, self.output_folder_callback, enable=True):  # Redirect only inside this block
                self.update_status("Starting Advanced Data Fusion...")

                if selected_task == "1":

                    if stop_event.is_set(): return
                    self.update_status("Running Penetration Rate Analysis...")
                    PenetrationRateAnalyzerWP(database_path, output_folder, stop_event=stop_event).run()

                    if stop_event.is_set(): return
                    self.update_status("Running OD Matrix Fusion...")
                    ODFusionAggregatorWP(database_path, output_folder, stop_event=stop_event).run()

                elif selected_task == "2":

                    if stop_event.is_set(): return
                    self.update_status("Running Penetration Rate Analysis...")
                    PenetrationRateAnalyzerTP(database_path, output_folder, stop_event=stop_event).run()

                    if stop_event.is_set(): return
                    self.update_status("Running OD Matrix Fusion...")
                    ODFusionAggregatorTP(database_path, output_folder, stop_event=stop_event).run()

            total_time = time.time() - start_time
            self.update_status("Advanced Data Fusion Completed", duration=total_time)

            # If task was completed normally, enable the next button
            if not stop_event.is_set():
                self.window.after(100, lambda: self.enable_next_task("Advanced_Data_Fusion"))

        except Exception as e:
            self.update_status(f"ERROR: {e}")

        # Restore button state (ensuring it's on the main thread)
        self.window.after(100, lambda: self.restore_button_state("Advanced_Data_Fusion"))

    def run_odme_simulation_task(self, input_folder, output_folder, database_path, stop_event):
        """ Runs the map matching process """
        try:
            selected_task = self.selected_packages.get()
            start_time = time.time()
            with StdoutRedirector(self.extra_entry, self.output_folder_callback, enable=True):  # Redirect only inside this block
                self.update_status("Starting ODME and Calibrate Initial Simulation Parameters...")

                if selected_task == "1":

                    if stop_event.is_set(): return
                    self.update_status("Running Time Dependent ODME...")
                    TimeDependentODAggregatorWP(database_path, output_folder, stop_event=stop_event).run()

                    if stop_event.is_set(): return
                    self.update_status("Running FD Calibration...")
                    FDCalibratorWP(database_path, output_folder, stop_event=stop_event).run()

                elif selected_task == "2":

                    if stop_event.is_set(): return
                    self.update_status("Running Time Dependent ODME...")
                    TimeDependentODAggregatorTP(database_path, output_folder, stop_event=stop_event).run()

                    if stop_event.is_set(): return
                    self.update_status("Running FD Calibration...")
                    FDCalibratorTP(database_path, output_folder, stop_event=stop_event).run()

            total_time = time.time() - start_time
            self.update_status("ODME and Calibrate Initial Simulation Parameters Completed", duration=total_time)

            # If task was completed normally, enable the next button
            if not stop_event.is_set():
                self.window.after(100, lambda: self.enable_next_task("ODME_and_Calibrate_Initial_Simulation_Parameters"))

        except Exception as e:
            self.update_status(f"ERROR: {e}")

        # Restore button state (ensuring it's on the main thread)
        self.window.after(100, lambda: self.restore_button_state("ODME_and_Calibrate_Initial_Simulation_Parameters"))


def start_main_gui(root):
    """Destroys the loading screen and launches the main GUI in the same Tk instance."""
    for widget in root.winfo_children():
        widget.destroy()  # Remove all existing widgets from the root window

    DataProcessingGUI(root)  # Initialize main GUI on the same root
    root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()  # Use a single Tk instance
    LoadingScreen(root, lambda: start_main_gui(root))  # Pass root to main GUI
    root.mainloop()
import os
import DTALite as dta

class DTALiteRunner:
    def __init__(self, input_folder, output_folder, stop_event=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.stop_event = stop_event

        if self.stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")

        os.makedirs(self.output_folder, exist_ok=True)

    def run_assignment(self):
        if self.stop_event and self.stop_event.is_set():
            print("Stopping ODME")
            return

        original_path = os.getcwd()

        try:
            os.chdir(self.input_folder)
            dta.assignment()
        finally:
            os.chdir(original_path)

    def move_output_files(self):
        if self.stop_event and self.stop_event.is_set():
            return

        output_files = [
            "link_performance.csv",
            "route_assignment.csv",
            "summary_log_file.txt",
            "sample_settings.csv",
            "sample_mode_type.csv",
            "ODME_log.txt",
            "od_performance.csv",
            "TAP_log.csv"
        ]

        for file in output_files:
            src = os.path.join(self.input_folder, file)
            dst = os.path.join(self.output_folder, file)

            if os.path.exists(src):
                os.rename(src, dst)
            else:
                print(f"Warning: {src} not found. Skipping.")

    def run(self):
        self.run_assignment()
        self.move_output_files()
        print("ODME completed.")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "input", "odme")
    OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output", "odme_simulation", "odme")

    odem_runner = DTALiteRunner(INPUT_FOLDER, OUTPUT_FOLDER)
    odem_runner.run()
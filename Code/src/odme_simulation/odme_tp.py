import os
import shutil
import pandas as pd
import DTALite as dta


class DTALiteRunnerTP:
    def __init__(self, input_folder, output_folder, stop_event=None):
        """
        Initialize ODME runner using top-level input and output folders.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.odme_folder = os.path.join(output_folder, "odme_simulation", "odme")
        os.makedirs(self.odme_folder, exist_ok=True)

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

        self.prepare_input_files()

    def generate_settings_csv(self, output_path):
        """
        Generate settings.csv matching the single-row structure you provided.
        """
        print("Generating settings.csv...")

        data = {
            "metric_system": [1],
            "number_of_iterations": [10],
            "number_of_processors": [8],
            "demand_period_starting_hours": [8],
            "demand_period_ending_hours": [9],
            "base_demand_mode": [0],
            "route_output": [1],
            "vehicle_output": [0],
            "log_file": [0],
            "odme_mode": [1],
            "odme_vmt": [0]
        }

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    def prepare_input_files(self):
        """
        Generate settings.csv, copy network files, copy OD demand, and merge link volume.
        """

        # Generate settings.csv
        settings_dst = os.path.join(self.odme_folder, "settings.csv")
        self.generate_settings_csv(settings_dst)

        # Copy network files
        for file in ["link.csv", "node.csv"]:
            src = os.path.join(self.input_folder, "network", file)
            dst = os.path.join(self.odme_folder, file)
            if os.path.exists(src):
                shutil.copy(src, dst)
                # print(f"Copied {file} to output folder.")
            else:
                print(f"Warning: {src} not found. Skipping.")

        # Copy demand file
        probe_od_src = os.path.join(self.output_folder, "odme_simulation", "probe_od_volume", "probe_od_volume_8_9am.csv")
        demand_dst = os.path.join(self.odme_folder, "demand.csv")
        if os.path.exists(probe_od_src):
            demand_df = pd.read_csv(probe_od_src)
            if "volume" in demand_df.columns:
                demand_df["volume"] = demand_df["volume"].replace(0, -1)
                demand_df.to_csv(demand_dst, index=False)
                # print(f"Copied probe OD file to {demand_dst}")
        else:
            print(f"Warning: {probe_od_src} not found. Skipping.")

        # Copy demand_target file
        populated_od_src = os.path.join(self.output_folder, "odme_simulation", "od_matrix_wp", "fused_od_volume_08_Weekday_tp.csv")
        demand_target_dst = os.path.join(self.odme_folder, "demand_target.csv")
        if os.path.exists(populated_od_src):
            demand_target_df = pd.read_csv(populated_od_src)
            if "adjusted_volume" in demand_target_df.columns:
                demand_target_df["adjusted_volume"] = demand_target_df["adjusted_volume"].replace(0, -1)
                demand_target_df = demand_target_df.rename(columns={"adjusted_volume": "volume"})
                demand_target_df.to_csv(demand_target_dst, index=False)
                # print(f"Copied populated OD file to {demand_target_dst}")
        else:
            print(f"Warning: {populated_od_src} not found. Skipping.")

        # Merge probe_link_volume into link.csv
        link_csv_path = os.path.join(self.odme_folder, "link.csv")
        probe_link_volume_path = os.path.join(self.output_folder, "odme_simulation", "probe_link_volume", "probe_link_volume_8_9am.csv")

        if os.path.exists(link_csv_path) and os.path.exists(probe_link_volume_path):
            link_df = pd.read_csv(link_csv_path)
            volume_df = pd.read_csv(probe_link_volume_path)
            volume_df = volume_df.rename(columns={"volume": "obs_volume"})

            # Merge on link_id
            volume_df = volume_df[["link_id", "obs_volume"]]
            merged_df = link_df.merge(volume_df, on="link_id", how="left")

            # Fill NaNs with 0 for unmatched links
            merged_df["obs_volume"] = merged_df["obs_volume"].fillna(-1).astype(int)

            # Save back to link.csv
            merged_df.to_csv(link_csv_path, index=False)
            # print(f"Appended probe link volume to {link_csv_path}")
        else:
            # print("Warning: Either link.csv or probe_link_volume_8_9am.csv not found. Skipping link merge.")
            print('Running DTALite ODME without link volume data...')

    def run_assignment(self):
        """
        Run DTALite assignment in the output folder.
        """
        if self.stop_event and self.stop_event.is_set():
            print("Stopping before DTALite execution.")
            return

        print("Running DTALite ODME...")
        original_path = os.getcwd()
        try:
            os.chdir(self.odme_folder)
            dta.assignment()
        finally:
            os.chdir(original_path)

    def run(self):
        """
        Run the full ODME workflow.
        """
        self.run_assignment()
        print("DTALite ODME completed.")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "input")
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")

    odem_runner_tp = DTALiteRunnerTP(DEFAULT_INPUT_FOLDER, DEFAULT_OUTPUT_FOLDER)
    odem_runner_tp.run()

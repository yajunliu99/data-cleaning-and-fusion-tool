import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class PenetrationRateAnalyzerWP:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize database connection."""
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"ERROR: Database not found at {database_path}")

        self.database_path = database_path
        self.output_folder = output_folder

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

        self.conn = sqlite3.connect(database_path)

    def get_lane_readings(self):
        """Retrieve and aggregate lane volume data by 15-minute intervals, then compute time-of-day averages."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_lane_readings")
            return pd.DataFrame()

        query = """
        SELECT zone_id, volume, local_time FROM lane_readings
        WHERE zone_id IN (223305, 197584, 197063, 196740, 196495, 197309, 197088)
        """
        df_lane = pd.read_sql(query, self.conn)
        df_lane["local_time"] = pd.to_datetime(df_lane["local_time"])
        df_lane["time_bin"] = df_lane["local_time"].dt.floor("15min")
        df_lane["time_of_day"] = df_lane["time_bin"].dt.time
        df_lane["weekday"] = df_lane["local_time"].dt.weekday
        df_lane["is_weekend"] = df_lane["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        # Aggregate total volume per 15-minute interval
        df_lane = df_lane.groupby(["time_bin", "is_weekend"])["volume"].sum().reset_index()

        # Compute daily time-of-day averages
        df_lane["time_of_day"] = df_lane["time_bin"].dt.time  # Ensure only time remains
        df_avg_lane = df_lane.groupby(["time_of_day", "is_weekend"])["volume"].mean().reset_index()

        return df_avg_lane

    def get_map_matching_volumes(self):
        """Retrieve and aggregate unique agent counts by 15-minute intervals, then compute time-of-day averages."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_map_matching_volumes")
            return pd.DataFrame()

        query = """
        SELECT agent_id, time FROM map_matching
        """
        df_map = pd.read_sql(query, self.conn)
        df_map["time"] = pd.to_datetime(df_map["time"])
        df_map["time_bin"] = df_map["time"].dt.floor("15min")
        df_map["time_of_day"] = df_map["time_bin"].dt.time
        df_map["weekday"] = df_map["time"].dt.weekday
        df_map["is_weekend"] = df_map["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        # Aggregate unique agent_id counts per 15-minute interval
        df_map = df_map.groupby(["time_bin", "is_weekend"])["agent_id"].nunique().reset_index()

        # Compute daily time-of-day averages
        df_map["time_of_day"] = df_map["time_bin"].dt.time
        df_avg_map = df_map.groupby(["time_of_day", "is_weekend"])["agent_id"].mean().reset_index()

        return df_avg_map

    def compute_penetration_rate(self, df_lane, df_map):
        """Compute penetration rates based on lane readings, map matching, and trajectory data."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping compute_penetration_rate")
            return

        df_lane = df_lane.rename(columns={"volume": "lane_volume"})
        df_map = df_map.rename(columns={"agent_id": "map_matching_volume"})

        # Merge datasets on time_of_day and is_weekend
        df_merged = df_lane.merge(df_map, on=["time_of_day", "is_weekend"], how="left")

        # Compute penetration rates
        df_merged["map_penetration_rate"] = df_merged["map_matching_volume"] / df_merged["lane_volume"]

        return df_merged

    def plot_penetration_rate(self, df_penetration):
        """Plot penetration rates over time-of-day for weekdays and weekends separately."""
        output_dir = os.path.join(self.output_folder, "advanced_data_fusion", "penetration_rate_wp")
        os.makedirs(output_dir, exist_ok=True)

        for is_weekend, label in [(0, "Weekday"), (1, "Weekend")]:
            if self.stop_event and self.stop_event.is_set():
                print(f"Stopping during plotting {label}")
                return

            df_filtered = df_penetration[df_penetration["is_weekend"] == is_weekend].copy()

            if df_filtered.empty:
                print(f"No data available for {label}, skipping plot.")
                continue

            df_filtered["time_bin"] = df_filtered["time_of_day"]
            df_filtered["time_of_day"] = pd.to_datetime(df_filtered["time_bin"], format="%H:%M:%S")

            plt.figure(figsize=(12, 8))
            plt.xlabel("Time of Day", fontsize=18)
            plt.ylabel("Penetration Rate", fontsize=18)
            plt.title(f"Penetration Rate - {label}", fontsize=18)

            plt.plot(df_filtered["time_of_day"], df_filtered["map_penetration_rate"], linestyle="-", linewidth=3,
                     color="#4472C5", label="Waypoint")

            plt.legend(fontsize=18)
            plt.grid(True)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            plt.tight_layout()
            # plt.show()

            filename = f"penetration_rate_{label.lower()}_wp.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()

    def run(self):
        """Execute penetration rate analysis."""
        if self.stop_event and self.stop_event.is_set():
            return
        df_lane = self.get_lane_readings()

        if self.stop_event and self.stop_event.is_set():
            return
        df_map = self.get_map_matching_volumes()

        if self.stop_event and self.stop_event.is_set():
            return
        df_penetration = self.compute_penetration_rate(df_lane, df_map)
        # print(df_penetration)

        # Save penetration rate data
        if self.stop_event and self.stop_event.is_set():
            return

        penetration_rate_dir = os.path.join(self.output_folder, "advanced_data_fusion", "penetration_rate_wp")
        os.makedirs(penetration_rate_dir, exist_ok=True)

        penetration_rate_path = os.path.join(penetration_rate_dir, "penetration_rates_wp.csv")
        df_penetration.to_csv(penetration_rate_path, encoding="utf-8-sig")

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_penetration_rate(df_penetration)

        self.conn.close()
        print("Penetration Rate Analysis Completed.")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    penetrationrate_analyzer_wp = PenetrationRateAnalyzerWP(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    penetrationrate_analyzer_wp.run()

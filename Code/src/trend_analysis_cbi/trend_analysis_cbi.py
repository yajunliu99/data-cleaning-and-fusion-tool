import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


class TrendAnalyzer:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initializes database connection and checks if the database exists."""
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"ERROR: Database not found at {database_path}")

        self.output_folder = output_folder
        self.database_path = os.path.join(output_folder, "database", "unified_database.db")

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event
        self.conn = sqlite3.connect(database_path)

    def get_tmc_codes(self):
        """Retrieves TMC codes and their corresponding road_order where road_order is not 1."""
        query = "SELECT tmc, road_order FROM TMC_Identification WHERE road_order != 1"
        df_tmc = pd.read_sql(query, self.conn)

        if df_tmc.empty:
            raise ValueError("No valid TMC codes found with road_order != 1.")

        # Convert road_order to integer
        df_tmc["road_order"] = df_tmc["road_order"].astype(int)

        return df_tmc

    def get_readings(self, df_tmc):
        """Fetches speed and local_time for selected TMC codes from the Reading table."""
        placeholders = ",".join(["?"] * len(df_tmc["tmc"]))
        query = f"""
        SELECT tmc_code, speed, local_time
        FROM Readings
        WHERE tmc_code IN ({placeholders})
        """
        df_reading = pd.read_sql(query, self.conn, params=df_tmc["tmc"].tolist())

        if df_reading.empty:
            raise ValueError("No reading data found for the selected TMC codes")

        # Ensure correct data types
        df_reading["local_time"] = pd.to_datetime(df_reading["local_time"])
        df_reading["speed"] = pd.to_numeric(df_reading["speed"], errors="coerce")

        # Merge `road_order` into `df_reading`
        df_reading = df_reading.merge(df_tmc, left_on="tmc_code", right_on="tmc", how="left")

        return df_reading

    def plot_speed_profiles(self, df_reading):
        """Plots 5-minute average speed profiles for each TMC code separately, sorted by road_order."""
        print("Plotting TMC Speed Profile...")

        # Define output directory
        output_dir = os.path.join(self.output_folder, "trend_analysis_cbi",  "tmc_speed_profile")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        # Extract time features
        df_reading["time_bin"] = df_reading["local_time"].dt.floor("5min")
        df_reading["time_of_day"] = df_reading["time_bin"].dt.time
        df_reading["weekday"] = df_reading["local_time"].dt.weekday
        df_reading["is_weekend"] = df_reading["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        # Sort TMC codes by road_order
        sorted_tmc_codes = df_reading[['tmc_code', 'road_order']].drop_duplicates().sort_values('road_order')

        for _, row in sorted_tmc_codes.iterrows():
            if self.stop_event and self.stop_event.is_set():
                print("Stopping during speed profile plotting (outer loop)")
                return

            tmc = row["tmc_code"]
            road_order = row["road_order"]

            for is_weekend, label in [(0, "Weekday"), (1, "Weekend")]:
                if self.stop_event and self.stop_event.is_set():
                    print(f"Stopping at TMC {tmc} ({label})")
                    return

                df_group = df_reading[df_reading["is_weekend"] == is_weekend]

                # Filter data for this specific TMC
                df_filtered = df_group[df_group["tmc_code"] == tmc].copy()

                if df_filtered.empty:
                    continue  # Skip if no data

                # Aggregate by time_of_day and compute average speed
                df_aggregated = df_filtered.groupby("time_of_day")["speed"].mean().reset_index()

                # Convert time_of_day to datetime format for plotting
                df_aggregated["time_of_day"] = pd.to_datetime(df_aggregated["time_of_day"], format="%H:%M:%S")

                # Create a new figure for each TMC code
                plt.figure(figsize=(12, 8))
                plt.plot(df_aggregated["time_of_day"], df_aggregated["speed"], label=f"TMC {tmc}", linewidth=3, color="#4472C5")

                # Enhance plot appearance
                plt.xlabel("Time", fontsize=18)
                plt.ylabel("Speed (mph)", fontsize=18)
                plt.title(f"Speed Profile for TMC {tmc} - Road Order {road_order} ({label})", fontsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.grid(True)
                # plt.legend(fontsize=14)
                plt.tight_layout()
                plt.ylim(5, 79)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                # plt.show()

                # Define output file path (consistent naming)
                filename = f"speed_profile_{tmc}_{road_order:02d}_{label.lower()}.png"
                filepath = os.path.join(output_dir, filename)

                # Save the figure
                plt.savefig(filepath, dpi=300)
                plt.close()  # Close the figure to free memory

    def run(self):
        """Executes the full Trend Analysis and CBI process."""
        if self.stop_event and self.stop_event.is_set():
            return
        tmc_codes = self.get_tmc_codes()

        if self.stop_event and self.stop_event.is_set():
            return
        df_reading = self.get_readings(tmc_codes)

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_speed_profiles(df_reading)

        print("TMC Speed Profile saved")

        self.conn.close()


class CBIAnalyzer:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initializes database connection and checks if the database exists."""
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"ERROR: Database not found at {database_path}")

        self.output_folder = output_folder
        self.database_path = database_path
        self.stop_event = stop_event
        self.conn = sqlite3.connect(database_path)

    def get_tmc_groups(self):
        """Retrieves TMC codes, road_order, and groups them into continuous road segments with the same direction."""
        query = "SELECT tmc, road, direction, road_order FROM TMC_Identification WHERE road_order != 1 ORDER BY direction, road_order"
        df_tmc = pd.read_sql(query, self.conn)

        if df_tmc.empty:
            raise ValueError("No valid TMC codes found with road_order != 1.")

        # Convert road_order to integer
        df_tmc["road_order"] = df_tmc["road_order"].astype(int)

        # Assign a group ID considering both road_order and direction
        df_tmc["group_id"] = df_tmc.groupby("direction")["road_order"].diff().ne(1).cumsum()

        # Extract unique group_id and direction mapping
        group_info = df_tmc[["group_id", "road", "direction", "road_order"]].drop_duplicates().sort_values(["group_id", "road_order"])

        # Print group mapping
        print("=== TMC Corridors ===")
        for _, row in group_info.iterrows():
            print(f"Corridor: {row['group_id']}, Road: {row['road']}, Direction: {row['direction']}, Road Order: {row['road_order']}")
        print("=======================================")

        return df_tmc

    def get_readings(self, df_tmc):
        """Fetches speed and local_time for selected TMC codes from the Reading table."""
        placeholders = ",".join(["?"] * len(df_tmc["tmc"]))
        query = f"""
        SELECT tmc_code, speed, local_time
        FROM Readings
        WHERE tmc_code IN ({placeholders})
        """
        df_reading = pd.read_sql(query, self.conn, params=df_tmc["tmc"].tolist())

        if df_reading.empty:
            raise ValueError("No reading data found for the selected TMC codes on 2024-08-05.")

        # Ensure correct data types
        df_reading["local_time"] = pd.to_datetime(df_reading["local_time"])
        df_reading["speed"] = pd.to_numeric(df_reading["speed"], errors="coerce")

        # Merge `road_order` and `group_id` into `df_reading`
        df_reading = df_reading.merge(df_tmc, left_on="tmc_code", right_on="tmc", how="left")

        return df_reading

    def plot_heatmap(self, df_reading):
        """Plots congestion bottleneck heatmap based on 15-minute speed averages for each road group,
        separately for weekdays and weekends."""

        print("Plotting TMC CBI Heatmap...")

        # Define output directory
        output_dir = os.path.join(self.output_folder, "trend_analysis_cbi", "tmc_cbi_heatmap")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        # Extract time features
        df_reading["time_bin"] = df_reading["local_time"].dt.floor("15min")
        df_reading["time_of_day"] = df_reading["time_bin"].dt.time
        df_reading["weekday"] = df_reading["local_time"].dt.weekday
        df_reading["is_weekend"] = df_reading["weekday"].apply(lambda x: 1 if x >= 5 else 0)

        # Iterate through each road group
        for group_id in sorted(df_reading["group_id"].unique()):
            if self.stop_event and self.stop_event.is_set():
                print("Stopping during heatmap plotting (outer loop)")
                return

            df_group = df_reading[df_reading["group_id"] == group_id].copy()

            for is_weekend, label in [(0, "Weekday"), (1, "Weekend")]:
                if self.stop_event and self.stop_event.is_set():
                    print(f"Stopping at Corridor {group_id} ({label})")
                    return

                df_subset = df_group[df_group["is_weekend"] == is_weekend]

                if df_subset.empty:
                    continue  # Skip if no data for this category

                # Aggregate speed by road_order and time_of_day
                df_aggregated = df_subset.groupby(["road_order", "time_of_day"])["speed"].mean().reset_index()

                # Pivot table for heatmap (road_order as index, time_of_day as columns)
                df_pivot = df_aggregated.pivot(index="road_order", columns="time_of_day", values="speed")

                # Plot heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(df_pivot, cmap="RdBu", annot=False, linewidths=0.5, cbar=True, vmin=35, vmax=65)

                # Customize plot appearance
                plt.xlabel("Time", fontsize=18)
                plt.ylabel("Road Order", fontsize=18)
                plt.title(f"TMC CBI Heatmap - Corridor {group_id} ({label})", fontsize=18)

                # Format x-axis labels
                x_labels = [t.strftime("%H:%M") for t in df_pivot.columns]
                tick_positions = range(0, len(x_labels), 8)  # Show every 4th tick (hourly)
                plt.xticks(ticks=tick_positions, labels=[x_labels[i] for i in tick_positions], fontsize=14, rotation=90)
                plt.yticks(fontsize=14)
                plt.tight_layout()

                # Save figure
                filename = f"tmc_cbi_heatmap_corridor_{group_id}_{label.lower()}.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300)
                plt.close()  # Close figure to free memory

    def run(self):
        """Executes the full CBI analysis process."""
        if self.stop_event and self.stop_event.is_set():
            return
        df_tmc = self.get_tmc_groups()  # Get grouped road segments

        if self.stop_event and self.stop_event.is_set():
            return
        df_reading = self.get_readings(df_tmc)  # Fetch speed data

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_heatmap(df_reading)  # Generate and save heatmap

        print("TMC CBI Heatmap saved.")

        self.conn.close()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    trend_analyzer = TrendAnalyzer(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    trend_analyzer.run()

    cbi_analyzer = CBIAnalyzer(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    cbi_analyzer.run()
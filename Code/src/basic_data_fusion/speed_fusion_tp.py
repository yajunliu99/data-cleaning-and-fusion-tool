import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


class SpeedFusionAggregatorTP:
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
        self.valid_link_ids = set()

    def get_tmc_readings(self):
        """Retrieve all readings data and compute 15-minute average speed per tmc_code."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_tmc_readings")
            return pd.DataFrame()

        query = """
        SELECT tmc_code, speed, local_time 
        FROM Readings
        WHERE tmc_code NOT IN (SELECT tmc FROM TMC_Identification WHERE road_order = 1)
        """
        df_readings = pd.read_sql(query, self.conn)

        if df_readings.empty:
            raise ValueError("No reading data found in Readings table.")

        df_readings["local_time"] = pd.to_datetime(df_readings["local_time"])
        df_readings["speed"] = pd.to_numeric(df_readings["speed"], errors="coerce")
        df_readings["time_bin"] = df_readings["local_time"].dt.floor("15min")
        df_avg_speed = df_readings.groupby(["tmc_code", "time_bin"])["speed"].mean().reset_index()

        return df_avg_speed

    def map_tmc_to_link(self, df_avg_speed):
        """Map tmc_code speeds to corresponding link_ids using tmc_to_link table, excluding road_order=1."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping map_tmc_to_link")
            return pd.DataFrame()

        query = """
        SELECT tmc, link_ids FROM tmc_to_link
        WHERE tmc NOT IN (SELECT tmc FROM TMC_Identification WHERE road_order = 1)
        """
        df_mapping = pd.read_sql(query, self.conn)

        if df_mapping.empty:
            raise ValueError("No mapping data found in tmc_to_link table.")

        df_mapping["link_ids"] = df_mapping["link_ids"].str.split(",")
        df_mapping = df_mapping.explode("link_ids").rename(columns={"link_ids": "link_id"})

        df_avg_speed = df_avg_speed.merge(df_mapping, left_on="tmc_code", right_on="tmc", how="left")
        df_avg_speed.drop(columns=["tmc"], inplace=True)

        self.valid_link_ids = set(df_mapping["link_id"].dropna().astype(str))

        return df_avg_speed

    def get_trajs_speed(self):
        """Retrieve trajs data and compute 15-minute average speed per SegmentId, excluding invalid link_id."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_segment_readings")
            return pd.DataFrame()

        query = """
        SELECT SegmentId, CrossingStartDateLocal, CrossingSpeedMph 
        FROM trajs
        """
        df_trajs = pd.read_sql(query, self.conn)

        if df_trajs.empty:
            raise ValueError("No trajs data found in trajs table.")

        # df_trajs["SegmentId"] = df_trajs["SegmentId"].astype(str)

        segment_link_df = pd.read_sql("SELECT SegmentId, link_id FROM SegmentId_to_link", self.conn)
        segment_link_df["SegmentId"] = segment_link_df["SegmentId"].astype(str)

        df_trajs = pd.merge(df_trajs, segment_link_df, on="SegmentId", how="left")
        df_trajs["link_id"] = df_trajs["link_id"].astype(str)

        df_trajs = df_trajs[df_trajs["link_id"].isin(self.valid_link_ids)]

        df_trajs["CrossingStartDateLocal"] = pd.to_datetime(df_trajs["CrossingStartDateLocal"])
        df_trajs["CrossingSpeedMph"] = pd.to_numeric(df_trajs["CrossingSpeedMph"], errors="coerce")
        df_trajs["time_bin"] = df_trajs["CrossingStartDateLocal"].dt.floor("15min")

        df_avg_speed = df_trajs.groupby(["link_id", "time_bin"])["CrossingSpeedMph"].mean().reset_index()

        df_avg_speed.rename(columns={"CrossingSpeedMph": "speed"}, inplace=True)

        return df_avg_speed

    def get_tmc_groups(self):
        """Retrieves TMC codes, road_order, and groups them into continuous road segments with the same direction."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_tmc_groups")
            return pd.DataFrame()

        query = """
        SELECT tmc, road, direction, road_order FROM TMC_Identification 
        WHERE road_order != 1 ORDER BY direction, road_order"""
        df_tmc = pd.read_sql(query, self.conn)

        if df_tmc.empty:
            raise ValueError("No valid TMC codes found with road_order != 1.")

        # Convert road_order to integer
        df_tmc["road_order"] = df_tmc["road_order"].astype(int)

        # Assign a group ID considering both road_order and direction
        df_tmc["group_id"] = df_tmc.groupby("direction")["road_order"].diff().ne(1).cumsum()

        return df_tmc

    def get_grouped_links(self, df_tmc):
        """Finds link_ids corresponding to each TMC group in the correct order."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping get_grouped_links")
            return pd.DataFrame()

        query = """
        SELECT tmc, link_ids FROM tmc_to_link"""
        df_tmc_to_link = pd.read_sql(query, self.conn)

        if df_tmc_to_link.empty:
            raise ValueError("No mapping data found in tmc_to_link table.")

        # Explode link_ids into separate rows
        df_tmc_to_link["link_ids"] = df_tmc_to_link["link_ids"].str.split(",")
        df_tmc_to_link = df_tmc_to_link.explode("link_ids").rename(columns={"link_ids": "link_id"})

        # Merge TMC group information with link data
        df_grouped_links = df_tmc.merge(df_tmc_to_link, on="tmc", how="left").dropna()

        # Ensure correct ordering based on road_order
        df_grouped_links = df_grouped_links.sort_values(["group_id", "road_order"]).drop_duplicates(
            ["group_id", "link_id"])

        # Group by group_id and aggregate links
        grouped_links = df_grouped_links.groupby("group_id")["link_id"].apply(list).reset_index()

        # Save to CSV
        output_dir = os.path.join(self.output_folder, "basic_data_fusion")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "grouped_links.csv")
        grouped_links.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("Grouped Links Extraction Completed.")
        return grouped_links

    def plot_link_speed_profiles(self, df_final_speed):
        """Plots 5-minute average speed profiles for each link_id separately."""
        print("Plotting Link Speed Profile...")

        # Define output directory
        output_dir = os.path.join(self.output_folder, "basic_data_fusion", "link_speed_profile_tp")
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        # Extract time features
        df_final_speed["time_bin"] = df_final_speed["time_of_day"]
        df_final_speed["time_of_day"] = pd.to_datetime(df_final_speed["time_bin"], format="%H:%M:%S")

        # Get unique link_ids
        unique_links = df_final_speed["link_id"].unique()

        for link_id in unique_links:
            for is_weekend, label in [(0, "Weekday"), (1, "Weekend")]:
                if self.stop_event and self.stop_event.is_set():
                    print(f"Stopping link_id {link_id} speed profile plotting")
                    return

                # Filter data for this specific link_id and weekday/weekend condition
                df_filtered = df_final_speed[
                    (df_final_speed["link_id"] == link_id) & (df_final_speed["is_weekend"] == is_weekend)
                    ].copy()

                if df_filtered.empty:
                    continue  # Skip if no data

                # Aggregate by time_of_day and compute average speed
                df_aggregated = df_filtered.groupby("time_of_day")["speed"].mean().reset_index()

                # Convert time_of_day to datetime format for plotting
                df_aggregated["time_of_day"] = pd.to_datetime(df_aggregated["time_of_day"], format="%H:%M:%S")

                # Create a new figure for each link_id
                plt.figure(figsize=(12, 8))
                plt.plot(df_aggregated["time_of_day"], df_aggregated["speed"], label=f"Link {link_id}", linewidth=3, color="#4472C5")

                # Enhance plot appearance
                plt.xlabel("Time", fontsize=18)
                plt.ylabel("Speed (mph)", fontsize=18)
                plt.title(f"Speed Profile for Link {link_id} ({label})", fontsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.grid(True)
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                plt.ylim(5, 79)
                plt.tight_layout()

                # Define output file path
                filename = f"speed_profile_link_{link_id}_{label.lower()}_tp.png"
                filepath = os.path.join(output_dir, filename)

                # Save the figure
                plt.savefig(filepath, dpi=300)
                plt.close()  # Close the figure to free memory

        print("Link Speed Profile saved.")

    def plot_heatmap(self, df_final_speed, grouped_links):
        """Plots congestion bottleneck heatmap based on 15-minute speed averages for each road group."""
        print("Plotting Link CBI Heatmap...")

        output_dir = os.path.join(self.output_folder, "basic_data_fusion", "link_cbi_heatmap_tp")
        os.makedirs(output_dir, exist_ok=True)

        for _, row in grouped_links.iterrows():
            if self.stop_event and self.stop_event.is_set():
                print(f"Stopping heatmap group {row['group_id']}")
                return

            group_id = row["group_id"]
            link_ids = row["link_id"]

            df_group = df_final_speed[df_final_speed["link_id"].isin(link_ids)].copy()

            if df_group.empty:
                continue

            df_group["time_bin"] = pd.to_datetime(df_group["time_of_day"], format='%H:%M:%S')

            for is_weekend, label in [(0, "Weekday"), (1, "Weekend")]:
                df_filtered = df_group[df_group["is_weekend"] == is_weekend]
                df_resampled = df_filtered.groupby(["link_id", "time_bin"])["speed"].mean().reset_index()

                df_pivot = df_resampled.pivot(index="link_id", columns="time_bin", values="speed")

                df_pivot = df_pivot.loc[link_ids]

                plt.figure(figsize=(12, 8))
                sns.heatmap(df_pivot, cmap="RdBu", annot=False, linewidths=0.5, cbar=True, vmin=35, vmax=65)

                plt.xlabel("Time", fontsize=18)
                plt.ylabel("Link ID", fontsize=18)
                plt.title(f"Link CBI Heatmap - Corridor {group_id} ({label})", fontsize=18)
                x_labels = df_pivot.columns.strftime("%H:%M")
                tick_positions = range(0, len(x_labels), 8)  # Show every 4th tick (hourly)
                plt.xticks(ticks=tick_positions, labels=[x_labels[i] for i in tick_positions], fontsize=14, rotation=90)
                plt.yticks(fontsize=14)
                plt.tight_layout()

                filename = f"link_cbi_heatmap_corridor_{group_id}_{label.lower()}_tp.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300)
                plt.close()

        print("Link CBI Heatmap saved.")

    def run(self):
        """Execute the data processing pipeline."""
        if self.stop_event and self.stop_event.is_set():
            return
        df_tmc_speed = self.get_tmc_readings()

        if self.stop_event and self.stop_event.is_set():
            return
        df_mapped_tmc_speed = self.map_tmc_to_link(df_tmc_speed)

        if self.stop_event and self.stop_event.is_set():
            return
        df_trajs_speed = self.get_trajs_speed()

        df_final_speed = pd.concat([df_mapped_tmc_speed, df_trajs_speed], ignore_index=True)
        df_final_speed["time_of_day"] = pd.to_datetime(df_final_speed["time_bin"]).dt.time
        df_final_speed["weekday"] = pd.to_datetime(df_final_speed["time_bin"]).dt.weekday
        df_final_speed["is_weekend"] = df_final_speed["weekday"].apply(lambda x: 1 if x >= 5 else 0)
        df_final_speed = df_final_speed.groupby(["link_id", "time_of_day", "is_weekend"])['speed'].mean().reset_index()

        if self.stop_event and self.stop_event.is_set():
            return
        df_tmc_groups = self.get_tmc_groups()

        if self.stop_event and self.stop_event.is_set():
            return
        grouped_links = self.get_grouped_links(df_tmc_groups)

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_link_speed_profiles(df_final_speed)

        if self.stop_event and self.stop_event.is_set():
            return
        self.plot_heatmap(df_final_speed, grouped_links)

        df_final_speed["link_id"] = pd.to_numeric(df_final_speed["link_id"])
        df_final_speed = df_final_speed.sort_values(by=["link_id", "time_of_day"])

        output_dir = os.path.join(self.output_folder, "basic_data_fusion")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "speed_fusion_tp.csv")
        df_final_speed.to_csv(output_file, index=False, encoding="utf-8-sig")

        self.conn.close()
        print("Speed Fusion Completed.")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    speed_aggregator_tp = SpeedFusionAggregatorTP(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    speed_aggregator_tp.run()
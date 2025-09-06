import os
import sqlite3
import pandas as pd


class ProbeLinkAnalyzer:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initialize the database path."""
        self.database_path = database_path
        self.output_folder = output_folder

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

        self.conn = sqlite3.connect(database_path)
        self.penetration_rate_df = None  # Placeholder for penetration rate data

    def load_data(self):
        """Load node, link, and map_matching data from the SQLite database."""
        conn = sqlite3.connect(self.database_path)

        # Read node data
        node_df = pd.read_sql("SELECT node_id, is_boundary FROM node", conn)

        # Read link data
        link_df = pd.read_sql("SELECT link_id, from_node_id, to_node_id FROM link", conn)

        # Read zone_name_to_link data
        zone_name_to_link_df = pd.read_sql("SELECT Zone_Name, link_id FROM zone_name_to_link", conn)

        conn.close()
        return node_df, link_df, zone_name_to_link_df

    def extract_link_volume(self, filter_day_parts, output_filename):
        """
        Extract traffic volume for each Zone_Name from the 'volume' table,
        """
        if self.stop_event and self.stop_event.is_set():
            print("Stopping extract_link_volume")
            return

        # Connect to the database and retrieve volume data
        with sqlite3.connect(self.database_path) as conn:
            volume_df = pd.read_sql(
                'SELECT Zone_Name, Day_Type, Day_Part, "Average_Daily_Segment_Traffic_(StL_Volume)" FROM volume',
                conn)

        volume_df["Day_Type"] = volume_df["Day_Type"].astype(str).str.extract(r"^(\d+)").astype(int)
        volume_df["Day_Part"] = volume_df["Day_Part"].astype(str).str.extract(r"^(\d+)").astype(int)

        volume_df = volume_df[(volume_df["Day_Type"] == 0) & (volume_df["Day_Part"].isin(filter_day_parts))]

        _, _, zone_name_to_link_df = self.load_data()

        zone_name_to_link_df["Zone_Name"] = zone_name_to_link_df["Zone_Name"].str.strip().str.lower()
        volume_df["Zone_Name"] = volume_df["Zone_Name"].str.strip().str.lower()

        volume_df = volume_df.merge(zone_name_to_link_df, on="Zone_Name", how="left")

        all_links = zone_name_to_link_df["link_id"].unique()

        zone_volume_matrix = volume_df.groupby("link_id")["Average_Daily_Segment_Traffic_(StL_Volume)"].sum().reset_index()
        # zone_volume_matrix = volume_df.groupby(["Zone_Name", "link_id"])["Average_Daily_Segment_Traffic_(StL_Volume)"].sum().reset_index()
        zone_volume_matrix.rename(columns={"Average_Daily_Segment_Traffic_(StL_Volume)": "volume"}, inplace=True)

        full_index = pd.DataFrame({"link_id": all_links})
        zone_volume_matrix = full_index.merge(zone_volume_matrix, on="link_id", how="left")
        # zone_volume_matrix = full_index.merge(zone_volume_matrix, on="link_id", how="left").fillna(0)

        output_dir = os.path.join(self.output_folder, "odme_simulation", "probe_link_volume")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        zone_volume_matrix.to_csv(output_path, encoding="utf-8-sig", index=False)

        # print(f"Link volume data saved to: {output_path}")

    def run(self):
        """Run the full pipeline: Compute penetration rates, then compute OD matrix."""
        if self.stop_event and self.stop_event.is_set():
            return

        # Check if 'volume' table exists
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='volume';
            """)
            table_exists = cursor.fetchone()

        if not table_exists:
            print("Link volume data not found. Skipping...")
        else:
            self.extract_link_volume([0], "probe_link_volume_all_day.csv")
            self.extract_link_volume([9], "probe_link_volume_8_9am.csv")

        print("Probe Link Data Extraction Completed.")
        self.conn.close()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    probe_link_analyzer = ProbeLinkAnalyzer(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    probe_link_analyzer.run()
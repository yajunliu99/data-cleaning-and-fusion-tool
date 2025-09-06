import os
import sqlite3
import pandas as pd


class BasicDataCleaner:
    def __init__(self, database_path, stop_event=None):
        """Initializes database connection and checks if the database exists."""
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"ERROR: Database not found at {database_path}")

        self.database_path = database_path

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

        self.conn = sqlite3.connect(database_path)

    def filter_trajs_by_segment(self):
        """Removes trajs rows where SegmentId is not in the link table and reports removed data percentage."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping trajs filtering")
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trajs';")
        if not cursor.fetchone():
            print("Trip Path data not found, skipping filtering trajs by segment.")
            return

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='SegmentId_to_link';")
        if not cursor.fetchone():
            print("SegmentId_to_link not found, skipping filtering trajs by segment.")
            return

        print("Filtering trajs on corridor by SegmentId...")

        df_trajs = pd.read_sql("SELECT * FROM trajs", self.conn)
        df_links = pd.read_sql("SELECT SegmentId FROM SegmentId_to_link", self.conn)

        # Get valid SegmentIds
        valid_segments = set(df_links["SegmentId"])

        # Original size
        original_size = len(df_trajs)

        # Filter data
        df_trajs_filtered = df_trajs[df_trajs["SegmentId"].isin(valid_segments)]

        # Compute percentage of removed rows
        removed_percentage = (1 - len(df_trajs_filtered) / original_size) * 100
        print(f"Removed {removed_percentage:.2f}% of out-of-network data from trip path")

        # Save back to database
        df_trajs_filtered.to_sql("trajs", self.conn, if_exists="replace", index=False)

    def filter_trajs_error_codes(self):
        """
        Removes rows in the 'trajs' table where the 'ErrorCodes' column is not empty,
        and reports the percentage of removed rows.
        """
        if self.stop_event and self.stop_event.is_set():
            print("Stopping filtering of trajs based on ErrorCodes")
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trajs';")
        if not cursor.fetchone():
            print("Trip Path data not found, skipping filtering by ErrorCodes.")
            return

        print("Filtering trajs rows with ErrorCodes...")

        # Read the trajs table
        df_trajs = pd.read_sql("SELECT * FROM trajs", self.conn)
        original_size = len(df_trajs)

        # Filter rows where 'ErrorCodes' is either NULL or an empty string
        df_trajs_filtered = df_trajs[(df_trajs["ErrorCodes"].isnull()) | (df_trajs["ErrorCodes"] == "")]

        removed_percentage = (1 - len(df_trajs_filtered) / original_size) * 100
        print(f"Removed {removed_percentage:.2f}% of trip path with ErrorCodes.")

        # Save the filtered data back to the trajs table
        df_trajs_filtered.to_sql("trajs", self.conn, if_exists="replace", index=False)

    def deduplicate_waypoint(self):
        """Removes duplicate waypoint entries based on journey_id and capture_time and reports removed percentage."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping waypoint deduplication")
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='waypoint';")
        if not cursor.fetchone():
            print("Waypoint data not found, skipping deduplication.")
            return

        print("Deduplicating waypoint...")

        df_waypoint = pd.read_sql("SELECT * FROM waypoint", self.conn)

        # Original size
        original_size = len(df_waypoint)

        # Remove duplicates
        df_waypoint_cleaned = df_waypoint.drop_duplicates(subset=["journey_id", "capture_time"], keep="first")

        # Compute percentage of removed duplicates
        removed_percentage = (1 - len(df_waypoint_cleaned) / original_size) * 100
        print(f"Removed {removed_percentage:.2f}% of duplicate data from waypoint")

        # Save back to database
        df_waypoint_cleaned.to_sql("waypoint", self.conn, if_exists="replace", index=False)

    def split_data_by_time_gap(self, group, time_gap=300):
        """Splits data into segments based on time gaps greater than `time_gap` seconds."""
        segments = []
        start_idx = 0
        for i in range(1, len(group)):
            if group["capture_time"].iloc[i] - group["capture_time"].iloc[i - 1] > time_gap:
                segments.append(group.iloc[start_idx:i])
                start_idx = i
        segments.append(group.iloc[start_idx:])
        return segments

    def calculate_missing_data_points(self, segment):
        """Calculates the number of missing data points in a segment based on 3-second intervals."""
        expected_data_points = (segment["capture_time"].max() - segment["capture_time"].min()) // 3 + 1
        actual_data_points = len(segment)
        missing_data_points = expected_data_points - actual_data_points
        return missing_data_points, expected_data_points

    def detect_missing_waypoint_data(self):
        """Identifies missing data in waypoint based on expected 3-second intervals and time gaps."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping missing data detection")
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='waypoint';")
        if not cursor.fetchone():
            print("Waypoint data not found, skipping missing data detection.")
            return

        print("Detecting fuzzed and missing data in waypoint...")

        df_waypoint = pd.read_sql("SELECT * FROM waypoint", self.conn)

        if df_waypoint.empty:
            print("Skipping fuzzed and missing data detection: waypoint table is empty.")
            return

        # Count fuzzed points before filtering
        total_rows = len(df_waypoint)
        fuzzed_count = df_waypoint["fuzzed_point"].eq("1").sum()
        fuzzed_percentage = (fuzzed_count / total_rows) * 100 if total_rows > 0 else 0

        # Sort by journey_id and capture_time
        df_waypoint = df_waypoint.sort_values(by=["journey_id", "capture_time"])

        # Remove rows where fuzzed_point is True
        df_waypoint = df_waypoint[df_waypoint["fuzzed_point"] != "1"]

        # Group by journey_id
        grouped = df_waypoint.groupby("journey_id", group_keys=False)

        # Split data into segments and calculate missing data points
        missing_data_points = []
        expected_data_points = []
        journey_ids = []  # Store journey IDs for corresponding segments

        for journey_id, group in grouped:
            if self.stop_event and self.stop_event.is_set():
                print(f"Stopping during journey_id {journey_id}")
                return

            for segment in self.split_data_by_time_gap(group):
                if self.stop_event and self.stop_event.is_set():
                    print(f"Stopping during segment of journey_id {journey_id}")
                    return

                segment = segment.copy()
                segment.loc[:, "journey_id"] = journey_id  # Ensure journey_id consistency

                missing, expected = self.calculate_missing_data_points(segment)
                missing_data_points.append(missing)
                expected_data_points.append(expected)
                journey_ids.append(journey_id)  # Store the corresponding journey_id

        # Compute missing percentage per segment
        missing_percentages = [(missing / expected) * 100 if expected > 0 else 0
                               for missing, expected in zip(missing_data_points, expected_data_points)]

        # # Display results per journey_id segment
        # for journey_id, missing, expected, percentage in zip(journey_ids, missing_data_points, expected_data_points, missing_percentages):
        #     print(f"Journey ID {journey_id}: Missing {missing}/{expected} data points ({percentage:.2f}%)")

        # Compute overall missing percentage
        total_missing = sum(missing_data_points)
        total_expected = sum(expected_data_points)
        overall_missing_percentage = (total_missing / total_expected) * 100 if total_expected > 0 else 0

        print(f"Overall fuzzed points percentage in waypoint: {fuzzed_percentage:.2f}%")
        print(f"Overall missing data points percentage in waypoint: {overall_missing_percentage:.2f}%")

    def run(self):
        """Runs all data cleaning and processing tasks."""
        if self.stop_event and self.stop_event.is_set():
            return
        self.filter_trajs_by_segment()

        if self.stop_event and self.stop_event.is_set():
            return
        self.filter_trajs_error_codes()

        if self.stop_event and self.stop_event.is_set():
            return
        self.deduplicate_waypoint()

        if self.stop_event and self.stop_event.is_set():
            return
        self.detect_missing_waypoint_data()

        self.conn.close()
        print("Data processing completed")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")

    cleaner = BasicDataCleaner(DEFAULT_DATABASE_PATH)
    cleaner.run()
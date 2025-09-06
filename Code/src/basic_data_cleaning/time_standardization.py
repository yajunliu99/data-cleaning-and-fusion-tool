import os
import sqlite3
import pandas as pd
import pytz
from datetime import datetime, timezone


class TimeStandardizationProcessor:
    def __init__(self, database_path, output_folder, stop_event=None):
        """Initializes database connection and output folder paths."""
        self.database_path = database_path
        self.output_folder = output_folder

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event

        if not os.path.exists(database_path):
            raise FileNotFoundError(f"ERROR: Database not found at {database_path}")

        self.conn = sqlite3.connect(database_path)

    def remove_timezone_from_timestamp(self, timestamp):
        """Removes timezone information from timestamp strings."""
        try:
            cleaned_timestamp = timestamp[:-3]
            dt = datetime.strptime(cleaned_timestamp, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return timestamp

    def convert_iso_to_local(self, iso_time, timezone_name):
        """Converts ISO 8601 UTC timestamps to local time."""
        try:
            if pd.isna(iso_time) or iso_time is None or pd.isna(timezone_name):
                return None

            dt = datetime.strptime(iso_time, "%Y-%m-%dT%H:%M:%S.%fZ")
            dt = dt.replace(tzinfo=pytz.utc)

            local_tz = pytz.timezone(timezone_name)
            local_dt = dt.astimezone(local_tz)

            return local_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Error converting time: {iso_time}, timezone: {timezone_name} -> {e}")
            return None

    def convert_unix_to_local(self, unix_time, timezone_name):
        """Converts Unix timestamps to local time using a given timezone."""
        try:
            if pd.isna(unix_time) or unix_time is None or pd.isna(timezone_name):
                return None

            dt = datetime.fromtimestamp(int(unix_time), tz=timezone.utc)

            local_tz = pytz.timezone(timezone_name)
            local_dt = dt.astimezone(local_tz)

            return local_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"Error converting time: {unix_time}, timezone: {timezone_name} -> {e}")
            return None

    def process_lane_readings(self):
        """Processes 'lane_readings' table by removing timezone information from timestamps."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping lane_readings")
            return

        print("Processing lane_readings...")
        df = pd.read_sql("SELECT * FROM lane_readings", self.conn)
        df["local_time"] = df["measurement_start"].apply(self.remove_timezone_from_timestamp)
        df.to_sql("lane_readings", self.conn, if_exists="replace", index=False)
        print("lane_readings updated")

    def process_readings(self):
        """Processes 'Readings' table by standardizing timestamp formats."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping Readings")
            return

        print("Processing Readings...")
        df = pd.read_sql("SELECT * FROM Readings", self.conn)
        df["local_time"] = df["measurement_tstamp"].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        )
        df.to_sql("Readings", self.conn, if_exists="replace", index=False)
        print("Readings updated")

    def process_trajs(self):
        """Processes 'trajs' table by converting timestamps and speed values."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping trajs")
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trajs';")
        if not cursor.fetchone():
            print("Trip Path not found, skipping...")
            return

        print("Processing trip path...")

        df = pd.read_sql("SELECT * FROM trajs", self.conn)
        df_trips = pd.read_sql("SELECT TripTimezone FROM trajs LIMIT 1", self.conn)

        if df_trips.empty:
            print("ERROR: No data found in 'trajs' table.")
            return

        trip_timezone = df_trips["TripTimezone"].iloc[0]
        limit = None  # Set limit to None to process the full dataset

        df_subset = df.head(limit).copy() if limit else df.copy()

        df_subset["CrossingStartDateLocal"] = df_subset["CrossingStartDateUtc"].apply(
            lambda iso_time: self.convert_iso_to_local(iso_time, trip_timezone)
        )
        df_subset["CrossingEndDateLocal"] = df_subset["CrossingEndDateUtc"].apply(
            lambda iso_time: self.convert_iso_to_local(iso_time, trip_timezone)
        )

        df_subset["CrossingSpeedMph"] = df_subset["CrossingSpeedKph"] * 0.621371

        df.loc[df_subset.index, "CrossingStartDateLocal"] = df_subset["CrossingStartDateLocal"]
        df.loc[df_subset.index, "CrossingEndDateLocal"] = df_subset["CrossingEndDateLocal"]
        df.loc[df_subset.index, "CrossingSpeedMph"] = df_subset["CrossingSpeedMph"]

        df.to_sql("trajs", self.conn, if_exists="replace", index=False)

        print("trip path updated")

    def process_waypoint(self):
        """Processes 'waypoint' table by converting Unix timestamps to local time."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping waypoint")
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='waypoint';")
        if not cursor.fetchone():
            print("Waypoint data not found, skipping...")
            return

        print("Processing waypoint...")

        df_waypoint = pd.read_sql("SELECT * FROM waypoint", self.conn)
        df_trajs = pd.read_sql("SELECT TripTimezone FROM trajs LIMIT 1", self.conn)

        if df_waypoint.empty:
            print("ERROR: No data found in 'waypoint' table.")
            return

        if df_trajs.empty:
            print("ERROR: No data found in 'trajs' table.")
            return

        trip_timezone = df_trajs["TripTimezone"].iloc[0]

        df_waypoint["local_time"] = df_waypoint["capture_time"].apply(
            lambda x: self.convert_unix_to_local(x, trip_timezone)
        )

        df_waypoint.to_sql("waypoint", self.conn, if_exists="replace", index=False)
        print("waypoint updated")

    def export_database_schema(self):
        """Extracts and saves the SQLite database schema to a text file."""
        if self.stop_event and self.stop_event.is_set():
            print("Stopping schema export")
            return

        schema_export_path = os.path.join(self.output_folder, "database", "schema.txt")

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("ERROR: No tables found in the database.")
            return

        print("Exporting schema from database...")

        with open(schema_export_path, "w", encoding="utf-8") as f:
            for table_name, in tables:
                f.write(f"TABLE: {table_name}\n")
                f.write("=" * (len(table_name) + 7) + "\n")

                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()

                for col in columns:
                    col_id, col_name, col_type, is_not_null, default_val, pk = col
                    f.write(f"{col_name} ({col_type}) {'PRIMARY KEY' if pk else ''}\n")

                f.write("\n")

        print("Database schema exported")

    def run(self):
        """Runs all data processing tasks sequentially."""
        if self.stop_event and self.stop_event.is_set():
            return
        self.process_lane_readings()

        if self.stop_event and self.stop_event.is_set():
            return
        self.process_readings()

        if self.stop_event and self.stop_event.is_set():
            return
        self.process_trajs()

        if self.stop_event and self.stop_event.is_set():
            return
        self.process_waypoint()

        if self.stop_event and self.stop_event.is_set():
            return
        self.export_database_schema()

        self.conn.close()
        print("Data processing completed")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_DATABASE_PATH = os.path.join(PROJECT_ROOT, "data", "output", "database", "unified_database.db")
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")

    processor = TimeStandardizationProcessor(DEFAULT_DATABASE_PATH, DEFAULT_OUTPUT_FOLDER)
    processor.run()
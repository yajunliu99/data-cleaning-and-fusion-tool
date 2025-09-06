import os
import sqlite3
import chardet
import pandas as pd

class CSVToSQLiteProcessor:
    def __init__(self, input_folder, output_folder, stop_event=None):
        """ Initializes paths and database connection. """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.conn, self.database_path = self._initialize_database()
        self.cursor = self.conn.cursor()

        if stop_event is None:
            print("WARNING: stop_event is None, stopping may not work!")
        self.stop_event = stop_event


    def _initialize_database(self):
        """ Ensures the output folder exists and initializes SQLite database. """
        database_dir = os.path.join(self.output_folder, "database")
        database_path = os.path.join(database_dir, "unified_database.db")

        if not os.path.exists(database_dir):
            os.makedirs(database_dir)

        conn = sqlite3.connect(database_path)
        return conn, database_path

    def _infer_column_types(self, df):
        """ Infers SQLite column types based on the data in the DataFrame. """
        column_types = {}
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col].dropna()):
                column_types[col] = "INTEGER"
            elif pd.api.types.is_float_dtype(df[col].dropna()):
                column_types[col] = "REAL"
            else:
                column_types[col] = "TEXT"
        return column_types

    def detect_encoding(self, file_path):
        """Detects the encoding of a given file."""
        with open(file_path, "rb") as f:
            result = chardet.detect(f.read(50000))
        return result["encoding"]

    def _read_csv_with_headers(self, file_path, headers_path=None):
        """ Reads a CSV file and assigns headers if missing. """
        dtype_map = {16: "str"}

        encoding = self.detect_encoding(file_path)

        if headers_path and os.path.exists(headers_path):
            headers = pd.read_csv(headers_path, header=None).iloc[0].tolist()
            df = pd.read_csv(file_path, header=None, names=headers, dtype=dtype_map, encoding=encoding, low_memory=False)
        else:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)

        return df

    def _create_table_with_types(self, table_name, column_types):
        """ Creates a table with the correct column types in SQLite. """
        columns_def = ", ".join([f'"{col}" {dtype}' for col, dtype in column_types.items()])
        sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_def});'
        self.cursor.execute(sql)

    def process_csv_files(self):
        """ Reads CSV files from input_folder and stores them in SQLite in output_folder. """
        if not os.path.exists(self.input_folder):
            print(f"ERROR: Input folder '{self.input_folder}' does not exist")
            return

        for subdir in os.listdir(self.input_folder):
            if self.stop_event and self.stop_event.is_set():
                print("Stopping Database processing...")
                return

            subdir_path = os.path.join(self.input_folder, subdir)

            if os.path.isdir(subdir_path):  # Ensure it's a folder
                print(f"Processing folder: {subdir}...")

                if subdir.lower() == "waypoint":
                    waypoint_table_created = False
                    for file in os.listdir(subdir_path):
                        if file.startswith("._"):
                            print(f"Skipping hidden file: {file}")
                            continue

                        if not file.endswith(".csv"):
                            continue

                        file_path = os.path.join(subdir_path, file)
                        if self.stop_event and self.stop_event.is_set():
                            print("Stopping before processing CSV file...")
                            return

                        encoding = self.detect_encoding(file_path)
                        print(f"Encoding detected for {file}: {encoding}")

                        if self.stop_event and self.stop_event.is_set():
                            print(f"Stopping before reading {file}...")
                            return

                        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)

                        if self.stop_event and self.stop_event.is_set():
                            print(f"Stopping before processing DataFrame for {file}...")
                            return

                        df.columns = [col.replace(" ", "_").replace("-", "_") for col in df.columns]

                        column_types = self._infer_column_types(df)

                        if self.stop_event and self.stop_event.is_set():
                            print(f"Stopping before inserting {file} into database...")
                            return

                        if not waypoint_table_created:
                            self._create_table_with_types("waypoint", column_types)
                            to_sql_mode = "replace"
                            waypoint_table_created = True
                        else:
                            to_sql_mode = "append"

                        df.to_sql("waypoint", self.conn, if_exists=to_sql_mode, index=False, dtype=column_types)

                        if self.stop_event and self.stop_event.is_set():
                            print(f"Stopping after inserting {file} into database...")
                            return

                else:
                    for file in os.listdir(subdir_path):
                        if file.startswith("._"):
                            print(f"Skipping hidden file: {file}")
                            continue

                        if self.stop_event and self.stop_event.is_set():
                            print("Stopping before processing CSV file...")
                            return

                        if file.endswith(".csv"):
                            file_path = os.path.join(subdir_path, file)
                            encoding = self.detect_encoding(file_path)
                            print(f"Encoding detected for {file}: {encoding}")

                            if file == "TripBulkReportTrajectoriesHeaders.csv":
                                continue

                            table_name = file.replace(".csv", "").replace("-", "_")

                            if self.stop_event and self.stop_event.is_set():
                                print(f"Stopping before reading {file}...")
                                return

                            if file == "trajs.csv":
                                headers_path = os.path.join(subdir_path, "TripBulkReportTrajectoriesHeaders.csv")
                                df = self._read_csv_with_headers(file_path, headers_path)
                            else:
                                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)

                            if self.stop_event and self.stop_event.is_set():
                                print(f"Stopping before processing DataFrame for {file}...")
                                return

                            df.columns = [col.replace(" ", "_").replace("-", "_") for col in df.columns]
                            column_types = self._infer_column_types(df)

                            self._create_table_with_types(table_name, column_types)

                            if self.stop_event and self.stop_event.is_set():
                                print(f"Stopping before inserting {file} into database...")
                                return

                            df.to_sql(table_name, self.conn, if_exists="replace", index=False, dtype=column_types)

                            if self.stop_event and self.stop_event.is_set():
                                print(f"Stopping after inserting {file} into database...")
                                return

        if self.stop_event and self.stop_event.is_set():
            print("Stopping before committing database changes...")
            return

        self.conn.commit()
        self.conn.close()
        print("SQLite database is ready")

    def run(self):
        """ Main function to process CSV files and store them in the database. """
        self.process_csv_files()


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DEFAULT_INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "input")
    DEFAULT_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")

    processor = CSVToSQLiteProcessor(DEFAULT_INPUT_FOLDER, DEFAULT_OUTPUT_FOLDER)
    processor.run()
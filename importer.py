import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.sql import quoted_name
import io
import re
import chardet
import os
from typing import List, Dict, Optional, Union
from datetime import datetime


# Configuration
class DBConfig:
    def __init__(self):
        self.USER = os.getenv('DB_USER', 'postgres')
        self.PASSWORD = os.getenv('DB_PASSWORD', '123456')
        self.HOST = os.getenv('DB_HOST', 'localhost')
        self.PORT = os.getenv('DB_PORT', '5432')
        self.NAME = os.getenv('DB_NAME', 'mydb')
        
    @property
    def connection_string(self):
        return f"postgresql+psycopg2://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}"

# Database Manager
class DatabaseManager:
    def __init__(self):
        self.config = DBConfig()
        self.engine = create_engine(self.config.connection_string)
        
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        try:
            with self.engine.connect() as conn:
                return pd.read_sql_query(text(query), conn)
        except Exception as e:
            st.error(f"❌ SQL Execution Failed: {e}")
            return None
            
    def create_table(self, table_name: str, columns: Dict[str, str]) -> bool:
        try:
            safe_table_name = quoted_name(table_name, quote=True)
            column_defs = ", ".join([f"{quoted_name(col, quote=True)} {dtype}" 
                                   for col, dtype in columns.items()])
            
            with self.engine.begin() as conn:
                conn.execute(text(f"CREATE TABLE IF NOT EXISTS {safe_table_name} ({column_defs});"))
            return True
        except Exception as e:
            st.error(f"❌ Error creating table: {e}")
            return False
            
    def insert_data(self, table_name: str, data: Dict) -> bool:
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join([f":{key}" for key in data.keys()])
            sql = text(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})")
            with self.engine.begin() as conn:
                conn.execute(sql, data)
            return True
        except Exception as e:
            st.error(f"❌ Insert failed: {e}")
            return False
        
    def table_exists(self, table_name: str) -> bool:
        return inspect(self.engine).has_table(table_name)

# Helper Functions
def validate_table_name(name: str) -> bool:
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))

def detect_file_encoding(uploaded_file) -> str:
    rawdata = uploaded_file.read()
    result = chardet.detect(rawdata)
    uploaded_file.seek(0)  # Reset file pointer
    return result['encoding']

import pandas as pd
import datetime
from typing import Union
import streamlit as st

def save_metadata(df: pd.DataFrame, table_name: str, uploaded_file: Union[st.runtime.uploaded_file_manager.UploadedFile, None], 
                  engine, metadata_table: str = "metadata_store") -> None:
    try:
        created_at = datetime.datetime.now()
        file_name = uploaded_file.name if uploaded_file else None
        file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 2) if uploaded_file else None

        metadata = pd.DataFrame([
            {
                "table_name": table_name,
                "column_name": col,
                "data_type": str(dtype),
                "num_records": len(df),
                "source_file": file_name,
                "file_size_kb": file_size_kb,
                "created_at": created_at
            }
            for col, dtype in df.dtypes.items()
        ])

        metadata.to_sql(metadata_table, engine, if_exists="append", index=False)
        
    except Exception as e:
        st.error(f"❌ Failed to save metadata: {e}")

def show_metadata(engine, table_name: str, metadata_table: str = "metadata_store"):
    import pandas as pd

    try:
        query = f"""
            SELECT column_name, data_type, num_records, source_file, file_size_kb, created_at
            FROM {metadata_table}
            WHERE table_name = table_name
            ORDER BY column_name
        """
        df_meta = pd.read_sql_query(query, engine, params={"table_name": table_name})
        st.subheader(f"📊 Metadata for '{table_name}'")
        st.dataframe(df_meta)

    except Exception as e:
        st.error(f"❌ Failed to show metadata: {e}")

# UI Components
def render_csv_importer(db: DatabaseManager):
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    table_name = st.text_input("Target table name")
    
    if uploaded_file and table_name:
        if not validate_table_name(table_name):
            st.error("Invalid table name format")
            return
            
        try:
            df = pd.read_csv(uploaded_file)
            safe_table_name = quoted_name(table_name, quote=True)
            
            with st.spinner(f"Uploading {len(df)} records..."):
                df.to_sql(safe_table_name, db.engine, if_exists='replace', index=False)
                # Save DataFrame to CSV for profiling use
                df.to_csv("source_data.csv", index=False)
            st.success(f"✅ Table '{table_name}' created successfully with {len(df)} rows.")
            st.dataframe(df.head())

            # Metadata
            save_metadata(df, table_name, uploaded_file, db.engine)
            show_metadata(db.engine, table_name)
        except Exception as e:
            st.error(f"❌ Error: {e}")
        

def render_txt_importer(db: DatabaseManager):
    st.subheader("Upload TXT File")
    uploaded_file = st.file_uploader("Choose a TXT file", type=["txt"])
    
    delimiter_map = {
        "Comma": ",",
        "Pipe": "|",
        "Tab": "\t",
        "Semicolon": ";"
    }
    
    delimiter_option = st.selectbox("Choose a delimiter", list(delimiter_map.keys()) + ["Other"])
    
    delimiter = st.text_input("Enter your custom delimiter") if delimiter_option == "Other" else delimiter_map.get(delimiter_option, ",")
    table_name = st.text_input("Target table name")
    
    if uploaded_file and delimiter and table_name:
        if not validate_table_name(table_name):
            st.error("Invalid table name format")
            return
            
        try:
            encoding = detect_file_encoding(uploaded_file)
            content = uploaded_file.getvalue().decode(encoding)
            
            with st.spinner("Processing file..."):
                df = pd.read_csv(io.StringIO(content), delimiter=delimiter)
                df.to_sql(quoted_name(table_name, quote=True), db.engine, if_exists='replace', index=False)
                df.to_csv("source_data.csv", index=False)
            st.success(f"✅ TXT imported as table '{table_name}' with {len(df)} records.")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"❌ Error: {e}")

def render_sql_query(db: DatabaseManager):
    st.subheader("Execute SQL Query")
    query = st.text_area("Enter your SQL query here")
    
    if st.button("Run Query") and query.strip():
        with st.spinner("Executing query..."):
            result = db.execute_query(query)
            if result is not None:
                result.to_csv("source_data.csv", index=False)
                st.success("✅ Query executed successfully")
                st.dataframe(result)

def render_manual_table(db):
    st.header("Create a Table Manually")

    # Table name input
    table_name = st.text_input("Enter table name")

    # Number of columns
    num_columns = st.number_input("Enter number of columns", min_value=1, step=1)

    column_names = []
    column_types = []

    st.subheader("Define columns")
    for i in range(num_columns):
        col1, col2 = st.columns(2)
        with col1:
            col_name = st.text_input(f"Column {i+1} name", key=f"col_name_{i}")
            column_names.append(col_name)
        with col2:
            col_type = st.selectbox(f"Column {i+1} type", options=["INTEGER", "TEXT", "REAL", "BOOLEAN", "DATE", "TIMESTAMP"], key=f"col_type_{i}")
            column_types.append(col_type)

    column_defs = dict(zip(column_names, column_types))

    if st.button("Create Table") and table_name and all(column_names):
        if not validate_table_name(table_name):
            st.error("❌ Invalid table name format")
            return

        if db.create_table(table_name, column_defs):
            st.success(f"✅ Table '{table_name}' created successfully.")
            # Show empty DataFrame and save to CSV
            df = pd.DataFrame(columns=column_names)
            st.dataframe(df.head())
            df.to_csv("source_data.csv", index=False)
            # Save state so insert form is shown after rerun
            st.session_state["table_created"] = True
            st.session_state["created_table_name"] = table_name
            st.session_state["created_columns"] = column_names
            st.session_state["created_columns_type"] = column_types
        else:
            st.error("❌ Failed to create table.")

    # Keep insert form visible after rerun
    if st.session_state.get("table_created"):
        render_data_insertion(
            db,
            st.session_state["created_table_name"],
            st.session_state["created_columns"],
            st.session_state["created_columns_type"]
        )

        if st.button("Reset Table Creation"):
            st.session_state["table_created"] = False
            st.session_state["created_table_name"] = ""
            st.session_state["created_columns"] = []
            st.session_state["created_columns_type"] = []

def render_data_insertion(db: DatabaseManager, table_name: str, column_names : List, column_types : List):
    st.subheader(f"Insert data into '{table_name}'")
    data = {}
    
    column_defs = dict(zip(column_names, column_types))
    for column in column_defs:
        col_type = column_defs[column]
        user_input = st.text_input(f"Enter value for '{column}' ({col_type})", key=f"insert_{column}")

        if user_input == "":
            data[column] = None
        else:
            try:
                if col_type == "INTEGER":
                    data[column] = int(user_input)
                elif col_type == "REAL":
                    data[column] = float(user_input)
                elif col_type == "BOOLEAN":
                    data[column] = user_input.lower() in ["true", "1", "yes"]
                elif col_type == "DATE":
                    data[column] = datetime.strptime(user_input, "%Y-%m-%d").date()
                elif col_type == "TIMESTAMP":
                    data[column] = datetime.strptime(user_input, "%Y-%m-%d %H:%M:%S")
                else:  # TEXT, VARCHAR...
                    data[column] = user_input
            except ValueError:
                st.error(f"❌ Invalid value for '{column}' with type '{col_type}'")
                return

    if st.button("Insert Data"):
        if db.insert_data(table_name, data):
            st.success("✅ Data inserted successfully.")
        else:
            st.error("❌ Failed to insert data.")
# Main App

def init_session_state():
    defaults = {
        "created_column_type": [],
        "created_table_name": "",
        "created_columns": [],
        "table_created": False
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def importer():
    init_session_state()
    st.title("Advanced Data Import Module")
    db = DatabaseManager()
    
    import_method = st.radio(
        "Select import method:",
        ["From CSV", "From TXT", "Manual Table", "SQL Query"],
        horizontal=True
    )
    
    if import_method == "From CSV":
        render_csv_importer(db)
    elif import_method == "From TXT":
        render_txt_importer(db)
    elif import_method == "Manual Table":
        render_manual_table(db)
    elif import_method == "SQL Query":
        render_sql_query(db)
    
    # Add database inspector
    if st.checkbox("Show available tables"):
        if tables := inspect(db.engine).get_table_names():
            st.write("Existing tables:", ", ".join(tables))
        else:
            st.info("No tables found in the database")

if __name__ == "__main__":
    importer()
import os
import json
import gspread
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone # Added timezone
# import psycopg2 # Not directly used if using SQLAlchemy engine
from sqlalchemy import create_engine, text as sql_text # Added text for SQLAlchemy core execution
from google.oauth2.service_account import Credentials # For gspread authentication

# --- Configuration from Environment Variables ---
DB_HOST = os.environ.get("NEON_DB_HOST")
DB_PORT = 5432
DB_NAME = os.environ.get("NEON_DB_NAME")
DB_USER = os.environ.get("NEON_DB_USER")
DB_PASSWORD = os.environ.get("NEON_DB_PASSWORD")

SPREADSHEET_NAME = os.environ.get("GSHEET_SPREADSHEET_NAME", "Unusual Options Flow Database")
GSHEET_CREDENTIALS_JSON_STR = os.environ.get("GSHEET_CREDENTIALS_JSON")

# --- Global Variables (Initialized in main_automated_ingestion) ---
db_engine = None

# --- Helper Functions (parse_human_readable_number, clean_strike_price_string) ---
def is_valid_date_format(date_string, date_format="%Y-%m-%d"):
    """Checks if a string matches the specified date format."""
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False
# Ensure these are defined as you had them:
def parse_human_readable_number(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    s_original_for_debug = str(value)
    s = str(value).strip().upper().replace('$', '').replace(',', '')
    if not s: return np.nan
    numeric_str_part = s
    multiplier = 1.0
    if s.endswith('K'): multiplier, numeric_str_part = 1_000.0, s[:-1].strip()
    elif s.endswith('M'): multiplier, numeric_str_part = 1_000_000.0, s[:-1].strip()
    elif s.endswith('B'): multiplier, numeric_str_part = 1_000_000_000.0, s[:-1].strip()
    if not numeric_str_part and (s.endswith('K') or s.endswith('M') or s.endswith('B')): return np.nan
    if not numeric_str_part: return np.nan
    try: return float(numeric_str_part) * multiplier
    except ValueError:
        print(f"DEBUG parse_human_readable_number: FAILED TO CONVERT. Original: {repr(s_original_for_debug)}, Cleaned s: {repr(s)}, Tried: {repr(numeric_str_part)}, Multiplier: {multiplier}.")
        return np.nan

def clean_strike_price_string(strike_value):
    if pd.isna(strike_value): return np.nan
    s = str(strike_value).strip()
    open_paren_index = s.find('(')
    if open_paren_index != -1: s = s[:open_paren_index].strip()
    try: return float(s)
    except ValueError:
        print(f"DEBUG clean_strike_price_string: Could not convert '{s}' (original: '{strike_value}') to float.")
        return np.nan

# --- SECTION and HEADER DEFINITIONS (as you had them) ---
SECTION_DEFINITIONS = {
    "CALLS BOUGHT": {"option_type": "CALL", "action": "BUY", "sentiment": "Bullish", "original_term": "Calls Bought"},
    "PUTS SOLD":    {"option_type": "PUT",  "action": "SELL","sentiment": "Bullish", "original_term": "Puts Sold"},
    "PUTS BOUGHT":  {"option_type": "PUT",  "action": "BUY", "sentiment": "Bearish", "original_term": "Puts Bought"},
    "CALLS SOLD":   {"option_type": "CALL", "action": "SELL","sentiment": "Bearish", "original_term": "Calls Sold"},
    "CALL BOUGHT":  {"option_type": "CALL", "action": "BUY", "sentiment": "Bullish", "original_term": "Call Bought"},
    "PUT SOLD":     {"option_type": "PUT",  "action": "SELL","sentiment": "Bullish", "original_term": "Put Sold"},
    "PUT BOUGHT":   {"option_type": "PUT",  "action": "BUY", "sentiment": "Bearish", "original_term": "Put Bought"},
    "CALL SOLD":    {"option_type": "CALL", "action": "SELL","sentiment": "Bearish", "original_term": "Call Sold"}
}
EXPECTED_DATA_HEADERS = ['TICKER', 'STRIKE', 'EXP', 'PREMIUM']

# --- Database Interaction Functions (modified to take engine) ---
def create_options_activity_table(engine): # Takes SQLAlchemy engine
    """Creates the main data table if it doesn't exist."""
    try:
        with engine.connect() as connection:
            connection.execute(sql_text("""
            CREATE TABLE IF NOT EXISTS options_activity (
                id SERIAL PRIMARY KEY,
                data_date DATE NOT NULL,
                underlying_ticker TEXT NOT NULL,
                strike_price REAL,
                expiration_date DATE,
                premium_usd REAL,
                option_action TEXT,
                option_type TEXT,
                sentiment TEXT,
                UNIQUE (data_date, underlying_ticker, strike_price, expiration_date, option_action, option_type, sentiment, premium_usd) -- Made more robust
            );
            """)) # Using option_type and sentiment in UNIQUE constraint
            connection.commit()
        print("Table 'options_activity' checked/created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")
        raise # Re-raise to fail the Action if critical

def delete_data_for_date(engine, target_date_str): # Takes SQLAlchemy engine
    """Deletes existing data for a specific date."""
    try:
        with engine.connect() as connection:
            connection.execute(sql_text("DELETE FROM options_activity WHERE data_date = :date"), {"date": target_date_str})
            connection.commit()
        print(f"Deleted existing data for date: {target_date_str} (if any).")
    except Exception as e:
        print(f"Error deleting data for date {target_date_str}: {e}")
        # Decide if this is fatal. If table didn't exist, it's fine.
        if "does not exist" not in str(e).lower(): # If error is not "table does not exist"
             raise # Re-raise other errors

# --- Google Sheet Parsing Functions (as you had them, but get_data_from_worksheet takes gspread client) ---
def find_data_table_start(all_rows, search_start_index, expected_headers_list_upper):
    # ... (Your existing find_data_table_start logic) ...
    for i in range(search_start_index, len(all_rows)):
        row = all_rows[i]
        if not row or not any(str(cell).strip() for cell in row): continue
        potential_headers_upper = [str(cell).strip().upper() for cell in row]
        if potential_headers_upper[:len(expected_headers_list_upper)] == expected_headers_list_upper:
            return i, all_rows[i][:len(expected_headers_list_upper)]
    return None, None

def get_data_from_worksheet_automated(gc_authorized_client, target_spreadsheet_name, worksheet_title_to_fetch):
    """
    Fetches and parses data from a specific worksheet.
    This function MUST contain your full, working section-parsing logic.
    """
    print(f"Attempting to open spreadsheet: '{target_spreadsheet_name}' and worksheet: '{worksheet_title_to_fetch}'")
    try:
        spreadsheet = gc_authorized_client.open(target_spreadsheet_name)
        worksheet = spreadsheet.worksheet(worksheet_title_to_fetch)
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{worksheet_title_to_fetch}' not found in spreadsheet '{target_spreadsheet_name}'. Skipping.")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        print(f"Error accessing Google Sheet '{target_spreadsheet_name}' or Worksheet '{worksheet_title_to_fetch}': {e}")
        raise # Re-raise critical errors

    print(f"Processing worksheet: '{worksheet.title}' with section-based logic...")
    all_values = worksheet.get_all_values()
    if not all_values:
        print(f"Worksheet '{worksheet.title}' is empty.")
        return pd.DataFrame()

    all_section_dfs = []
    rename_map = {'Ticker': 'UNDERLYING_TICKER', 'Strike': 'STRIKE_PRICE', 'Exp': 'EXPIRATION_DATE', 'Premium': 'PREMIUM_USD'}
    current_row_index = 0
    while current_row_index < len(all_values):
        row_values = all_values[current_row_index]
        if not row_values or not str(row_values[0]).strip():
            current_row_index += 1
            continue
        current_section_indicator_raw = str(row_values[0]).strip()
        current_section_indicator_upper = current_section_indicator_raw.upper()
        matched_section_key, section_properties = None, None
        if current_section_indicator_upper in SECTION_DEFINITIONS:
            matched_section_key = current_section_indicator_upper
            section_properties = SECTION_DEFINITIONS[matched_section_key]
        else:
            for def_key, props in SECTION_DEFINITIONS.items():
                if def_key in current_section_indicator_upper:
                    matched_section_key, section_properties = def_key, props
                    break
        if section_properties:
            print(f"  Found section indicator: '{current_section_indicator_raw}' (Interpreted as: '{matched_section_key}') at sheet row {current_row_index + 1}.")
            data_header_sheet_idx, actual_data_headers_from_sheet = find_data_table_start(all_values, current_row_index + 1, EXPECTED_DATA_HEADERS)
            if actual_data_headers_from_sheet:
                cleaned_actual_data_headers = [str(h).strip() for h in actual_data_headers_from_sheet]
                print(f"    Located data table headers: {cleaned_actual_data_headers} at sheet row {data_header_sheet_idx + 1}.")
                section_data_rows_content = []
                data_body_start_sheet_idx = data_header_sheet_idx + 1
                temp_loop_idx = data_body_start_sheet_idx
                while temp_loop_idx < len(all_values):
                    data_row_candidate = all_values[temp_loop_idx]
                    if not data_row_candidate or not str(data_row_candidate[0]).strip(): break
                    first_cell_upper_candidate = str(data_row_candidate[0]).strip().upper()
                    is_another_section = False
                    if first_cell_upper_candidate in SECTION_DEFINITIONS: is_another_section = True
                    else:
                        for def_key in SECTION_DEFINITIONS.keys():
                            if def_key in first_cell_upper_candidate: is_another_section = True; break
                    if is_another_section: break
                    section_data_rows_content.append(data_row_candidate[:len(cleaned_actual_data_headers)])
                    temp_loop_idx += 1
                current_row_index = temp_loop_idx
                if section_data_rows_content:
                    section_df = pd.DataFrame(section_data_rows_content, columns=cleaned_actual_data_headers)
                    strike_col_name_in_df = next((col for col in cleaned_actual_data_headers if col.upper() == 'STRIKE'), None)
                    if strike_col_name_in_df and strike_col_name_in_df in section_df.columns:
                        section_df.loc[:, strike_col_name_in_df] = section_df[strike_col_name_in_df].apply(clean_strike_price_string)
                    section_df['OPTION_ACTION'] = SECTION_DEFINITIONS[matched_section_key]['original_term']
                    section_df['OPTION_TYPE'] = section_properties['option_type']
                    section_df['SENTIMENT'] = section_properties['sentiment']
                    section_df.rename(columns=rename_map, inplace=True)
                    all_section_dfs.append(section_df)
                    print(f"    Processed {len(section_df)} data rows for section '{SECTION_DEFINITIONS[matched_section_key]['original_term']}'.")
                else: print(f"    No data rows found for section '{SECTION_DEFINITIONS[matched_section_key]['original_term']}'.")
            else:
                print(f"  Warning: Section indicator '{current_section_indicator_raw}' found, but no data table headers.")
                current_row_index += 1
        else: current_row_index += 1
    if not all_section_dfs: return pd.DataFrame()
    final_df = pd.concat(all_section_dfs, ignore_index=True)
    print(f"Finished parsing worksheet '{worksheet.title}'. Combined DataFrame has {len(final_df)} rows.")
    return final_df


# --- Main Ingestion Logic for Automation ---
def main_automated_ingestion():
    global db_engine
    print(f"options_analyzer.py script started at {datetime.now(timezone.utc)}")
    print("Starting automated data ingestion with new sync logic...")

    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, GSHEET_CREDENTIALS_JSON_STR, SPREADSHEET_NAME]):
        print("ERROR: Critical environment variables (DB details, GSheet creds, Spreadsheet name) are missing. Exiting.")
        return

    try:
        db_engine_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        db_engine = create_engine(db_engine_url)
        print("SQLAlchemy engine created successfully.")
        create_options_activity_table(db_engine) # Renamed for consistency
    except Exception as e:
        print(f"Failed to create SQLAlchemy engine or initial table: {e}")
        return

    # Authenticate gspread
    try:
        gsheet_creds_dict_parsed = json.loads(GSHEET_CREDENTIALS_JSON_STR)
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        gspread_credentials = Credentials.from_service_account_info(gsheet_creds_dict_parsed, scopes=scopes)
        gc = gspread.authorize(gspread_credentials)
        print("Google Sheets API authorized successfully.")
        spreadsheet = gc.open(SPREADSHEET_NAME)
        print(f"Successfully opened spreadsheet: '{SPREADSHEET_NAME}'")
    except Exception as e:
        print(f"Failed to authorize Google Sheets API or open spreadsheet: {e}")
        return

    # 1. Get all worksheet titles from Google Sheets that look like dates
    gsheet_tab_dates_to_check = []
    try:
        worksheets = spreadsheet.worksheets()
        for ws in worksheets:
            if is_valid_date_format(ws.title): # Assuming YYYY-MM-DD format for titles
                gsheet_tab_dates_to_check.append(ws.title)
        print(f"Found {len(gsheet_tab_dates_to_check)} potential date tabs in Google Sheet: {gsheet_tab_dates_to_check}")
    except Exception as e:
        print(f"Error fetching worksheet list from Google Sheets: {e}")
        return
        
    if not gsheet_tab_dates_to_check:
        print("No valid date-formatted worksheet tabs found in Google Sheet. Nothing to process.")
        print(f"options_analyzer.py script finished at {datetime.now(timezone.utc)}")
        return

    # 2. Get distinct data_dates already in the PostgreSQL database
    try:
        with db_engine.connect() as connection:
            result = connection.execute(sql_text("SELECT DISTINCT data_date FROM options_activity;"))
            # Convert date objects from DB to strings in 'YYYY-MM-DD' format
            dates_in_db_raw = [row[0] for row in result]
            dates_in_db = set()
            for d_raw in dates_in_db_raw:
                if isinstance(d_raw, (datetime, pd.Timestamp)):
                    dates_in_db.add(d_raw.strftime('%Y-%m-%d'))
                elif isinstance(d_raw, str): # If already string
                    if is_valid_date_format(d_raw): # Validate it
                        dates_in_db.add(d_raw)
                # Add other type checks if necessary (e.g. datetime.date)
            
        print(f"Found {len(dates_in_db)} distinct dates in the database: {sorted(list(dates_in_db))[:10]}...") # Print first 10
    except Exception as e:
        print(f"Error fetching distinct dates from database: {e}")
        return

    # 3. Determine dates to process (in GSheet tabs but not yet fully in DB, or for re-processing)
    # For now, let's process any date in GSheet. The delete_data_for_date will handle idempotency.
    # A more advanced logic could be: dates_to_process = [d for d in gsheet_tab_dates_to_check if d not in dates_in_db]
    # But if a sheet is updated, you'd want to re-process it. So, processing all found GSheet tabs is safer with delete-first.
    dates_to_process = sorted([d for d in gsheet_tab_dates_to_check if d not in dates_in_db], reverse=True)
    
    if not dates_to_process:
        print("No new dates found in Google Sheets to process.")
    else:
        print(f"Identified {len(dates_to_process)} GSheet tabs to check/process: {dates_to_process}")

    processed_count = 0
    for target_date_str_for_sheet in dates_to_process:
        print(f"\n--- Processing sheet tab: {target_date_str_for_sheet} ---")
        
        # Get data from the specific sheet tab using your existing detailed parsing function
        options_df_from_sheet = get_data_from_worksheet_automated(gc, SPREADSHEET_NAME, target_date_str_for_sheet)

        if options_df_from_sheet is None or options_df_from_sheet.empty:
            print(f"No data loaded from worksheet '{target_date_str_for_sheet}'. Skipping ingestion for this date.")
            continue
        
        ingest_df = options_df_from_sheet.copy()
        
        # Standardize data_date format for DB
        data_date_for_db = pd.to_datetime(target_date_str_for_sheet).strftime('%Y-%m-%d')
        ingest_df['data_date'] = data_date_for_db

        if 'PREMIUM_USD' in ingest_df.columns:
            ingest_df['premium_usd_numeric'] = ingest_df['PREMIUM_USD'].apply(parse_human_readable_number)
        else:
            print(f"ERROR: 'PREMIUM_USD' column not found for sheet {target_date_str_for_sheet}. Adding NaN column.")
            ingest_df['premium_usd_numeric'] = np.nan

        if 'EXPIRATION_DATE' in ingest_df.columns:
            ingest_df['EXPIRATION_DATE'] = pd.to_datetime(ingest_df['EXPIRATION_DATE'], format='%m/%d/%y', errors='coerce')
        
        db_columns_map = {
            'data_date': 'data_date', 'UNDERLYING_TICKER': 'underlying_ticker',
            'STRIKE_PRICE': 'strike_price', 'EXPIRATION_DATE': 'expiration_date',
            'premium_usd_numeric': 'premium_usd', 'OPTION_ACTION': 'option_action',
            'OPTION_TYPE': 'option_type', 'SENTIMENT': 'sentiment'
        }
        
        final_ingest_df = pd.DataFrame()
        for df_col, db_col_name in db_columns_map.items():
            if df_col in ingest_df.columns:
                final_ingest_df[db_col_name] = ingest_df[df_col]
            else:
                # Only print warning for truly missing source columns expected by the map
                if df_col not in ['data_date', 'premium_usd_numeric']: # These are created, others from sheet
                     print(f"Warning: Source column '{df_col}' for DB column '{db_col_name}' not found in sheet '{target_date_str_for_sheet}'.")
                final_ingest_df[db_col_name] = np.nan 
        
        # Ensure date columns are ready for PostgreSQL DATE type
        if 'data_date' in final_ingest_df.columns:
            final_ingest_df['data_date'] = pd.to_datetime(final_ingest_df['data_date'], errors='coerce').dt.date
        if 'expiration_date' in final_ingest_df.columns:
            final_ingest_df['expiration_date'] = pd.to_datetime(final_ingest_df['expiration_date'], errors='coerce').dt.date
        
        # Ensure numeric types for REAL columns
        for col_to_num in ['strike_price', 'premium_usd']:
            if col_to_num in final_ingest_df.columns:
                final_ingest_df[col_to_num] = pd.to_numeric(final_ingest_df[col_to_num], errors='coerce')

        if not final_ingest_df.empty:
            expected_db_cols = list(db_columns_map.values())
            missing_cols = [col for col in expected_db_cols if col not in final_ingest_df.columns]
            if missing_cols:
                print(f"ERROR: DataFrame for {target_date_str_for_sheet} is missing required columns for DB: {missing_cols}. Skipping ingestion.")
                continue

            delete_data_for_date(db_engine, data_date_for_db) # Delete then insert for idempotency
            try:
                final_ingest_df.to_sql('options_activity', db_engine, if_exists='append', index=False, chunksize=1000)
                print(f"Successfully ingested {len(final_ingest_df)} rows for date {data_date_for_db} into PostgreSQL.")
                processed_count += 1
            except Exception as e:
                print(f"Error ingesting data into PostgreSQL for date {data_date_for_db}: {e}")
                print(f"  Attempted to ingest columns: {final_ingest_df.columns.tolist()}")
                print(f"  Data types of ingest DF: \n{final_ingest_df.dtypes}")
        else:
            print(f"No data in final_ingest_df for {target_date_str_for_sheet} to load into database.")
            
    print(f"\nProcessed and attempted ingestion for {processed_count} date(s).")
    print(f"options_analyzer.py script finished at {datetime.now(timezone.utc)}")

if __name__ == '__main__':
    main_automated_ingestion()

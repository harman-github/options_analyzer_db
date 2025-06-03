import os
import json
import gspread
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone # Added timezone
# import psycopg2 # Not directly used if using SQLAlchemy engine
from sqlalchemy import create_engine, text as sql_text # Added text for SQLAlchemy core execution
from google.oauth2.service_account import Credentials # For gspread authentication
import time
from tiingo import TiingoClient

# --- Configuration from Environment Variables ---
DB_HOST = os.environ.get("NEON_DB_HOST")
DB_PORT = 5432
DB_NAME = os.environ.get("NEON_DB_NAME")
DB_USER = os.environ.get("NEON_DB_USER")
DB_PASSWORD = os.environ.get("NEON_DB_PASSWORD")
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY") 

SPREADSHEET_NAME = os.environ.get("GSHEET_SPREADSHEET_NAME", "Unusual Options Flow Database")
GSHEET_CREDENTIALS_JSON_STR = os.environ.get("GSHEET_CREDENTIALS_JSON")

# --- Global Variables (Initialized in main_automated_ingestion) ---
db_engine = None
tiingo_client = None

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
                market_cap_ingested BIGINT,
                price_on_data_date REAL,        
                UNIQUE (data_date, underlying_ticker, strike_price, expiration_date, option_action, option_type, sentiment, premium_usd) -- Made more robust
            );
            """))
            print("Table 'options_activity' checked/created successfully.")

            connection.execute(sql_text("""
            CREATE TABLE IF NOT EXISTS ticker_market_caps (
                ticker TEXT PRIMARY KEY,
                market_cap BIGINT,
                last_updated TIMESTAMP WITHOUT TIME ZONE
            );
            """))
            print("Table 'ticker_market_caps' checked/created successfully.")
            connection.commit()
    except Exception as e:
        print(f"Error creating supporting tables: {e}")
        raise

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

def get_data_from_worksheet_automated(spreadsheet_object, worksheet_title_to_fetch): # Changed parameters
    """
    Fetches and parses data from a specific worksheet object within an already opened spreadsheet.
    This function MUST contain your full, working section-parsing logic.
    """
    print(f"Accessing worksheet: '{worksheet_title_to_fetch}' from pre-opened spreadsheet object.")
    try:
        # Get the specific worksheet (tab) from the spreadsheet object
        worksheet = spreadsheet_object.worksheet(worksheet_title_to_fetch)
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{worksheet_title_to_fetch}' not found. Skipping.")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        print(f"Error accessing Worksheet '{worksheet_title_to_fetch}': {e}")
        # Consider if you want to raise this to stop the whole script or just skip this sheet
        # For now, returning empty DataFrame to allow script to try other sheets
        return pd.DataFrame() 

    print(f"Processing worksheet: '{worksheet.title}' with section-based logic...")
    all_values = worksheet.get_all_values() # This is an API call to read data
    if not all_values:
        print(f"Worksheet '{worksheet.title}' is empty or returned no values.")
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
    
def fetch_daily_data_from_tiingo(ticker_symbol, specific_date_obj, client):
    """
    Fetches EOD price and daily market cap for a ticker on a specific date from Tiingo.
    Returns (price_on_date, market_cap_on_date)
    """
    price_on_date, market_cap_on_date = None, None
    date_str = specific_date_obj.strftime('%Y-%m-%d')
    print(f"Tiingo: Fetching EOD & daily fundamentals for {ticker_symbol} on {date_str}...")
    try:
        # Get EOD price
        eod_prices = client.get_ticker_price(ticker_symbol, fmt='json', startDate=date_str, endDate=date_str)
        if eod_prices and isinstance(eod_prices, list) and len(eod_prices) > 0:
            price_on_date = eod_prices[0].get('adjClose', eod_prices[0].get('close'))

        # Get daily fundamentals (includes marketCap for that day)
        daily_fundamentals = client.get_fundamentals_daily(ticker_symbol, startDate=date_str, endDate=date_str, fmt='json')
        if daily_fundamentals and isinstance(daily_fundamentals, list) and len(daily_fundamentals) > 0:
            market_cap_on_date = daily_fundamentals[0].get('marketCap')
            if market_cap_on_date is not None:
                market_cap_on_date = int(market_cap_on_date)

        print(f"Tiingo Result for {ticker_symbol} on {date_str}: Price={price_on_date}, MarketCap={market_cap_on_date}")
        return price_on_date, market_cap_on_date
    except Exception as e:
        print(f"Tiingo: Error fetching daily data for {ticker_symbol} on {date_str}: {e}")
        return None, None

def get_and_update_general_market_cap(engine, ticker_symbol, client, force_refresh_days=7):
    """
    Gets market cap for a ticker for general "current" display purposes.
    Uses DB cache with weekly refresh. Fetches latest available from Tiingo if needed.
    """
    db_mcap, last_updated = None, None
    try:
        with engine.connect() as connection:
            result = connection.execute(
                sql_text("SELECT market_cap, last_updated FROM ticker_market_caps WHERE ticker = :ticker"),
                {"ticker": ticker_symbol}
            ).fetchone()
            if result: db_mcap, last_updated = result
    except Exception as e: print(f"DB Cache: Error reading market cap for {ticker_symbol}: {e}")

    if db_mcap and last_updated and \
       (datetime.now(timezone.utc).date() - pd.to_datetime(last_updated).date()) < timedelta(days=force_refresh_days):
        print(f"DB Cache: Using general market cap for {ticker_symbol} (updated {last_updated.strftime('%Y-%m-%d')}).")
        return int(db_mcap)

    print(f"DB Cache: General market cap for {ticker_symbol} outdated or not found. Fetching latest from Tiingo.")
    # Fetch a recent market cap from Tiingo (e.g., for yesterday, as daily fundamentals are EOD)
    # For market cap, daily fundamental for a recent date is good.
    recent_date_obj = datetime.now(timezone.utc).date() - timedelta(days=1) # Use yesterday's EOD fundamentals
    _, latest_mcap = fetch_daily_data_from_tiingo(ticker_symbol, recent_date_obj, client) # Price not needed here

    if latest_mcap is not None:
        try:
            with engine.connect() as connection:
                connection.execute(sql_text("""
                    INSERT INTO ticker_market_caps (ticker, market_cap, last_updated)
                    VALUES (:ticker, :market_cap, :last_updated)
                    ON CONFLICT (ticker) DO UPDATE SET
                        market_cap = EXCLUDED.market_cap, last_updated = EXCLUDED.last_updated;
                """), {"ticker": ticker_symbol, "market_cap": latest_mcap, "last_updated": datetime.now(timezone.utc)})
                connection.commit()
            print(f"DB Cache: Updated general market cap for {ticker_symbol} to {latest_mcap}.")
            return int(latest_mcap)
        except Exception as e: print(f"DB Cache: Error updating market cap for {ticker_symbol}: {e}")

    return db_mcap # Return old one if new fetch failed, or None

# --- Main Ingestion Logic for Automation ---
def main_automated_ingestion():
    global db_engine, tiingo_client
    print(f"options_analyzer.py script started at {datetime.now(timezone.utc)}")
    print("Starting automated data ingestion with new sync logic...")

    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, GSHEET_CREDENTIALS_JSON_STR, SPREADSHEET_NAME, TIINGO_API_KEY]):
        print("ERROR: Critical environment variables are missing. Exiting.")
        return

    try:
        db_engine_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        db_engine = create_engine(db_engine_url)
        print("SQLAlchemy engine created successfully.")
        create_options_activity_table(db_engine) 
    except Exception as e:
        print(f"Failed to create SQLAlchemy engine or initial table: {e}")
        return
    
    try:
        tiingo_config = {'api_key': TIINGO_API_KEY, 'session': True}
        tiingo_client = TiingoClient(tiingo_config)
        print("Tiingo client initialized.")
    except Exception as e: print(f"Failed to initialize Tiingo client: {e}"); return

    # Authenticate gspread AND OPEN SPREADSHEET ONCE
    try:
        gsheet_creds_dict_parsed = json.loads(GSHEET_CREDENTIALS_JSON_STR)
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        gspread_credentials = Credentials.from_service_account_info(gsheet_creds_dict_parsed, scopes=scopes)
        gc = gspread.authorize(gspread_credentials)
        print("Google Sheets API authorized successfully.")
        
        # --- OPEN THE SPREADSHEET HERE, ONCE ---
        spreadsheet = gc.open(SPREADSHEET_NAME)
        print(f"Successfully opened spreadsheet: '{SPREADSHEET_NAME}' by name.")
    except Exception as e:
        print(f"Failed to authorize Google Sheets API or open spreadsheet '{SPREADSHEET_NAME}': {e}")
        return

    # 1. Get all worksheet titles from the already opened spreadsheet
    gsheet_tab_dates_to_check = []
    try:
        worksheets_list = spreadsheet.worksheets() # API call
        for ws in worksheets_list:
            if is_valid_date_format(ws.title):
                gsheet_tab_dates_to_check.append(ws.title)
        print(f"Found {len(gsheet_tab_dates_to_check)} potential date tabs in Google Sheet: {gsheet_tab_dates_to_check}")
    except Exception as e:
        print(f"Error fetching worksheet list from Google Sheets: {e}")
        return
        
    if not gsheet_tab_dates_to_check:
        print("No valid date-formatted worksheet tabs found in Google Sheet. Nothing to process.")
        print(f"options_analyzer.py script finished at {datetime.now(timezone.utc)}")
        return

    # 2. Get distinct data_dates already in the PostgreSQL database (logic remains the same)
    
    dates_in_db = set()
    try:
        with db_engine.connect() as connection:
            # Ensure you have: from sqlalchemy import text as sql_text
            result = connection.execute(sql_text("SELECT DISTINCT data_date FROM options_activity;"))
            dates_in_db_raw = [row[0] for row in result]
            dates_in_db = set() # Initialize an empty set

            for d_raw in dates_in_db_raw:
                if d_raw is not None: # Important: Ensure the value from DB isn't NULL
                    try:
                        # pd.to_datetime can handle datetime.date, datetime.datetime, pd.Timestamp,
                        # and even well-formatted date strings (though DB should return date/time objects).
                        date_str = pd.to_datetime(d_raw).strftime('%Y-%m-%d')
                        dates_in_db.add(date_str)
                    except Exception as e:
                        # This might happen if d_raw is an unexpected type or unparseable
                        print(f"Warning: Could not convert database date value '{repr(d_raw)}' to YYYY-MM-DD string: {e}")
            
        print(f"Found {len(dates_in_db)} distinct dates in the database: {sorted(list(dates_in_db))[:10]}...") # Print first 10
    except Exception as e:
        print(f"Error fetching distinct dates from database: {e}")
        # You might want to return or exit here if this step is critical and fails
        print(f"options_analyzer.py script finished with error at {datetime.now(timezone.utc)}")
        return # Stop further execution if DB dates can't be fetched


    # 3. Determine dates to process
    # Process all valid date tabs found in GSheet. delete_data_for_date handles idempotency.
    dates_to_process = sorted(gsheet_tab_dates_to_check, reverse=True) 
    
    if not dates_to_process:
        print("No new or all dates from Google Sheets to process based on current logic.")
    else:
        print(f"Identified {len(dates_to_process)} GSheet tabs to check/process: {dates_to_process}")

    processed_count = 0
    for target_date_str_for_sheet in dates_to_process:
        print(f"\n--- Processing sheet tab: {target_date_str_for_sheet} ---")
        
        # --- PASS THE OPENED SPREADSHEET OBJECT ---
        options_df_from_sheet = get_data_from_worksheet_automated(spreadsheet, target_date_str_for_sheet)

        if options_df_from_sheet is None or options_df_from_sheet.empty:
            print(f"No data loaded from worksheet '{target_date_str_for_sheet}'. Skipping ingestion for this date.")
            continue
        
        # ... (rest of your ingestion logic: adding data_date, parsing premium, renaming, to_sql) ...
        # Ensure this part is complete and correct as per your last working version
        ingest_df = options_df_from_sheet.copy()
        data_date_for_db = pd.to_datetime(target_date_str_for_sheet).strftime('%Y-%m-%d')
        ingest_df['data_date'] = data_date_for_db
        if 'PREMIUM_USD' in ingest_df.columns:
            ingest_df['premium_usd_numeric'] = ingest_df['PREMIUM_USD'].apply(parse_human_readable_number)
        else:
            ingest_df['premium_usd_numeric'] = np.nan
        if 'EXPIRATION_DATE' in ingest_df.columns:
            ingest_df['EXPIRATION_DATE'] = pd.to_datetime(ingest_df['EXPIRATION_DATE'], format='%m/%d/%y', errors='coerce')
        
        db_columns_map = {
            'data_date': 'data_date', 'UNDERLYING_TICKER': 'underlying_ticker',
            'STRIKE_PRICE': 'strike_price', 'EXPIRATION_DATE': 'expiration_date',
            'premium_usd_numeric': 'premium_usd', 'OPTION_ACTION': 'option_action',
            'OPTION_TYPE': 'option_type', 'SENTIMENT': 'sentiment',
            'market_cap_ingested': 'market_cap_ingested', # New
            'price_on_data_date': 'price_on_data_date'
        }
        final_ingest_df = pd.DataFrame()
        for df_col, db_col_name in db_columns_map.items():
            if df_col in ingest_df.columns: final_ingest_df[db_col_name] = ingest_df[df_col]
            else: final_ingest_df[db_col_name] = np.nan
        if 'data_date' in final_ingest_df.columns: final_ingest_df['data_date'] = pd.to_datetime(final_ingest_df['data_date'], errors='coerce').dt.date
        if 'expiration_date' in final_ingest_df.columns: final_ingest_df['expiration_date'] = pd.to_datetime(final_ingest_df['expiration_date'], errors='coerce').dt.date
        print(f"\nDataFrame prepared for ingestion (first 5 rows): \n{final_ingest_df.head()}")
        print(f"DEBUG: Columns in final_ingest_df before to_sql: {final_ingest_df.columns.tolist()}") # <<< ADD THIS
        print(f"DEBUG: Index of final_ingest_df: {final_ingest_df.index}") # Optional: check index
        print(f"Data types: \n{final_ingest_df.dtypes}")    
        if 'id' in final_ingest_df.columns:
            print("WARNING: 'id' column IS PRESENT in final_ingest_df. This is likely the cause of the error.")    
        for col_to_num in ['strike_price', 'premium_usd']:
            if col_to_num in final_ingest_df.columns: final_ingest_df[col_to_num] = pd.to_numeric(final_ingest_df[col_to_num], errors='coerce')

        if not final_ingest_df.empty:
            expected_db_cols = list(db_columns_map.values())
            missing_cols = [col for col in expected_db_cols if col not in final_ingest_df.columns]
            if missing_cols:
                print(f"ERROR: DataFrame for {target_date_str_for_sheet} missing DB columns: {missing_cols}. Skipping.")
                continue
            delete_data_for_date(db_engine, data_date_for_db)
            try:
                final_ingest_df.to_sql('options_activity', db_engine, if_exists='append', index=False, chunksize=1000)
                print(f"Successfully ingested {len(final_ingest_df)} rows for date {data_date_for_db} into PostgreSQL.")
                processed_count += 1
            except Exception as e:
                print(f"Error ingesting data into PostgreSQL for {data_date_for_db}: {e}")
        else:
            print(f"No data in final_ingest_df for {target_date_str_for_sheet} to load.")
            
    print(f"\nProcessed and attempted ingestion for {processed_count} date(s).")
    print(f"options_analyzer.py script finished at {datetime.now(timezone.utc)}")

if __name__ == '__main__':
    main_automated_ingestion()

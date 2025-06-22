import os
import json
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
from sqlalchemy import create_engine, text as sql_text
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import yfinance as yf

# --- Configuration from Environment Variables (to be set in GitHub Secrets) ---
DB_HOST = os.environ.get("NEON_DB_HOST")
DB_PORT = os.environ.get("NEON_DB_PORT", "6543") # Default to NEON pooler port
DB_NAME = os.environ.get("NEON_DB_NAME", "postgres")
DB_USER = os.environ.get("NEON_DB_USER", "postgres")
DB_PASSWORD = os.environ.get("NEON_DB_PASSWORD")
GSHEET_CREDENTIALS_JSON_STR = os.environ.get("GSHEET_CREDENTIALS_JSON")
SPREADSHEET_NAME = os.environ.get("GSHEET_SPREADSHEET_NAME", "Unusual Options Flow Database")

# --- Helper Functions (as defined before) ---
def is_valid_date_format(date_string, date_format="%Y-%m-%d"):
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False

def parse_human_readable_number(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    s = str(value).strip().upper().replace('$', '').replace(',', '')
    if not s: return np.nan
    numeric_str_part = s
    multiplier = 1.0
    if s.endswith('K'): multiplier, numeric_str_part = 1_000.0, s[:-1].strip()
    elif s.endswith('M'): multiplier, numeric_str_part = 1_000_000.0, s[:-1].strip()
    elif s.endswith('B'): multiplier, numeric_str_part = 1_000_000_000.0, s[:-1].strip()
    if not numeric_str_part: return np.nan
    try: return float(numeric_str_part) * multiplier
    except ValueError: return np.nan

def clean_strike_price_string(strike_value):
    if pd.isna(strike_value): return np.nan
    s = str(strike_value).strip()
    open_paren_index = s.find('(')
    if open_paren_index != -1: s = s[:open_paren_index].strip()
    try: return float(s)
    except ValueError: return np.nan

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

def find_data_table_start(all_rows, search_start_index, expected_headers_list_upper):
    for i in range(search_start_index, len(all_rows)):
        row = all_rows[i]
        if not row or not any(str(cell).strip() for cell in row): continue
        potential_headers_upper = [str(cell).strip().upper() for cell in row]
        if potential_headers_upper[:len(expected_headers_list_upper)] == expected_headers_list_upper:
            return i, all_rows[i][:len(expected_headers_list_upper)]
    return None, None

def get_data_from_worksheet_automated(spreadsheet_object, worksheet_title_to_fetch):
    try:
        worksheet = spreadsheet_object.worksheet(worksheet_title_to_fetch)
    except gspread.exceptions.WorksheetNotFound:
        print(f"Worksheet '{worksheet_title_to_fetch}' not found. Skipping.")
        return pd.DataFrame()
    all_values = worksheet.get_all_values()
    if not all_values: return pd.DataFrame()
    all_section_dfs = []
    rename_map = {'Ticker': 'UNDERLYING_TICKER', 'Strike': 'STRIKE_PRICE', 'Exp': 'EXPIRATION_DATE', 'Premium': 'PREMIUM_USD'}
    current_row_index = 0
    while current_row_index < len(all_values):
        row_values = all_values[current_row_index]
        if not row_values or not str(row_values[0]).strip(): current_row_index += 1; continue
        current_section_indicator_upper = str(row_values[0]).strip().upper()
        matched_section_key, section_properties = None, None
        for def_key, props in SECTION_DEFINITIONS.items():
            if def_key in current_section_indicator_upper:
                matched_section_key, section_properties = def_key, props
                break
        if section_properties:
            data_header_sheet_idx, actual_data_headers = find_data_table_start(all_values, current_row_index + 1, EXPECTED_DATA_HEADERS)
            if actual_data_headers:
                cleaned_headers = [str(h).strip() for h in actual_data_headers]
                section_data_rows = []
                temp_loop_idx = data_header_sheet_idx + 1
                while temp_loop_idx < len(all_values):
                    data_row = all_values[temp_loop_idx]
                    if not data_row or not str(data_row[0]).strip(): break
                    first_cell = str(data_row[0]).strip().upper()
                    if any(key in first_cell for key in SECTION_DEFINITIONS): break
                    section_data_rows.append(data_row[:len(cleaned_headers)])
                    temp_loop_idx += 1
                current_row_index = temp_loop_idx
                if section_data_rows:
                    section_df = pd.DataFrame(section_data_rows, columns=cleaned_headers)
                    strike_col_name = next((col for col in cleaned_headers if col.upper() == 'STRIKE'), None)
                    if strike_col_name: section_df[strike_col_name] = section_df[strike_col_name].apply(clean_strike_price_string)
                    section_df['OPTION_ACTION'] = section_properties['original_term']
                    section_df['OPTION_TYPE'] = section_properties['option_type']
                    section_df['SENTIMENT'] = section_properties['sentiment']
                    section_df.rename(columns=rename_map, inplace=True)
                    all_section_dfs.append(section_df)
            else: current_row_index += 1
        else: current_row_index += 1
    if not all_section_dfs: return pd.DataFrame()
    return pd.concat(all_section_dfs, ignore_index=True)

def create_supporting_tables(engine):
    with engine.connect() as connection:
        connection.execute(sql_text("""
        CREATE TABLE IF NOT EXISTS options_activity (
            id SERIAL PRIMARY KEY, data_date DATE NOT NULL, underlying_ticker TEXT NOT NULL,
            strike_price REAL, expiration_date DATE, premium_usd REAL, option_action TEXT, 
            option_type TEXT, sentiment TEXT, market_cap_ingested BIGINT, price_on_data_date REAL,
            UNIQUE (data_date, underlying_ticker, strike_price, expiration_date, option_action, option_type, sentiment, premium_usd)
        );
        """))
        print("Table 'options_activity' checked/created successfully.")
        connection.execute(sql_text("""
        CREATE TABLE IF NOT EXISTS ticker_market_caps ( ticker TEXT PRIMARY KEY, market_cap BIGINT, last_updated TIMESTAMP );
        """))
        print("Table 'ticker_market_caps' checked/created successfully.")
        connection.commit()

def delete_data_for_date(engine, target_date_str):
    try:
        with engine.connect() as connection:
            connection.execute(sql_text("DELETE FROM options_activity WHERE data_date = :date"), {"date": target_date_str})
            connection.commit()
            print(f"Deleted existing data for date: {target_date_str} (if any).")
    except Exception as e:
        if "does not exist" in str(e).lower(): print(f"Table does not exist yet, skipping delete for {target_date_str}.")
        else: raise e

def main_automated_ingestion():
    global db_engine
    print(f"options_analyzer.py script started at {datetime.now(timezone.utc)}")
    print("Starting automated data ingestion...")

    # --- Configuration and Connection Setup ---
    if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, GSHEET_CREDENTIALS_JSON_STR, SPREADSHEET_NAME]):
        print("ERROR: Critical environment variables are missing. Exiting.")
        return

    try:
        db_engine_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        db_engine = create_engine(db_engine_url)
        create_supporting_tables(db_engine)
    except Exception as e:
        print(f"Failed to create SQLAlchemy engine or initial tables: {e}")
        return

    try:
        gsheet_creds_dict_parsed = json.loads(GSHEET_CREDENTIALS_JSON_STR)
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        gspread_credentials = Credentials.from_service_account_info(gsheet_creds_dict_parsed, scopes=scopes)
        gc = gspread.authorize(gspread_credentials)
        spreadsheet = gc.open(SPREADSHEET_NAME)
        print("Google Sheets API authorized and spreadsheet opened successfully.")
    except Exception as e:
        print(f"Failed to authorize Google Sheets API or open spreadsheet: {e}")
        return

    # --- Data Processing Logic ---
    try:
        worksheets = spreadsheet.worksheets()
        all_gsheet_tabs = [ws.title for ws in worksheets if is_valid_date_format(ws.title)]
        if not all_gsheet_tabs:
            print("No valid date-formatted worksheet tabs found in Google Sheet.")
            return
        
        print(f"Found {len(all_gsheet_tabs)} potential date tabs in Google Sheet.")
        # Logic to decide which tabs to process (e.g., all, or only new ones)
        # For a robust "catch-up and update" a full sync is good.
        dates_to_process = sorted(all_gsheet_tabs, reverse=True)
    except Exception as e:
        print(f"Error fetching worksheet list: {e}")
        return
    
    processed_count = 0
    run_market_data_cache = {}

    for date_to_process in dates_to_process:
        print(f"\n--- Processing sheet tab: {date_to_process} ---")
        options_df = get_data_from_worksheet_automated(spreadsheet, date_to_process)
        if options_df.empty:
            print(f"No data parsed from sheet '{date_to_process}'. Skipping.")
            continue

        ingest_df = options_df.copy()
        date_obj = pd.to_datetime(date_to_process).date()
        
        # --- Fetch Market Data ---
        ingest_df['market_cap_ingested'] = np.nan
        ingest_df['price_on_data_date'] = np.nan
        unique_tickers_in_sheet = ingest_df['UNDERLYING_TICKER'].dropna().unique()

        for ticker in unique_tickers_in_sheet:
            if ticker in run_market_data_cache:
                market_data = run_market_data_cache[ticker]
            else:
                print(f"  Fetching yfinance data for {ticker}...")
                try:
                    ticker_obj = yf.Ticker(ticker)
                    mcap = ticker_obj.info.get('marketCap')
                    hist = ticker_obj.history(start=date_obj, end=(date_obj + timedelta(days=1)))
                    price = hist['Close'].iloc[0] if not hist.empty else None
                    market_data = {'mcap': mcap, 'price': price}
                    time.sleep(2) # Throttle calls
                except Exception as e:
                    print(f"  -> Error fetching yfinance data for {ticker}: {e}")
                    market_data = {'mcap': None, 'price': None}
                run_market_data_cache[ticker] = market_data
            
            ingest_df.loc[ingest_df['UNDERLYING_TICKER'] == ticker, 'market_cap_ingested'] = market_data['mcap']
            ingest_df.loc[ingest_df['UNDERLYING_TICKER'] == ticker, 'price_on_data_date'] = market_data['price']

        # --- Prepare DataFrame for Database ---
        # 1. First, parse any string values that are needed for mapping
        ingest_df['premium_usd_numeric'] = ingest_df['PREMIUM_USD'].apply(parse_human_readable_number)

        # 2. Define the map from source columns (in ingest_df) to target DB columns
        db_columns_map = {
            'data_date': 'data_date', # This column will be added next
            'UNDERLYING_TICKER': 'underlying_ticker',
            'STRIKE_PRICE': 'strike_price',
            'EXPIRATION_DATE': 'expiration_date',
            'premium_usd_numeric': 'premium_usd',
            'OPTION_ACTION': 'option_action',
            'OPTION_TYPE': 'option_type',
            'SENTIMENT': 'sentiment',
            'market_cap_ingested': 'market_cap_ingested',
            'price_on_data_date': 'price_on_data_date'
        }
        
        # 3. Create the final DataFrame with the correct column names (lowercase)
        final_ingest_df = pd.DataFrame()
        final_ingest_df['data_date'] = pd.to_datetime(date_to_process).date() # Add data_date
        for source_col, target_col in db_columns_map.items():
            if source_col != 'data_date': # data_date is already added
                if source_col in ingest_df.columns:
                    final_ingest_df[target_col] = ingest_df[source_col]
                else:
                    final_ingest_df[target_col] = np.nan # Add as NaN if a source column is missing

        # 4. NOW, perform final type conversions on the final DataFrame with correct column names
        if 'strike_price' in final_ingest_df.columns:
            final_ingest_df['strike_price'] = pd.to_numeric(final_ingest_df['strike_price'], errors='coerce')
        if 'premium_usd' in final_ingest_df.columns:
            final_ingest_df['premium_usd'] = pd.to_numeric(final_ingest_df['premium_usd'], errors='coerce')
        if 'expiration_date' in final_ingest_df.columns:
            final_ingest_df['expiration_date'] = pd.to_datetime(final_ingest_df['expiration_date'], format='%m/%d/%y', errors='coerce').dt.date
        
        final_ingest_df.dropna(subset=['data_date', 'underlying_ticker'], inplace=True)
        
        # --- Ingest into Database ---
        if not final_ingest_df.empty:
            delete_data_for_date(db_engine, date_to_process)
            try:
                final_ingest_df.to_sql('options_activity', db_engine, if_exists='append', index=False, chunksize=500)
                print(f"Successfully ingested {len(final_ingest_df)} rows for date {date_to_process} into Supabase.")
                processed_count += 1
            except Exception as e:
                print(f"Error ingesting data for {date_to_process}: {e}")
        else:
            print(f"No valid data to ingest for {date_to_process}.")
            
    print(f"\nMigration script finished. Processed and attempted ingestion for {processed_count} date(s).")

if __name__ == '__main__':
    main_automated_ingestion()

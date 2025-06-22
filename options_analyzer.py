import os
import json
import time
from datetime import datetime, timedelta, timezone
import gspread
import numpy as np
import pandas as pd
import yfinance as yf
from google.oauth2.service_account import Credentials
from sqlalchemy import create_engine, text as sql_text

# --- Configuration from Environment Variables (to be set in GitHub Secrets) ---
DB_HOST = os.environ.get("NEON_DB_HOST") # Use NEON_ or update secrets to NEON_
DB_PORT = os.environ.get("NEON_DB_PORT", "6543")
DB_NAME = os.environ.get("NEON_DB_NAME", "postgres")
DB_USER = os.environ.get("NEON_DB_USER", "postgres")
DB_PASSWORD = os.environ.get("NEON_DB_PASSWORD")
GSHEET_CREDENTIALS_JSON_STR = os.environ.get("GSHEET_CREDENTIALS_JSON")
SPREADSHEET_NAME = os.environ.get("GSHEET_SPREADSHEET_NAME", "Unusual Options Flow Database")

# --- Helper Functions ---
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
    numeric_str_part, multiplier = s, 1.0
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

def find_data_table_start(all_rows, start_index, expected_headers_list_upper):
    for i in range(start_index, len(all_rows)):
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
        );"""))
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
    if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, GSHEET_CREDENTIALS_JSON_STR, SPREADSHEET_NAME]):
        print("ERROR: Critical environment variables are missing. Exiting.")
        return

    try:
        db_engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        create_supporting_tables(db_engine)
    except Exception as e:
        print(f"FATAL: Could not connect to DB or create tables: {e}"); return

    try:
        gsheet_creds_dict = json.loads(GSHEET_CREDENTIALS_JSON_STR)
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        gspread_credentials = Credentials.from_service_account_info(gsheet_creds_dict, scopes=scopes)
        gc = gspread.authorize(gspread_credentials)
        spreadsheet = gc.open(SPREADSHEET_NAME)
        print("Google Sheets API authorized and spreadsheet opened successfully.")
    except Exception as e:
        print(f"FATAL: Could not authorize/open Google Sheets: {e}"); return
        
    all_gsheet_tabs = [ws.title for ws in spreadsheet.worksheets() if is_valid_date_format(ws.title)]
    print(f"Found {len(all_gsheet_tabs)} date-formatted tabs in Google Sheet to process.")
    if not all_gsheet_tabs: return

    run_market_data_cache = {}
    processed_count = 0
    for date_to_process in sorted(all_gsheet_tabs, reverse=True):
        print(f"\n--- Processing sheet tab: {date_to_process} ---")
        options_df = get_data_from_worksheet_automated(spreadsheet, date_to_process)
        if options_df.empty: print(f"No data parsed from sheet '{date_to_process}'."); continue
        
        # Add new columns before renaming, using original uppercase names
        options_df['data_date'] = pd.to_datetime(date_to_process).date()
        options_df['premium_usd_numeric'] = options_df['PREMIUM_USD'].apply(parse_human_readable_number)
        options_df['market_cap_ingested'] = np.nan
        options_df['price_on_data_date'] = np.nan
        
        unique_tickers = options_df['UNDERLYING_TICKER'].dropna().unique()
        for ticker in unique_tickers:
            if ticker not in run_market_data_cache:
                print(f"  Fetching yfinance data for {ticker}...")
                try:
                    ticker_obj = yf.Ticker(ticker)
                    mcap = ticker_obj.info.get('marketCap')
                    hist = ticker_obj.history(start=date_to_process, end=(pd.to_datetime(date_to_process) + timedelta(days=1)).strftime('%Y-%m-%d'))
                    price = hist['Close'].iloc[0] if not hist.empty else None
                    run_market_data_cache[ticker] = {'mcap': mcap, 'price': price}
                    print(f"  -> {ticker}: Mcap={mcap}, Price={price}")
                    time.sleep(2) # Throttle calls
                except Exception as e:
                    print(f"  -> Error fetching yfinance data for {ticker}: {e}")
                    run_market_data_cache[ticker] = {'mcap': None, 'price': None}
            
            options_df.loc[options_df['UNDERLYING_TICKER'] == ticker, 'market_cap_ingested'] = run_market_data_cache[ticker]['mcap']
            options_df.loc[options_df['UNDERLYING_TICKER'] == ticker, 'price_on_data_date'] = run_market_data_cache[ticker]['price']

        # Define the map from source DataFrame columns to final DB columns
        db_columns_map = {
            'data_date': 'data_date', 
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
        
        # Select and rename columns
        columns_to_select = [col for col in db_columns_map.keys() if col in options_df.columns]
        final_ingest_df = options_df[columns_to_select].rename(columns=db_columns_map)
        
        # Perform final type conversions on the renamed (lowercase) columns
        if 'strike_price' in final_ingest_df.columns:
            final_ingest_df['strike_price'] = pd.to_numeric(final_ingest_df['strike_price'], errors='coerce')
        if 'expiration_date' in final_ingest_df.columns:
            final_ingest_df['expiration_date'] = pd.to_datetime(final_ingest_df['expiration_date'], format='%m/%d/%y', errors='coerce').dt.date

        final_ingest_df.dropna(subset=['data_date', 'underlying_ticker'], inplace=True)
        
        if not final_ingest_df.empty:
            delete_data_for_date(db_engine, date_to_process)
            try:
                final_ingest_df.to_sql('options_activity', db_engine, if_exists='append', index=False, chunksize=500)
                print(f"Successfully ingested {len(final_ingest_df)} rows for date {date_to_process} into NEON.")
                processed_count += 1
            except Exception as e:
                print(f"Error ingesting data for {date_to_process}: {e}")
        else:
            print(f"No valid data to ingest for {date_to_process}.")
            
    print(f"\nMigration script finished. Processed and attempted ingestion for {processed_count} date(s).")

if __name__ == '__main__':
    main_automated_ingestion()

import streamlit as st
import pandas as pd
from sqlalchemy import text # Added text import
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import plotly.express as px
import matplotlib


# --- DATABASE CONNECTION DETAILS ---
# DB_HOST = "localhost"
# DB_PORT = "5432"
# DB_NAME = "options_analyzer_db"
# DB_USER = "options_app_user"
# DB_PASSWORD = "Morpheus321"

# db_connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Helper Functions ---
market_cap_cache = {}
def get_market_cap_st(ticker_symbol):
    if ticker_symbol in market_cap_cache:
        return market_cap_cache[ticker_symbol]
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        market_cap = info.get('marketCap')
        if market_cap:
            market_cap_cache[ticker_symbol] = market_cap
            return market_cap
        market_cap_cache[ticker_symbol] = None
        return None
    except Exception:
        market_cap_cache[ticker_symbol] = None
        return None

@st.cache_data(ttl=300)
def get_current_price(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        price = ticker.fast_info.get('last_price', None)
        if price:
            return price
        tod = ticker.history(period='2d')
        if not tod.empty:
            return tod['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching current price for {ticker_symbol}: {e}")
    return None

@st.cache_data(ttl=3600)
def get_db_date_range():
    min_db_date, max_db_date = None, None
    try:
        # engine = create_engine(db_connection_string)
        conn = st.connection("postgresql", type="sql") 
        min_max_df = conn.query("SELECT MIN(data_date) as min_val, MAX(data_date) as max_val FROM options_activity;", ttl=0) # ttl=0 for query if outer func is cached

        if not min_max_df.empty:
            min_val = min_max_df['min_val'].iloc[0]
            max_val = min_max_df['max_val'].iloc[0]
            if pd.notna(min_val): 
                min_db_date = pd.to_datetime(min_val).date()
            if pd.notna(max_val): 
                max_db_date = pd.to_datetime(max_val).date()
        return min_db_date, max_db_date
    except Exception as e:
        print(f"Error fetching date range from DB: {e}")
        return None, None

@st.cache_data(ttl=3600)
def fetch_options_activity_for_range(start_date, end_date):
    start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
    try:
        # engine = create_engine(db_connection_string)
        conn = st.connection("postgresql", type="sql")
        query = """
        SELECT * FROM options_activity 
        WHERE data_date >= :start_date AND data_date <= :end_date 
        ORDER BY data_date, underlying_ticker;
        """
        df = conn.query(query, params={'start_date': start_date_str, 'end_date': end_date_str}, ttl =0)
        if not df.empty:
            df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
            df['data_date'] = pd.to_datetime(df['data_date'], errors='coerce')
            df['premium_usd'] = pd.to_numeric(df['premium_usd'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Error connecting to DB or fetching data for range {start_date_str}-{end_date_str}: {e}")
        return pd.DataFrame()

def analyze_ticker_dashboard(options_df_for_period, selected_range_start_date_dt):
    if options_df_for_period.empty:
        st.info("analyze_ticker_dashboard received an empty DataFrame. No analysis to perform.") # UI feedback
        return pd.DataFrame()

    # --- Add this DEBUG block ---
    st.subheader("DEBUG: Input to analyze_ticker_dashboard") # Temporary for UI visibility
    st.write("Data types of `options_df_for_period`:", options_df_for_period.dtypes)
    st.write("First 5 rows of `options_df_for_period` (relevant columns):")
    st.dataframe(options_df_for_period[['underlying_ticker', 'option_action', 'sentiment', 'premium_usd']].head())
    if 'sentiment' in options_df_for_period.columns:
        st.write("Unique sentiment values in input:", options_df_for_period['sentiment'].unique())
        st.write("Counts of sentiment values in input:", options_df_for_period['sentiment'].value_counts(dropna=False))
    else:
        st.warning("DEBUG: 'sentiment' column is NOT in options_df_for_period!")
    st.markdown("---")
    analysis_results = []
    unique_tickers = sorted(options_df_for_period['underlying_ticker'].unique())

    for ticker in unique_tickers:
        if not ticker or pd.isna(ticker) or str(ticker).strip().upper() == 'NAN' or not str(ticker).strip():
            continue

        market_cap = get_market_cap_st(ticker)
        current_price = get_current_price(ticker)
        ticker_df = options_df_for_period[options_df_for_period['underlying_ticker'] == ticker]

        total_premium_for_ticker, total_call_premium, total_put_premium, bullish_premium, bearish_premium = 0.0, 0.0, 0.0, 0.0, 0.0
        if 'premium_usd' in ticker_df.columns and pd.api.types.is_numeric_dtype(ticker_df['premium_usd']):
            ticker_df_cleaned_premium = ticker_df.dropna(subset=['premium_usd'])
            total_premium_for_ticker = ticker_df_cleaned_premium['premium_usd'].sum()
            total_call_premium = ticker_df_cleaned_premium[ticker_df_cleaned_premium['option_type'] == 'CALL']['premium_usd'].sum()
            total_put_premium = ticker_df_cleaned_premium[ticker_df_cleaned_premium['option_type'] == 'PUT']['premium_usd'].sum()
            bullish_premium = ticker_df_cleaned_premium[ticker_df_cleaned_premium['sentiment'] == 'Bullish']['premium_usd'].sum()
            bearish_premium = ticker_df_cleaned_premium[ticker_df_cleaned_premium['sentiment'] == 'Bearish']['premium_usd'].sum()
        
        mcap_val_actual = market_cap if market_cap else 0
        
        # --- NEW: Calculate Premium per $1M of Market Cap ---
        # Ratio = Premium / MarketCap
        # Premium per $1M MCap = (Premium / MarketCap) * 1,000,000
        
        call_prem_per_mil_mcap = (total_call_premium / mcap_val_actual) * 1_000_000 if mcap_val_actual > 0 else 0
        put_prem_per_mil_mcap = (total_put_premium / mcap_val_actual) * 1_000_000 if mcap_val_actual > 0 else 0
        total_prem_per_mil_mcap = (total_premium_for_ticker / mcap_val_actual) * 1_000_000 if mcap_val_actual > 0 else 0
        
        price_at_period_start = None
        price_change_pct = np.nan
        try:
            start_hist_date = pd.to_datetime(selected_range_start_date_dt)
            history = yf.Ticker(ticker).history(start=start_hist_date, end=(start_hist_date + pd.Timedelta(days=4)))
            if not history.empty:
                price_at_period_start = history['Close'].iloc[0]
                if current_price and price_at_period_start and price_at_period_start != 0:
                    price_change_pct = ((current_price - price_at_period_start) / price_at_period_start) * 100
        except Exception as e:
            print(f"Could not fetch hist price for {ticker} on {selected_range_start_date_dt.strftime('%Y-%m-%d')}: {e}")

        analysis_results.append({
            "Ticker": ticker, "Market Cap": market_cap, "Current Price": current_price,
            "Price at Period Start": price_at_period_start, "Price Change %": price_change_pct,
            
            "Total Activity Prem": total_premium_for_ticker, # Absolute premium
            "Call Vol. Prem": total_call_premium,           # Absolute premium
            "Put Vol. Prem": total_put_premium,             # Absolute premium
            "Bullish Prem": bullish_premium,                 # Absolute premium
            "Bearish Prem": bearish_premium,                 # Absolute premium

            # Renamed and recalculated MCap ratios
            "Total P/$M MCap": total_prem_per_mil_mcap,
            "Call P/$M MCap": call_prem_per_mil_mcap,
            "Put P/$M MCap": put_prem_per_mil_mcap,
        })
    return pd.DataFrame(analysis_results)

def create_expiration_summary_table(options_df_for_period):
    if options_df_for_period.empty or not all(col in options_df_for_period.columns for col in ['expiration_date', 'premium_usd', 'underlying_ticker', 'option_type', 'sentiment']):
        return pd.DataFrame()
    summary_df = options_df_for_period.copy()
    summary_df['expiration_date'] = pd.to_datetime(summary_df['expiration_date'], errors='coerce')
    summary_df.dropna(subset=['expiration_date', 'premium_usd'], inplace=True)
    if summary_df.empty: return pd.DataFrame()
    today_date = pd.to_datetime(datetime.now().date())
    summary_df['time_to_expiry_days'] = (summary_df['expiration_date'] - today_date).dt.days
    summary_df['call_premium_exp'] = np.where(summary_df['option_type'] == 'CALL', summary_df['premium_usd'], 0)
    summary_df['put_premium_exp'] = np.where(summary_df['option_type'] == 'PUT', summary_df['premium_usd'], 0)
    summary_df['bullish_premium_exp'] = np.where(summary_df['sentiment'] == 'Bullish', summary_df['premium_usd'], 0)
    summary_df['bearish_premium_exp'] = np.where(summary_df['sentiment'] == 'Bearish', summary_df['premium_usd'], 0)
    expiration_analysis = summary_df.groupby(['expiration_date', 'time_to_expiry_days']).agg(
        total_premium_expiring=('premium_usd', 'sum'), unique_tickers_expiring=('underlying_ticker', 'nunique'),
        total_call_premium_expiring=('call_premium_exp', 'sum'), total_put_premium_expiring=('put_premium_exp', 'sum'),
        total_bullish_premium_expiring=('bullish_premium_exp', 'sum'), total_bearish_premium_expiring=('bearish_premium_exp', 'sum'),
        number_of_options=('underlying_ticker', 'count')).reset_index()
    expiration_analysis.sort_values(by='time_to_expiry_days', ascending=True, inplace=True)
    expiration_analysis.rename(columns={
        'expiration_date': 'Expiration Date', 'time_to_expiry_days': 'Days to Expiry',
        'total_premium_expiring': 'Total Premium', 'unique_tickers_expiring': 'Unique Tickers',
        'total_call_premium_expiring': 'Call Premium', 'total_put_premium_expiring': 'Put Premium',
        'total_bullish_premium_expiring': 'Bullish Premium', 'total_bearish_premium_expiring': 'Bearish Premium',
        'number_of_options': 'Options Count'}, inplace=True)
    return expiration_analysis

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")

col_title, col_attribution = st.columns([0.1, 0.1]) # Adjust ratios as needed (e.g., 3:1)

with col_attribution:
    st.markdown(
        """
        <div style='text-align: right; font-size: small; color: grey;'>
            <p>Developed by Harman S.<br>
            Raw Data Assistance: TearRepresentative 56</p>
        </div>
        """,
        unsafe_allow_html=True
    )
st.title("📈 Options Flow Analyzer Dashboard")

# --- SIDEBAR ---
st.sidebar.header("Filters")

# Fetch the absolute earliest and latest dates of data in your database
min_overall_db_date, max_overall_db_date = get_db_date_range()
today = datetime.now().date()

# If no data at all, default to a reasonable past period (e.g., last year up to today).
calendar_min_boundary = min_overall_db_date if min_overall_db_date else today - pd.Timedelta(days=365)
calendar_max_boundary = today # Max selectable date is today

# Ensure calendar_min_boundary is not after calendar_max_boundary
if calendar_min_boundary > calendar_max_boundary:
    calendar_min_boundary = calendar_max_boundary - pd.Timedelta(days=1) # Adjust if something is odd

# Determine the default selected date for initial view (most recent data day or today)
if max_overall_db_date:
    initial_default_date = max_overall_db_date
    # Ensure default is not before the calendar's absolute minimum
    if initial_default_date < calendar_min_boundary:
        initial_default_date = calendar_min_boundary
    # Ensure default is not after the calendar's absolute maximum
    if initial_default_date > calendar_max_boundary:
        initial_default_date = calendar_max_boundary
else:
    # No data in DB, or error fetching range, default to today (or the max boundary)
    initial_default_date = calendar_max_boundary


# Set default values for the date pickers (both start and end to the same date initially)
default_start_val = initial_default_date
default_end_val = initial_default_date

# Ensure default_start_val is not after default_end_val (shouldn't be if they are same)
if default_start_val > default_end_val:
    default_start_val = default_end_val


selected_start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start_val,
    min_value=calendar_min_boundary,
    max_value=calendar_max_boundary, # Max selectable for start_date picker
    key="main_dashboard_start_date"
)

# For the end date picker, its min_value must be the currently selected_start_date
# and its value must be >= selected_start_date.
# Also, its value must be <= calendar_max_boundary.
final_default_end_val = default_end_val
if final_default_end_val < selected_start_date: # Ensure default end is not before selected start
    final_default_end_val = selected_start_date
if final_default_end_val > calendar_max_boundary: # Ensure default end is not after calendar max
    final_default_end_val = calendar_max_boundary


selected_end_date = st.sidebar.date_input(
    "End Date",
    value=final_default_end_val,
    min_value=selected_start_date,  # Min for end_date is the chosen start_date
    max_value=calendar_max_boundary, # Max selectable is overall calendar max
    key="main_dashboard_end_date"
)

if selected_start_date > selected_end_date:
    st.sidebar.error("Error: Start date must be before or same as end date.")
    st.stop()

st.sidebar.markdown("---")
searched_ticker = st.sidebar.text_input("Search Specific Ticker (e.g., NVDA):", key="main_ticker_search").strip().upper()

st.sidebar.markdown("---") # Visual separator
st.sidebar.caption(
    "**Disclaimer:** This tool is for informational and research purposes only. "
    "No financial decisions should be made solely based on the data presented here. "
    "Developer (Harman S.) is not responsible for any trading decisions or outcomes. "
    "This is a hobby development for data viewing and research."
)

# --- MAIN PAGE ---
# The header will now initially show a single day or the default range
header_title_suffix = ""
if selected_start_date == selected_end_date:
    header_title_suffix = f"for {selected_start_date.strftime('%Y-%m-%d')}"
else:
    header_title_suffix = f"from {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}"

if searched_ticker:
    st.header(f"Analysis for Ticker: {searched_ticker} ({header_title_suffix.replace('for ', '')})") # Adjust phrasing
else:
    st.header(f"Overall Analysis ({header_title_suffix})")


# Fetch data for the selected range (initially this will be for a single day)
options_data = fetch_options_activity_for_range(selected_start_date, selected_end_date)
view_data = options_data.copy()

if searched_ticker:
    if 'underlying_ticker' in view_data.columns:
        original_row_count = len(view_data)
        view_data = view_data[view_data['underlying_ticker'].str.upper() == searched_ticker]
        if view_data.empty and original_row_count > 0:
            st.warning(f"No data found for ticker '{searched_ticker}' within the selected date range.")
    elif not view_data.empty : 
        st.error("'underlying_ticker' column not found in the data. Cannot filter by ticker.")

# Debug line from user's code - remove or keep as needed
# st.dataframe(view_data.head()) 

if view_data.empty: # This is the primary check
    # Construct a helpful message
    message_parts = ["No options data found"]
    if searched_ticker:
        message_parts.append(f"for ticker '{searched_ticker}'")
    if selected_start_date == selected_end_date:
        message_parts.append(f"for {selected_start_date.strftime('%Y-%m-%d')}.")
    else:
        message_parts.append(f"for the period {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}.")
    message_parts.append("Please select a different date/range or check if data has been ingested for this period.")
    st.warning(" ".join(message_parts))
else:
    # --- Overall Activity Snapshot ---
    try:
        snapshot_title = f"Overall Activity Snapshot{f' for {searched_ticker}' if searched_ticker else ' (Selected Period)'}"
        with st.expander(snapshot_title, expanded=True):
            total_premium = view_data['premium_usd'].sum()
            num_tickers = view_data['underlying_ticker'].nunique()
            num_trades = len(view_data)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Premium", f"${total_premium:,.0f}")
            col2.metric("Trades Count", f"{num_trades}")
            col3.metric("Ticker(s) Active", f"{num_tickers if not searched_ticker else (searched_ticker if num_tickers > 0 else '0')}")
    except Exception as e:
        st.error(f"Error displaying Overall Activity Snapshot: {e}")
    # --- Distribution Charts ---
    # Check if data exists before attempting to plot these general distribution charts
    if not view_data.empty: 
        # Only show these general distribution charts if no specific ticker is searched OR if there's enough data for the searched ticker
        if not searched_ticker or (searched_ticker and len(view_data) > 1): # Arbitrary threshold for "enough data"
            with st.expander("Activity Distribution Charts", expanded=False):
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    if 'premium_usd' in view_data.columns:
                        fig_prem_size = px.histogram(view_data.dropna(subset=['premium_usd']), x="premium_usd", nbins=30, title="Distribution of Premium Sizes ($)")
                        st.plotly_chart(fig_prem_size, use_container_width=True)
                with col_chart2:
                    if 'expiration_date' in view_data.columns:
                        view_data_for_dte = view_data.copy()
                        view_data_for_dte['expiration_date'] = pd.to_datetime(view_data_for_dte['expiration_date'], errors='coerce')
                        view_data_for_dte.dropna(subset=['expiration_date'], inplace=True)
                        if not view_data_for_dte.empty:
                            view_data_for_dte['DTE'] = (view_data_for_dte['expiration_date'] - pd.to_datetime(datetime.now().date())).dt.days
                            fig_dte = px.histogram(view_data_for_dte.dropna(subset=['DTE']), x="DTE", nbins=30, title="Distribution of DTE (from today)")
                            st.plotly_chart(fig_dte, use_container_width=True)
                
                if 'sentiment' in view_data.columns and 'premium_usd' in view_data.columns:
                    sentiment_summary_for_pie = view_data.groupby('sentiment')['premium_usd'].sum().reset_index()
                    if not sentiment_summary_for_pie.empty and sentiment_summary_for_pie['premium_usd'].sum() > 0 :
                        fig_sentiment_pie = px.pie(sentiment_summary_for_pie, values='premium_usd', names='sentiment', title="Premium by Sentiment in Period", color_discrete_map={'Bullish':'green', 'Bearish':'red', 'Unknown':'grey'})
                        st.plotly_chart(fig_sentiment_pie, use_container_width=True)
        

    # --- Per-Ticker Analysis ---
    per_ticker_title = f"Detailed Ticker Analysis{f' for {searched_ticker}' if searched_ticker else ' (Per Ticker in Period)'}"
    with st.expander(per_ticker_title, expanded=True): # Often useful to have this expanded
        if view_data.empty: # Should ideally be caught by the main 'if view_data.empty:'
            st.info("No data available to conduct per-ticker analysis based on current filters.")
        else:
            with st.spinner(f"Analyzing data... Fetching market info..."):
                # Pass selected_start_date (as datetime object) for historical price context
                ticker_analysis_df = analyze_ticker_dashboard(view_data, pd.to_datetime(selected_start_date)) 
            
            if not ticker_analysis_df.empty:
                # 1. Define the desired column order for the display table
                ordered_cols = [
                    "Ticker", 
                    "Call P/$M MCap",       # Prominent, and for default sort
                    "Put P/$M MCap",        # Prominent
                    "Total P/$M MCap",
                    "Market Cap", 
                    "Current Price", 
                    "Price at Period Start", 
                    "Price Change %",
                    "Bullish Prem",
                    "Bearish Prem",
                    "Total Call Vol. Prem", 
                    "Total Put Vol. Prem",  
                    "Total Activity Prem"
                ]
                
                # Ensure all columns in ordered_cols exist in ticker_analysis_df, add with NaN if not
                # This prevents errors if analyze_ticker_dashboard changes its output slightly
                df_for_display_intermediate = ticker_analysis_df.copy()
                for col in ordered_cols:
                    if col not in df_for_display_intermediate.columns:
                        df_for_display_intermediate[col] = np.nan
                
                # Select only the ordered columns for the final display DataFrame
                df_for_display = df_for_display_intermediate[ordered_cols].copy()

                # 2. Ensure columns for sorting and styling are numeric
                numeric_cols_for_style_sort = [
                    "Call P/$M MCap", "Put P/$M MCap", "Total P/$M MCap",
                    "Price Change %"
                ] # Add others if they are styled with gradients and might not be numeric yet
                
                for col in numeric_cols_for_style_sort:
                    if col in df_for_display.columns:
                        df_for_display[col] = pd.to_numeric(df_for_display[col], errors='coerce')

                # 3. Apply Default Sort by "Call P/$M MCap" descending
                if "Call P/$M MCap" in df_for_display.columns and not df_for_display.empty:
                    # Sort before formatting strings, using the numeric values
                    df_for_display.sort_values(by="Call P/$M MCap", ascending=False, inplace=True)
                    df_for_display.reset_index(drop=True, inplace=True)

                # 4. Define Formatting for Styler
                format_dict = {}
                currency_cols_int = ['Market Cap', 'Total Activity Prem', 'Total Call Vol. Prem', 'Total Put Vol. Prem', 'Bullish Prem', 'Bearish Prem']
                currency_cols_float = ['Current Price', 'Price at Period Start']
                ratio_cols = ["Call P/$M MCap", "Put P/$M MCap", "Total P/$M MCap"]

                for col in currency_cols_int:
                    if col in df_for_display.columns: format_dict[col] = "${:,.0f}"
                for col in currency_cols_float:
                    if col in df_for_display.columns: format_dict[col] = "${:,.2f}"
                for col in ratio_cols: # Prem per $M MCap columns
                    if col in df_for_display.columns: format_dict[col] = "{:,.0f}" # Display as whole numbers
                if 'Price Change %' in df_for_display.columns:
                     format_dict['Price Change %'] = "{:.2f}%"

                # 5. Apply Styles using Pandas Styler
                styler = df_for_display.style
                
                # Background gradient for "Premium per $M MCap" columns
                # You can adjust cmap and vmin/vmax. Not setting vmax lets it scale to data.
                # For small ratios that become larger numbers (e.g., 349), vmax might be 500 or 1000.
                if "Call P/$M MCap" in df_for_display.columns:
                    styler = styler.background_gradient(subset=["Call P/$M MCap"], cmap='Greens', vmin=0)
                if "Put P/$M MCap" in df_for_display.columns:
                    styler = styler.background_gradient(subset=["Put P/$M MCap"], cmap='Reds', vmin=0) # Assuming higher put P/$M might be bearish interest
                if "Total P/$M MCap" in df_for_display.columns:
                    styler = styler.background_gradient(subset=["Total P/$M MCap"], cmap='Oranges', vmin=0)
                
                # Background gradient for Price Change %
                if 'Price Change %' in df_for_display.columns:
                     styler = styler.background_gradient(subset=['Price Change %'], cmap='RdYlGn', vmin=-10, vmax=10, axis=0) # vmin/vmax in percent

                # Apply formatting and other properties
                styler = styler.format(format_dict, na_rep="N/A")
                styler = styler.set_properties(**{'text-align': 'right'}) # Align text to right for numerics/currency

                # 6. Display the Styled DataFrame
                st.dataframe(styler, use_container_width=True)

                # 7. Chart: Top N Tickers (using original numeric data from ticker_analysis_df)
                #    This chart still uses "Total P/$M MCap" as an example, you can change to "Call P/$M MCap" if preferred
                if not searched_ticker: # Only show this general chart if not searching for a specific ticker
                    st.markdown("---") # Separator for the chart
                    top_n_key_metric = "Total P/$M MCap" # Or "Call P/$M MCap"
                    
                    if top_n_key_metric in ticker_analysis_df.columns:
                        top_n_count = st.slider(f"Number of Top Tickers to Chart ({top_n_key_metric}):", 
                                                min_value=5, max_value=25, value=10, 
                                                key="top_n_generic_mcap_slider")
                        
                        df_for_chart_top_n = ticker_analysis_df.copy()
                        # Ensure the metric column is numeric for sorting and plotting
                        df_for_chart_top_n[top_n_key_metric] = pd.to_numeric(df_for_chart_top_n[top_n_key_metric], errors='coerce')
                        
                        df_sorted_for_top_n_chart = df_for_chart_top_n.dropna(subset=[top_n_key_metric]).sort_values(
                            by=top_n_key_metric, ascending=False
                        ).head(top_n_count)
                        
                        if not df_sorted_for_top_n_chart.empty:
                            fig_top_tickers = px.bar(df_sorted_for_top_n_chart, 
                                                         x="Ticker", 
                                                         y=top_n_key_metric, 
                                                         title=f"Top {top_n_count} Tickers by {top_n_key_metric}",
                                                         hover_data=['Total Activity Prem', 'Market Cap'], # Use updated names
                                                         labels={top_n_key_metric: top_n_key_metric, 'Ticker': 'Ticker Symbol'})
                            st.plotly_chart(fig_top_tickers, use_container_width=True)
                        else:
                            st.caption(f"Not enough data to display Top Tickers by {top_n_key_metric} chart.")
            else:
                st.info("No detailed ticker analysis results to display based on current filters.")

    # --- Expiration Summary Table & Charts ---
    expiration_title = f"Expiration Date Summaries & Charts{f' for {searched_ticker}' if searched_ticker else ' (Selected Period)'}"
    with st.expander(expiration_title, expanded=False):
        expiration_summary_df = create_expiration_summary_table(view_data)
        if not expiration_summary_df.empty:
            # Charts first (using numeric data from expiration_summary_df)
            if 'Expiration Date' in expiration_summary_df.columns: # Check if valid for x-axis
                fig_total_prem_exp = px.bar(expiration_summary_df, x="Expiration Date", y="Total Premium", title="Total Premium by Expiration Date")
                st.plotly_chart(fig_total_prem_exp, use_container_width=True)
                
                df_melted_cp_exp = expiration_summary_df.melt(id_vars=['Expiration Date'], value_vars=['Call Premium', 'Put Premium'], var_name='Option Type', value_name='Premium')
                fig_cp_expiry = px.bar(df_melted_cp_exp, x='Expiration Date', y='Premium', color='Option Type', title='Call vs. Put Premium by Expiry', color_discrete_map={'Call Premium': 'mediumspringgreen', 'Put Premium': 'salmon'})
                st.plotly_chart(fig_cp_expiry, use_container_width=True)

                df_melted_sent_exp = expiration_summary_df.melt(id_vars=['Expiration Date'], value_vars=['Bullish Premium', 'Bearish Premium'], var_name='Sentiment Type', value_name='Premium')
                fig_sent_expiry = px.bar(df_melted_sent_exp, x='Expiration Date', y='Premium', color='Sentiment Type', title='Bullish vs. Bearish Premium by Expiry', color_discrete_map={'Bullish Premium': 'green', 'Bearish Premium': 'red'})
                st.plotly_chart(fig_sent_expiry, use_container_width=True)

                fig_count_expiry = px.bar(expiration_summary_df, x="Expiration Date", y="Options Count", title="Options Count by Expiration Date")
                st.plotly_chart(fig_count_expiry, use_container_width=True)

            # Then display the formatted table
            display_exp_summary_df = expiration_summary_df.copy()
            if 'Expiration Date' in display_exp_summary_df.columns:
                display_exp_summary_df['Expiration Date'] = pd.to_datetime(display_exp_summary_df['Expiration Date']).dt.strftime('%Y-%m-%d (%a)')
            for col_name in ['Total Premium', 'Call Premium', 'Put Premium', 'Bullish Premium', 'Bearish Premium']:
                if col_name in display_exp_summary_df.columns:
                    display_exp_summary_df[col_name] = display_exp_summary_df[col_name].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "$0")
            
            columns_to_display_exp = ['Expiration Date', 'Days to Expiry', 'Total Premium', 'Options Count', 'Unique Tickers', 'Call Premium', 'Put Premium', 'Bullish Premium', 'Bearish Premium']
            final_columns_exp = [col for col in columns_to_display_exp if col in display_exp_summary_df.columns]
            st.dataframe(display_exp_summary_df[final_columns_exp].reset_index(drop=True), use_container_width=True)
        else:
            st.info("No expiration summary to display based on current filters.")

    # --- Raw Data Display ---
    raw_data_title = f"Raw Options Data{f' for {searched_ticker}' if searched_ticker else ' (Selected Period)'}"
    with st.expander(raw_data_title, expanded=False): # Collapsed by default
        st.dataframe(view_data, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Provisional Dashboard - Data from `yfinance` is subject to its terms and can have delays.")

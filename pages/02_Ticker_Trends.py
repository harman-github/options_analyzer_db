# pages/02_Ticker_Trends.py
import streamlit as st
import pandas as pd
from sqlalchemy import text # Ensure 'text' is imported if used
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# DB_HOST = "localhost"
# DB_PORT = "5432"
# DB_NAME = "options_analyzer_db"
# DB_USER = "options_app_user"
# DB_PASSWORD = "Morpheus321"  # <<< MAKE SURE THIS IS YOUR CORRECT PASSWORD
# db_connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

@st.cache_data(ttl=3600)
def get_all_unique_tickers_from_db():
    """Fetches all unique underlying tickers from the database."""
    try:
        # Now db_connection_string is defined in this script's scope
        # engine = create_engine(db_connection_string)
        conn = st.connection("postgresql", type="sql")
        query = "SELECT DISTINCT underlying_ticker FROM options_activity ORDER BY underlying_ticker;"
        df = conn.query(query, ttl=0)
        return df['underlying_ticker'].tolist()
    except Exception as e:
        st.error(f"Error fetching unique tickers: {e}")
        return []

@st.cache_data(ttl=3600)
def get_all_unique_tickers_from_db():
    """Fetches all unique underlying tickers from the database."""
    try:
        # engine = create_engine(db_connection_string)
        conn = st.connection("postgresql", type="sql")
        query = "SELECT DISTINCT underlying_ticker FROM options_activity ORDER BY underlying_ticker;"
        df = conn.query(query, ttl=0)
        return df['underlying_ticker'].tolist()
    except Exception as e:
        st.error(f"Error fetching unique tickers: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_activity_for_specific_ticker(ticker_symbol, start_date=None, end_date=None):
    """Fetches all activity for a specific ticker, optionally within a date range."""
    try:
        # engine = create_engine(db_connection_string)
        conn = st.connection("postgresql", type="sql")
        params = {'ticker': ticker_symbol}
        date_conditions = []
        if start_date:
            params['start_date'] = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            date_conditions.append("data_date >= :start_date")
        if end_date:
            params['end_date'] = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            date_conditions.append("data_date <= :end_date")
        
        date_filter_sql = " AND ".join(date_conditions) if date_conditions else ""

        query = f"""
        SELECT * FROM options_activity 
        WHERE underlying_ticker = :ticker
        { "AND " + date_filter_sql if date_filter_sql else ""}
        ORDER BY data_date;
        """
        df = conn.query(query, params=params, ttl=0)
        if not df.empty:
            df['data_date'] = pd.to_datetime(df['data_date'])
            df['expiration_date'] = pd.to_datetime(df['expiration_date'], errors='coerce')
            df['premium_usd'] = pd.to_numeric(df['premium_usd'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400) # Cache historical prices for a day
def fetch_stock_history(ticker_symbol, start_date, end_date):
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = (end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d') # yfinance end is exclusive
        hist = yf.Ticker(ticker_symbol).history(start=start_str, end=end_str, auto_adjust=True)
        return hist['Close']
    except Exception as e:
        print(f"Error fetching stock history for {ticker_symbol}: {e}")
        return pd.Series(dtype=float)

# --- Streamlit Page Layout ---
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

st.title("🔎 Individual Ticker - Historical Options Flow Trends")

all_tickers = get_all_unique_tickers_from_db()

if not all_tickers:
    st.warning("No tickers found in the database to analyze.")
    st.stop()

selected_ticker = st.sidebar.selectbox("Select a Ticker to Analyze:", all_tickers, index=0 if all_tickers else None)

if selected_ticker:
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Date Range for {selected_ticker} History:")
    
    # Fetch min/max dates for the selected ticker to set as defaults/bounds
    ticker_specific_activity_full_range = fetch_activity_for_specific_ticker(selected_ticker)
    
    if not ticker_specific_activity_full_range.empty:
        min_ticker_date = ticker_specific_activity_full_range['data_date'].min().date()
        max_ticker_date = ticker_specific_activity_full_range['data_date'].max().date()

        trend_start_date = st.sidebar.date_input("Trend Start Date", value=min_ticker_date, 
                                                 min_value=min_ticker_date, max_value=max_ticker_date, 
                                                 key=f"{selected_ticker}_trend_start")
        trend_end_date = st.sidebar.date_input("Trend End Date", value=max_ticker_date, 
                                               min_value=trend_start_date, max_value=max_ticker_date, 
                                               key=f"{selected_ticker}_trend_end")

        if trend_start_date > trend_end_date:
            st.sidebar.error("Trend start date must be before or same as trend end date.")
            st.stop()

        # Fetch data for the selected ticker within the chosen trend date range
        ticker_activity_df = fetch_activity_for_specific_ticker(selected_ticker, trend_start_date, trend_end_date)
    else:
        st.warning(f"No activity data found for {selected_ticker} in the database.")
        ticker_activity_df = pd.DataFrame()


    if not ticker_activity_df.empty:
        st.header(f"Historical Trends for: {selected_ticker}")
        st.markdown(f"Displaying data from **{trend_start_date.strftime('%Y-%m-%d')}** to **{trend_end_date.strftime('%Y-%m-%d')}**")

        # 1. Overall Sentiment Summary for Selected Ticker & Period
        with st.expander("Overall Sentiment Summary", expanded=True):
            bullish_total = ticker_activity_df[ticker_activity_df['sentiment'] == 'Bullish']['premium_usd'].sum()
            bearish_total = ticker_activity_df[ticker_activity_df['sentiment'] == 'Bearish']['premium_usd'].sum()
            net_bullish = bullish_total - bearish_total
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Bullish Premium", f"${bullish_total:,.0f}")
            col2.metric("Total Bearish Premium", f"${bearish_total:,.0f}")
            col3.metric("Net Bullish Premium", f"${net_bullish:,.0f}")

            first_activity = ticker_activity_df['data_date'].min().strftime('%Y-%m-%d')
            last_activity = ticker_activity_df['data_date'].max().strftime('%Y-%m-%d')
            st.caption(f"First activity in selected range: {first_activity} | Last activity: {last_activity}")


        # 2. Sentiment Timeline Chart & Stock Price Overlay
        with st.expander("Sentiment & Price Timeline", expanded=True):
            # Aggregate net bullish premium per day
            daily_sentiment = ticker_activity_df.groupby('data_date')['premium_usd'].apply(
                lambda x: ticker_activity_df.loc[x.index, 'premium_usd'][ticker_activity_df.loc[x.index, 'sentiment'] == 'Bullish'].sum() - 
                          ticker_activity_df.loc[x.index, 'premium_usd'][ticker_activity_df.loc[x.index, 'sentiment'] == 'Bearish'].sum()
            ).reset_index(name='Net Bullish Premium')
            
            stock_price_history = fetch_stock_history(selected_ticker, trend_start_date, trend_end_date)

            if not daily_sentiment.empty:
                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Bar(x=daily_sentiment['data_date'], y=daily_sentiment['Net Bullish Premium'], 
                                              name='Net Bullish Premium', marker_color='lightblue'))
                
                if not stock_price_history.empty:
                    fig_timeline.add_trace(go.Scatter(x=stock_price_history.index, y=stock_price_history, 
                                                      name='Stock Price', yaxis='y2', line=dict(color='orange')))
                
                fig_timeline.update_layout(
                    title_text=f"Daily Net Bullish Premium & Stock Price for {selected_ticker}",
                    xaxis_title="Date",
                    yaxis_title="Net Bullish Premium ($)",
                    yaxis2=dict(title="Stock Price ($)", overlaying='y', side='right', showgrid=False),
                    legend_title_text='Legend'
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("Not enough data to plot sentiment timeline.")

        # 3. Cumulative Net Premium Flow Chart
        with st.expander("Cumulative Net Premium Flow", expanded=False):
            if not daily_sentiment.empty:
                daily_sentiment['Cumulative Net Premium'] = daily_sentiment['Net Bullish Premium'].cumsum()
                fig_cumulative = px.line(daily_sentiment, x='data_date', y='Cumulative Net Premium',
                                         title=f"Cumulative Net Premium Flow for {selected_ticker}",
                                         labels={'data_date': 'Date', 'Cumulative Net Premium': 'Cumulative Net Premium ($)'})
                st.plotly_chart(fig_cumulative, use_container_width=True)

        # 4. Significant Flow Days Table
        with st.expander("Top Significant Flow Days", expanded=False):
            if not daily_sentiment.empty:
                # Add absolute net premium for sorting by magnitude
                daily_sentiment['Abs Net Premium'] = daily_sentiment['Net Bullish Premium'].abs()
                top_n_days = daily_sentiment.sort_values(by='Abs Net Premium', ascending=False).head(10)
                st.markdown("Top 10 days by magnitude of Net Bullish/Bearish Premium:")
                # Select and format columns for display
                display_top_days = top_n_days[['data_date', 'Net Bullish Premium']].copy()
                display_top_days['data_date'] = pd.to_datetime(display_top_days['data_date']).dt.strftime('%Y-%m-%d')
                display_top_days['Net Bullish Premium'] = display_top_days['Net Bullish Premium'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(display_top_days.reset_index(drop=True), use_container_width=True)
            else:
                st.info("No daily sentiment data to show top flow days.")
        
        # 5. Price Performance During Activity Lifecycle (within selected trend range)
        with st.expander("Price Performance in Selected Trend Range", expanded=False):
            if not stock_price_history.empty:
                start_price = stock_price_history.iloc[0]
                end_price = stock_price_history.iloc[-1]
                price_change_val = end_price - start_price
                price_change_pct = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0
                
                st.metric(label=f"Start Price ({stock_price_history.index[0].strftime('%Y-%m-%d')})", value=f"${start_price:,.2f}")
                st.metric(label=f"End Price ({stock_price_history.index[-1].strftime('%Y-%m-%d')})", value=f"${end_price:,.2f}", delta=f"${price_change_val:,.2f} ({price_change_pct:.2f}%)")
            else:
                st.info(f"Could not load price history for {selected_ticker} in the selected range.")
    else:
        if selected_ticker: # Only show if a ticker was selected but no data found for it
            st.info(f"No options activity data in the database for {selected_ticker} within the specified date range to display trends.")

else:
    st.info("Select a ticker from the sidebar to view its historical trends.")

st.sidebar.markdown("---") # Visual separator
st.sidebar.caption(
    "**Disclaimer:** This tool is for informational and research purposes only. "
    "No financial decisions should be made solely based on the data presented here. "
    "Developer (Harman S.) is not responsible for any trading decisions or outcomes. "
    "This is a hobby development for data viewing and research."
)

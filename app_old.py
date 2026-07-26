import math
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='S&P 500 Returns & Drawdowns', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
div[data-testid="metric-container"] {background: #f7f7f5; border: 1px solid #e3e2dd; padding: 14px 16px; border-radius: 14px;}
[data-testid="stSidebar"] {background: #f8f8f6;}
</style>
''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    import yfinance as yf
    df = yf.download('^GSPC', start='1957-01-01', auto_adjust=False, progress=False, interval='1d', threads=False)
    if df.empty:
        raise ValueError('Yahoo Finance returned no data for ^GSPC.')
    if isinstance(df.columns, pd.MultiIndex):
        if ('Close', '^GSPC') in df.columns:
            close = df[('Close', '^GSPC')].copy()
        else:
            close = df.xs('Close', axis=1, level=0).iloc[:, 0].copy()
    else:
        close = df['Close'].copy()
    close = close.dropna().to_frame('close')
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    close = close[close.index.year >= 1957]
    return close


def calc_year_stats(df_year: pd.DataFrame) -> dict:
    s = df_year['close'].astype(float).copy()
    running_peak = s.cummax()
    drawdown = (s - running_peak) / running_peak

    year = int(s.index[0].year)
    start_price = float(s.iloc[0])
    end_price = float(s.iloc[-1])
    annual_return = (end_price / start_price) - 1.0

    trough_date = drawdown.idxmin()
    max_drawdown = float(drawdown.min())

    peak_window = s.loc[:trough_date]
    peak_price = float(peak_window.max())
    peak_date = peak_window[peak_window == peak_price].index[0]
    trough_price = float(s.loc[trough_date])
    trading_days = int(s.index.get_loc(trough_date) - s.index.get_loc(peak_date))

    if max_drawdown >= 0:
        peak_date = s.index[0]
        trough_date = s.index[0]
        peak_price = float(s.iloc[0])
        trough_price = float(s.iloc[0])
        trading_days = 0
        max_drawdown = 0.0

    return {
        'year': year,
        'start_date': s.index[0].date(),
        'end_date': s.index[-1].date(),
        'start_price': round(start_price, 2),
        'end_price': round(end_price, 2),
        'annual_return_pct': annual_return * 100,
        'max_drawdown_pct': max_drawdown * 100,
        'peak_date': peak_date.date(),
        'peak_price': round(peak_price, 2),
        'trough_date': trough_date.date(),
        'trough_price': round(trough_price, 2),
        'trading_days_peak_to_trough': trading_days,
    }


@st.cache_data(show_spinner=False)
def build_yearly_table(close_df: pd.DataFrame):
    rows = []
    by_year = {}
    for year, grp in close_df.groupby(close_df.index.year):
        grp = grp.loc[(grp.index >= pd.Timestamp(f'{year}-01-01')) & (grp.index <= pd.Timestamp(f'{year}-12-31'))]
        if not grp.empty:
            rows.append(calc_year_stats(grp))
            by_year[int(year)] = grp
    table = pd.DataFrame(rows).sort_values('year').reset_index(drop=True)
    return table, by_year


def make_combo_chart(df_plot: pd.DataFrame):
    colors = np.where(df_plot['annual_return_pct'] >= 0, '#22b14c', '#ff1d25')
    fig = go.Figure()
    fig.add_bar(
        x=df_plot['year'],
        y=df_plot['annual_return_pct'],
        name='Annual return',
        marker_color=colors,
        hovertemplate='Year %{x}<br>Return %{y:.2f}%<extra></extra>'
    )
    fig.add_scatter(
        x=df_plot['year'],
        y=df_plot['max_drawdown_pct'],
        mode='markers+lines',
        name='Max drawdown',
        line=dict(color='#f28c28', width=2, dash='dot'),
        marker=dict(color='#f28c28', size=9),
        hovertemplate='Year %{x}<br>Max drawdown %{y:.2f}%<extra></extra>'
    )
    fig.update_layout(
        title='S&P 500 Returns and Drawdowns',
        paper_bgcolor='#efefed',
        plot_bgcolor='#efefed',
        height=620,
        bargap=0.18,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=20, r=20, t=80, b=20),
        font=dict(color='#0b4f7d', size=16),
        xaxis=dict(title='', tickmode='linear', showgrid=False, zeroline=False),
        yaxis=dict(title='Percent', ticksuffix='%', gridcolor='rgba(60,60,60,0.20)', zerolinecolor='rgba(80,80,80,0.45)'),
    )
    return fig


def make_year_path_chart(df_year: pd.DataFrame, year: int):
    s = df_year['close'].astype(float)
    running_peak = s.cummax()
    dd = ((s - running_peak) / running_peak) * 100
    fig = go.Figure()
    fig.add_scatter(
        x=s.index,
        y=dd,
        mode='lines',
        line=dict(color='#f28c28', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(242,140,40,0.15)',
        name=f'{year} drawdown path',
        hovertemplate='%{x|%Y-%m-%d}<br>Drawdown %{y:.2f}%<extra></extra>'
    )
    fig.update_layout(
        title=f'{year} drawdown path within calendar year',
        height=340,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(title=''),
        yaxis=dict(title='Drawdown', ticksuffix='%', gridcolor='rgba(60,60,60,0.12)'),
        showlegend=False,
    )
    return fig


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')


st.title('S&P 500 Annual Returns and Drawdowns')
st.caption('Uses Yahoo Finance ^GSPC daily closing prices. Annual return is first close to last close within each calendar year. Max drawdown is computed strictly within each calendar year using running peaks formed only inside that year.')

with st.sidebar:
    st.header('Controls')
    st.write('Refresh data to pull the latest available Yahoo Finance history for ^GSPC.')
    refresh = st.button('Refresh Yahoo data', use_container_width=True)
    year_focus = st.number_input('Focus year', min_value=1957, max_value=2100, value=2020, step=1)
    show_tail = st.slider('Years shown in main chart', min_value=10, max_value=80, value=36, step=1)
    st.markdown('---')
    st.markdown('**Methodology**')
    st.markdown('- Daily close only')
    st.markdown('- Jan 1 to Dec 31 window only')
    st.markdown('- Running peak resets each calendar year')
    st.markdown('- Peak date must be on or before trough date')

if refresh:
    load_data.clear()
    build_yearly_table.clear()

try:
    close_df = load_data()
    yearly, by_year = build_yearly_table(close_df)
except Exception as e:
    st.error(f'Data load failed: {e}')
    st.stop()

latest_year = int(yearly['year'].max())
if year_focus not in by_year:
    year_focus = latest_year

bench = yearly.loc[yearly['year'] == 2020].iloc[0] if 2020 in yearly['year'].values else None
latest = yearly.iloc[-1]
worst = yearly.loc[yearly['max_drawdown_pct'].idxmin()]
best = yearly.loc[yearly['annual_return_pct'].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric('Years covered', f"{int(yearly['year'].min())}-{int(yearly['year'].max())}")
c2.metric('Worst annual max drawdown', f"{worst['max_drawdown_pct']:.2f}%", str(int(worst['year'])))
c3.metric('Best annual return', f"{best['annual_return_pct']:.2f}%", str(int(best['year'])))
if bench is not None:
    c4.metric('2020 benchmark drawdown', f"{bench['max_drawdown_pct']:.2f}%", f"Peak {bench['peak_date']} → Trough {bench['trough_date']}")
else:
    c4.metric('Latest year return', f"{latest['annual_return_pct']:.2f}%", str(int(latest['year'])))

plot_df = yearly.tail(show_tail)
st.plotly_chart(make_combo_chart(plot_df), use_container_width=True)

left, right = st.columns([1.5, 1])
with left:
    display = yearly.copy()
    date_cols = ['start_date', 'end_date', 'peak_date', 'trough_date']
    for col in date_cols:
        display[col] = display[col].astype(str)
    fmt = {
        'annual_return_pct': '{:.2f}%',
        'max_drawdown_pct': '{:.2f}%',
        'start_price': '{:,.2f}',
        'end_price': '{:,.2f}',
        'peak_price': '{:,.2f}',
        'trough_price': '{:,.2f}',
        'trading_days_peak_to_trough': '{:d}',
    }
    st.subheader('Yearly table')
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button('Download annual returns and drawdowns CSV', data=to_csv_download(display), file_name='gspc_annual_returns_and_drawdowns.csv', mime='text/csv')

with right:
    st.subheader(f'{year_focus} validation')
    row = yearly.loc[yearly['year'] == year_focus].iloc[0]
    st.write(f"**Annual return:** {row['annual_return_pct']:.2f}%")
    st.write(f"**Max drawdown:** {row['max_drawdown_pct']:.2f}%")
    st.write(f"**Peak:** {row['peak_date']} at {row['peak_price']:,.2f}")
    st.write(f"**Trough:** {row['trough_date']} at {row['trough_price']:,.2f}")
    st.write(f"**Trading days peak to trough:** {int(row['trading_days_peak_to_trough'])}")
    if year_focus == 2020:
        st.info('Benchmark check: expected peak Feb 19, 2020 at 3,386.15 and trough Mar 23, 2020 at 2,237.40 for about -33.92%. Yahoo close history should reproduce this to normal rounding tolerance.')
    st.plotly_chart(make_year_path_chart(by_year[year_focus], year_focus), use_container_width=True)

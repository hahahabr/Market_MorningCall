import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from statsmodels.tsa.filters.hp_filter import hpfilter

warnings.filterwarnings("ignore")

# ============================================================
# 기본 설정
# ============================================================
# API 키는 코드에 직접 적지 않고 Streamlit secrets에서 읽어옵니다.
# 로컬 실행: 프로젝트 폴더에 .streamlit/secrets.toml 파일을 만들고 아래처럼 채워주세요.
#   FRED_API_KEY = "여기에_FRED_API_키"
#   ECOS_API_KEY = "여기에_ECOS_API_키"
# Streamlit Cloud 배포: 앱 설정(Settings) > Secrets 메뉴에 동일한 내용을 붙여넣으세요.
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
ECOS_API_KEY = st.secrets.get("ECOS_API_KEY", "")

st.set_page_config(
    page_title="Market Morning Call",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

if not FRED_API_KEY or not ECOS_API_KEY:
    st.warning(
        "FRED_API_KEY 또는 ECOS_API_KEY가 설정되지 않았어요. "
        "Streamlit Cloud의 Settings > Secrets에 키를 추가하면 금리 데이터가 정상적으로 표시됩니다.",
        icon="⚠️",
    )

# 깔끔한 디자인을 위한 커스텀 CSS (HTML 버전과 비슷한 톤)
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1a1a2e; }
    h2 { border-left: 4px solid #1a73e8; padding-left: 10px; margin-top: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; padding: 0 18px; border-radius: 6px 6px 0 0;
    }
    .caption-note { color: #888; font-size: 0.82rem; margin: 4px 0 12px; }
    .warn-note {
        color: #c0392b; background: #fff8f0; padding: 10px 14px;
        border-radius: 6px; border-left: 4px solid #e67e22; margin: 8px 0;
    }
    .info-note {
        background: #e8f0fe; border-left: 4px solid #1a73e8; padding: 12px 16px;
        border-radius: 6px; margin-bottom: 16px; font-size: 0.92rem; line-height: 1.6;
    }
    .metric-row { display: flex; gap: 24px; flex-wrap: wrap; margin: 4px 0 12px; font-size: 0.92rem; color: #444; }
    .signal-badge {
        font-size: 0.78rem; font-weight: normal; background: #f0f0f0;
        border-radius: 12px; padding: 2px 10px; margin-left: 8px; color: #555;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 데이터 수집 함수 (시장 데이터)
# ============================================================
def get_currency(ticker):
    t = ticker.upper()
    # 주가지수는 화폐 단위가 아니라 포인트(pt)로 표시
    INDEX_TICKERS = {
        "^GSPC", "^IXIC", "^DJI", "^VIX", "^STOXX50E", "^GDAXI",
        "^KS11", "^KQ11", "000001.SS", "^N225", "^NSEI",
    }
    if t in INDEX_TICKERS:
        return "pt"
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "KRW"
    return "USD"


@st.cache_data(ttl=300, show_spinner=False)
def calculate_performance(ticker, name, label_col="종목명", currency=None, as_of_date=None):
    """
    as_of_date: 'YYYY-MM-DD' 문자열 또는 None.
                지정하면 그 날짜를 '오늘'처럼 취급해서 1D/1W/1M/3M/1Y/YTD를 재계산.
                None이면 가장 최근 거래일 기준(기존 동작).
    """
    try:
        if as_of_date is not None:
            now = pd.Timestamp(as_of_date)
            start = f"{now.year - 1}-01-01"
            end = (now + pd.DateOffset(days=1)).strftime("%Y-%m-%d")  # 기준일 포함, 그 이후는 제외
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        else:
            now = pd.Timestamp.now()
            start = f"{now.year - 1}-01-01"
            data = yf.download(ticker, start=start, progress=False, auto_adjust=True)

        if data.empty or len(data) < 5:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        cur = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[-2])

        def ref(offset):
            sub = data[data.index >= offset]
            return float(sub["Close"].iloc[0]) if not sub.empty else cur

        ccy = currency if currency is not None else get_currency(ticker)
        return {
            label_col: name,
            "통화": ccy,
            "전일종가": round(cur, 2),
            "기준일": data.index[-1].strftime("%Y-%m-%d"),
            "1D(%)": round((cur - prev) / prev * 100, 2),
            "1W(%)": round((cur - ref(now - pd.DateOffset(weeks=1))) / ref(now - pd.DateOffset(weeks=1)) * 100, 2),
            "1M(%)": round((cur - ref(now - pd.DateOffset(months=1))) / ref(now - pd.DateOffset(months=1)) * 100, 2),
            "3M(%)": round((cur - ref(now - pd.DateOffset(months=3))) / ref(now - pd.DateOffset(months=3)) * 100, 2),
            "1Y(%)": round((cur - ref(now - pd.DateOffset(years=1))) / ref(now - pd.DateOffset(years=1)) * 100, 2),
            "YTD(%)": round((cur - ref(pd.Timestamp(f"{now.year}-01-01"))) / ref(pd.Timestamp(f"{now.year}-01-01")) * 100, 2),
        }
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker, period="1y"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history_range(ticker, start, end=None):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fred(series_id, start="2000-01-01"):
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": FRED_API_KEY,
                    "file_type": "json", "observation_start": start},
            timeout=15,
        )
        df = pd.DataFrame(resp.json()["observations"])[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna().set_index("date").rename(columns={"value": series_id})
    except Exception:
        return pd.DataFrame()


# ============================================================
# 한국은행 ECOS API - 국고채 금리
# ============================================================
ECOS_BASE_URL = "https://ecos.bok.or.kr/api"
ECOS_BOND_STAT_CODE = "817Y002"
ECOS_ITEM_CODE_KR3Y = "010200000"
ECOS_ITEM_CODE_KR10Y = "010210000"


def _ecos_find_item_code(keyword, stat_code=ECOS_BOND_STAT_CODE):
    try:
        url = f"{ECOS_BASE_URL}/StatisticItemList/{ECOS_API_KEY}/json/kr/1/100/{stat_code}"
        resp = requests.get(url, timeout=15)
        rows = resp.json().get("StatisticItemList", {}).get("row", [])
        for row in rows:
            if keyword in row.get("ITEM_NAME1", ""):
                return row.get("ITEM_CODE1")
    except Exception:
        pass
    return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_ecos_market_rate(item_code, item_keyword, start="20200101", end=None, stat_code=ECOS_BOND_STAT_CODE):
    if end is None:
        end = pd.Timestamp.now().strftime("%Y%m%d")

    debug_info = {}

    def _call(code):
        url = (f"{ECOS_BASE_URL}/StatisticSearch/{ECOS_API_KEY}/json/kr/1/3000/"
               f"{stat_code}/D/{start}/{end}/{code}")
        resp = requests.get(url, timeout=15)
        data = resp.json()
        debug_info["url"] = url
        debug_info["status_code"] = resp.status_code
        debug_info["raw_response"] = data
        rows = data.get("StatisticSearch", {}).get("row")
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)[["TIME", "DATA_VALUE"]]
        df["TIME"] = pd.to_datetime(df["TIME"], format="%Y%m%d")
        df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
        return df.dropna().set_index("TIME").rename(columns={"DATA_VALUE": "value"})

    try:
        df = _call(item_code)
        if not df.empty:
            return df, debug_info
        found_code = _ecos_find_item_code(item_keyword, stat_code)
        if found_code:
            df = _call(found_code)
        return df, debug_info
    except Exception as e:
        debug_info["exception"] = str(e)
        return pd.DataFrame(), debug_info


# ============================================================
# HP 필터
# ============================================================
def compute_hp_signal(prices, lam=1600):
    cycle_arr, trend_arr = hpfilter(prices.values, lamb=lam)
    sigma = np.std(cycle_arr)
    df = pd.DataFrame(
        {"price": prices.values, "trend": trend_arr, "cycle": cycle_arr},
        index=prices.index
    )
    df["sigma"] = sigma
    df["signal"] = "중립"
    df.loc[df["cycle"] > sigma, "signal"] = "과열(매도)"
    df.loc[df["cycle"] < -sigma, "signal"] = "과매도(매수)"
    return df


def pct_color(v):
    return "#c0392b" if v > 0 else ("#2471a3" if v < 0 else "#555555")


def style_pct_df(df, pct_cols, price_col=None):
    """percent 컬럼에 색상/+- 부호를 입힌 Styler 반환"""
    def _color(val):
        try:
            v = float(val)
            return f"color: {pct_color(v)}; font-weight: bold"
        except Exception:
            return ""
    fmt = {c: "{:+.2f}%" for c in pct_cols if c in df.columns}
    if price_col and price_col in df.columns:
        fmt[price_col] = "{:,.2f}"
    styler = df.style.format(fmt)
    valid_pct_cols = [c for c in pct_cols if c in df.columns]
    if valid_pct_cols:
        styler = styler.applymap(_color, subset=valid_pct_cols)
    return styler


# ============================================================
# 개별 종목 상승 Top 5 (미국 / 한국)
# ============================================================
# 미국 시가총액 상위권 + 인지도 높은 종목 풀 (S&P500 전체 스캔은 무거워서 대표 풀로 대체)
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_name(ticker: str) -> str:
    """yfinance에서 종목/지수의 실제 이름을 가져온다. 못 가져오면 티커 그대로 반환."""
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName")
        if name:
            return name
    except Exception:
        pass
    return ticker


YAHOO_GAINERS_URL = "https://finance.yahoo.com/markets/stocks/gainers/"
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo_us_top_movers(top_n: int = 5) -> pd.DataFrame:
    """
    Yahoo Finance 'Top Gainers' 페이지를 실시간으로 스크래핑해서 미국 시장 전체 기준
    상승률 상위 N개를 반환. 실패 시 빈 DataFrame 반환 (호출부에서 자연스럽게 숨김 처리).
    컬럼: 종목명, 티커, 전일종가, 1D(%)
    """
    try:
        resp = requests.get(YAHOO_GAINERS_URL, headers=YAHOO_HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        rows = []
        # Yahoo는 테이블 행마다 종목 링크(/quote/TICKER/)를 가지고 있음
        for tr in soup.find_all("tr"):
            link = tr.find("a", href=re.compile(r"/quote/"))
            if not link:
                continue
            m = re.search(r"/quote/([A-Za-z0-9\.\-]+)", link["href"])
            if not m:
                continue
            ticker = m.group(1)

            tds = tr.find_all("td")
            if len(tds) < 5:
                continue

            row_text = [td.get_text(strip=True) for td in tds]
            # 회사명은 보통 두번째 컬럼(티커 다음)
            name = row_text[1] if len(row_text) > 1 and row_text[1] else ticker

            price = None
            pct = None
            for cell in row_text:
                if pct is None and "%" in cell:
                    m_pct = re.search(r"[-+]?\d+\.?\d*%", cell)
                    if m_pct:
                        pct = float(m_pct.group().replace("%", ""))
                if price is None:
                    cell_clean = cell.replace(",", "")
                    m_price = re.fullmatch(r"[-+]?\d+\.?\d*", cell_clean)
                    if m_price:
                        price = float(cell_clean)

            if price is None or pct is None:
                continue

            rows.append({
                "종목명": name,
                "티커": ticker,
                "전일종가": price,
                "1D(%)": pct,
            })

        if not rows:
            return pd.DataFrame()

        df_out = pd.DataFrame(rows).drop_duplicates(subset="티커")
        df_out = df_out.sort_values("1D(%)", ascending=False).head(top_n).reset_index(drop=True)
        return df_out
    except Exception:
        return pd.DataFrame()


NAVER_SISE_RISE_URL = "https://finance.naver.com/sise/sise_rise.naver"
NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_naver_top_movers(market: str = "kospi", top_n: int = 5) -> pd.DataFrame:
    """
    네이버 금융 상승률 상위 페이지(class="type_2" 테이블)를 실시간으로 스크래핑.
    market: 'kospi' 또는 'kosdaq' (코스닥은 ?sosok=1 파라미터)
    실패 시 빈 DataFrame 반환 (호출부에서 자연스럽게 숨김 처리 가능).
    컬럼: 종목명, 티커, 전일종가, 1D(%)
    """
    try:
        params = {"sosok": "1"} if market == "kosdaq" else {}
        resp = requests.get(NAVER_SISE_RISE_URL, params=params, headers=NAVER_HEADERS, timeout=10)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.find("table", class_="type_2")
        if table is None:
            return pd.DataFrame()

        rows = []
        for tr in table.find_all("tr"):
            name_link = tr.find("a", class_="tltle")
            if name_link is None:
                continue
            m = re.search(r"code=(\d+)", name_link["href"])
            if not m:
                continue
            code = m.group(1)
            name = name_link.get_text(strip=True)

            number_cells = tr.find_all("td", class_="number")
            if len(number_cells) < 3:
                continue

            try:
                price = float(number_cells[0].get_text(strip=True).replace(",", ""))
            except ValueError:
                continue

            pct_text = number_cells[2].get_text(strip=True).replace("%", "").replace("+", "")
            try:
                pct = float(pct_text)
            except ValueError:
                continue

            rows.append({
                "종목명": name,
                "티커": f"{code}.KS" if market == "kospi" else f"{code}.KQ",
                "전일종가": price,
                "1D(%)": pct,
            })

        if not rows:
            return pd.DataFrame()

        df_out = pd.DataFrame(rows).sort_values("1D(%)", ascending=False).head(top_n).reset_index(drop=True)
        return df_out
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def enrich_with_3m_ytd(df_top: pd.DataFrame, as_of_date=None) -> pd.DataFrame:
    """크롤링으로 가져온 Top N 종목에 3M(%), YTD(%)를 yfinance로 추가 조회.
    전일종가가 없는 입력(예: 다음 금융처럼 등락률만 제공하는 경우)도 처리 가능 -
    그 경우 yfinance 조회 결과의 가격을 사용."""
    if df_top.empty:
        return df_top
    has_price = "전일종가" in df_top.columns
    rows = []
    for _, row in df_top.iterrows():
        r = calculate_performance(row["티커"], row["종목명"], label_col="종목명", as_of_date=as_of_date)
        price = row["전일종가"] if has_price else (r["전일종가"] if r else None)
        if r:
            rows.append({
                "종목명": row["종목명"],
                "전일종가": price,
                "1D(%)": row["1D(%)"],
                "3M(%)": r["3M(%)"],
                "YTD(%)": r["YTD(%)"],
            })
        else:
            rows.append({
                "종목명": row["종목명"], "전일종가": price,
                "1D(%)": row["1D(%)"], "3M(%)": None, "YTD(%)": None,
            })
    return pd.DataFrame(rows)


def format_ticker_label(ticker: str, name: str, html: bool = True) -> str:
    """
    국내 종목은 종목코드를, 해외 종목/지수는 티커를 이름 옆에 작게 표시.
    html=True면 마크다운/HTML용 문자열, False면 plotly 제목 등 plain text용.
    """
    t = ticker.upper()
    if t.endswith(".KS") or t.endswith(".KQ"):
        code = t.split(".")[0]  # 005930.KS -> 005930
    else:
        code = ticker  # 해외 종목/지수는 티커 그대로 (예: AAPL, ^KS11)

    if html:
        return f"{name} <span style='font-size:0.7em; color:#888;'>{code}</span>"
    return f"{name} ({code})"


# ============================================================
# KIS(한국신용평가) 등급공시 웹 스크래핑
# https://www.kisrating.com/ratings/hot_disclosure.do
# ============================================================
KIS_DISCLOSURE_URL = "https://www.kisrating.com/ratings/hot_disclosure.do"
KIS_TAB_ISSUER_RATING = "4"

KIS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": KIS_DISCLOSURE_URL,
}


def fetch_kis_disclosure_html(start_dt, end_dt, tab_type=KIS_TAB_ISSUER_RATING, timeout=15):
    payload = {
        "tabType": tab_type,
        "searchYn": "Y",
        "startDt": start_dt,
        "endDt": end_dt,
    }
    resp = requests.post(KIS_DISCLOSURE_URL, data=payload, headers=KIS_HEADERS, timeout=timeout)
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def parse_kis_issuer_rating(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find(id="issueList")
    if table is None:
        return pd.DataFrame()

    rows_data = []
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 11:
            continue

        chk_input = tds[0].find("input")
        kiscd = chk_input.get("kiscd") if chk_input else None

        company_a = tds[1].find("a")
        company_name = company_a.get_text(strip=True) if company_a else tds[1].get_text(strip=True)

        report_a = tds[11].find("a") if len(tds) > 11 else None
        pdf_file = None
        if report_a and report_a.get("href"):
            m = re.search(r"fn_file\([^)]*'([^']+\.pdf)'", report_a["href"])
            if m:
                pdf_file = m.group(1)

        rows_data.append({
            "회사명": company_name,
            "재무기준일": tds[2].get_text(strip=True),
            "평가종류": tds[3].get_text(strip=True),
            "직전등급": tds[4].get_text(strip=True),
            "직전Outlook": tds[5].get_text(strip=True),
            "현재등급": tds[7].get_text(strip=True),
            "현재Outlook": tds[8].get_text(strip=True),
            "기준통화": tds[9].get_text(strip=True),
            "평가일": tds[10].get_text(strip=True),
            "kiscd": kiscd,
            "리포트파일": pdf_file,
        })

    return pd.DataFrame(rows_data)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_kis_issuer_rating(start_dt, end_dt):
    """start_dt, end_dt: 'YYYY.MM.DD' 형식"""
    try:
        html = fetch_kis_disclosure_html(start_dt, end_dt, tab_type=KIS_TAB_ISSUER_RATING)
        df = parse_kis_issuer_rating(html)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ============================================================
# 한국기업평가(KHR) 등급공시 웹 스크래핑
# https://www.korearatings.com/cms/frCmnCon/index.do?MENU_ID=360
# ============================================================
KHR_DISCLOSURE_AJAX_URL = "https://www.korearatings.com/ajaxf/frDisclosureSvc/getRatingDisclosureList.do"
KHR_SVCTY_ICR = "10"

KHR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.korearatings.com/cms/frCmnCon/index.do?MENU_ID=360",
    "X-Requested-With": "XMLHttpRequest",
}


def fetch_khr_disclosure_json(start_dt, end_dt, svcty_codes=(KHR_SVCTY_ICR,), changed_only=False, timeout=15):
    """start_dt, end_dt: 'YYYY-MM-DD' 형식"""
    payload = {
        "MENU_ID": "360",
        "CONTENTS_NO": "1",
        "SITE_NO": "2",
        "COMP_CD": "",
        "STDT": start_dt,
        "ENDT": end_dt,
        "SVCTY_CD": list(svcty_codes),
    }
    if changed_only:
        payload["CHNG_ONLY_YN"] = "Y"

    resp = requests.post(KHR_DISCLOSURE_AJAX_URL, data=payload, headers=KHR_HEADERS, timeout=timeout)
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.json()


def parse_khr_icr(json_data, data_key="data34"):
    block = json_data.get(data_key)
    if not block:
        return pd.DataFrame()

    rows = block.get("Data", [])
    if not rows:
        return pd.DataFrame()

    GR_CHN_MAP = {"1": "상향", "2": "하향", "3": "관찰(유동적)"}

    out_rows = []
    for r in rows:
        out_rows.append({
            "회사명": r.get("COMP_NM"),
            "구분": r.get("EVAL_DIV_NM"),
            "직전등급": r.get("BFR_GRD") if r.get("BFR_GRD") not in (None, "0") else "",
            "직전Outlook": r.get("BFR_OL_NM") if r.get("BFR_OL_NM") not in (None, "0") else "",
            "현재등급": r.get("GRD") if r.get("GRD") not in (None, "0") else "",
            "현재Outlook": r.get("OL_NM") if r.get("OL_NM") not in (None, "0") else "",
            "평가일": r.get("EVAL_DT"),
            "공시일": r.get("DSCLS_DT"),
            "등급변동": GR_CHN_MAP.get(r.get("GR_CHN_DVCD"), ""),
        })

    return pd.DataFrame(out_rows)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_khr_icr_rating(start_dt, end_dt):
    """start_dt, end_dt: 'YYYY-MM-DD' 형식"""
    debug_info = {}
    try:
        json_data = fetch_khr_disclosure_json(start_dt, end_dt, svcty_codes=(KHR_SVCTY_ICR,))
        debug_info["raw_response"] = json_data
        df = parse_khr_icr(json_data)
        return df, None, debug_info
    except Exception as e:
        debug_info["exception"] = str(e)
        return pd.DataFrame(), str(e), debug_info


# ============================================================
# NICE신용평가 등급공시 웹 스크래핑
# https://www.nicerating.com/disclosure/dayRatingNews.do
# ============================================================
NICE_DISCLOSURE_URL = "https://www.nicerating.com/disclosure/dayRatingNews.do"
NICE_SECUTYP_ICR = "ICR"

NICE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": NICE_DISCLOSURE_URL,
}


def fetch_nice_disclosure_html(today, start_dt, end_dt, secu_typ=NICE_SECUTYP_ICR, timeout=15):
    """today, start_dt, end_dt: 'YYYY-MM-DD' 형식"""
    params = {
        "today": today,
        "cmpCd": "",
        "seriesNm": "",
        "secuTyp": secu_typ,
        "strDate": start_dt,
        "endDate": end_dt,
    }
    resp = requests.get(NICE_DISCLOSURE_URL, params=params, headers=NICE_HEADERS, timeout=timeout)
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def parse_nice_icr(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find(id="tbl4")
    if table is None:
        return pd.DataFrame()

    rows_data = []
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 9:
            continue

        company_a = tds[0].find("a")
        company_name = company_a.get_text(strip=True) if company_a else tds[0].get_text(strip=True)

        rows_data.append({
            "기업명": company_name,
            "구분": tds[1].get_text(strip=True),
            "직전등급": tds[2].get_text(strip=True),
            "직전전망": tds[3].get_text(strip=True),
            "현재등급": tds[4].get_text(strip=True),
            "현재전망": tds[5].get_text(strip=True),
            "등급결정일": tds[6].get_text(strip=True),
            "등급확정일": tds[7].get_text(strip=True),
            "유효기간": tds[8].get_text(strip=True),
        })

    return pd.DataFrame(rows_data)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nice_icr_rating(start_dt, end_dt, today=None):
    """start_dt, end_dt: 'YYYY-MM-DD' 형식"""
    if today is None:
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
    try:
        html = fetch_nice_disclosure_html(today, start_dt, end_dt, secu_typ=NICE_SECUTYP_ICR)
        df = parse_nice_icr(html)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ============================================================
# 탭 1: 글로벌 증시
# ============================================================
GLOBAL_EQUITY = {
    "미국": {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "Dow Jones", "^VIX": "VIX"},
    "유럽": {"^STOXX50E": "EuroStoxx 50", "^GDAXI": "독일 DAX"},
    "아시아·태평양": {"^KS11": "KOSPI", "^KQ11": "KOSDAQ", "000001.SS": "중국 상해종합",
                     "^N225": "일본 Nikkei 225", "^NSEI": "인도 Nifty 50"},
}
M7 = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
    "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla",
}
ETF_LIST = {
    "SPY": "SPDR S&P 500", "QQQ": "Invesco NASDAQ 100",
    "TQQQ": "ProShares UltraPro QQQ 3x", "SOXL": "Direxion Semi Bull 3x",
    "ARKK": "ARK Innovation", "XLK": "Technology Select",
    "XLE": "Energy Select", "XLF": "Financial Select",
    "GLD": "SPDR Gold", "TLT": "iShares 20Y Treasury",
    "IWM": "iShares Russell 2000", "EEM": "iShares Emerging Markets",
}
BONDS_YF = {"^IRX": "미국 2년", "^TNX": "미국 10년"}
FX = {
    "DX-Y.NYB": "달러 지수", "EURUSD=X": "유로/달러",
    "JPY=X": "엔/달러", "KRW=X": "원/달러", "CNY=X": "위안화/달러",
}
COMMODITIES = {
    "CL=F": "WTI (bbl)", "NG=F": "천연가스 (MMBtu)",
    "GC=F": "금 (oz)", "SI=F": "은 (oz)", "HG=F": "구리 (lb)",
}
US_SECTOR_ETF = {
    "XLK": "Technology", "XLF": "Financial Services", "XLC": "Communication",
    "XLY": "Consumer Cyclical", "XLI": "Industrials", "XLV": "Healthcare",
    "XLE": "Energy", "XLP": "Consumer Defensive",
    "XLB": "Basic Materials", "XLRE": "Real Estate", "XLU": "Utilities",
}

PCT_FULL = ["1D(%)", "1W(%)", "1M(%)", "3M(%)", "1Y(%)", "YTD(%)"]
PCT_BASE = ["1D(%)", "1W(%)", "1M(%)", "YTD(%)"]


def render_tab_global():
    st.markdown("### 조회 기준일")
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        use_past_date = st.checkbox("과거 날짜 기준으로 조회", value=False, key="use_past_date")
    as_of_date = None
    if use_past_date:
        with col_d1:
            picked_date = st.date_input("기준일", value=date.today() - timedelta(days=1),
                                          max_value=date.today(), key="global_as_of_date")
            as_of_date = picked_date.strftime("%Y-%m-%d")
        st.caption(f"{as_of_date}을 '오늘'로 보고 1D/1W/1M/3M/1Y/YTD를 다시 계산합니다.")

    st.markdown("## 글로벌 증시")
    with st.spinner("지수 데이터 로딩 중..."):
        eq_rows = []
        for region, tickers in GLOBAL_EQUITY.items():
            for t, n in tickers.items():
                r = calculate_performance(t, n, label_col="지수", as_of_date=as_of_date)
                if r:
                    r["지역"] = region
                    eq_rows.append(r)
    if eq_rows:
        df_eq = pd.DataFrame(eq_rows)[["지역", "지수", "전일종가", "기준일"] + PCT_FULL]
        st.dataframe(style_pct_df(df_eq, PCT_FULL, "전일종가"), use_container_width=True, hide_index=True)
    else:
        st.info("데이터 없음")

    st.markdown("## Magnificent 7 (단위: USD)")
    with st.spinner("M7 데이터 로딩 중..."):
        m7_rows = [r for r in (calculate_performance(t, n, label_col="종목", currency="USD", as_of_date=as_of_date) for t, n in M7.items()) if r]
    if m7_rows:
        df_m7 = pd.DataFrame(m7_rows)[["종목", "전일종가", "기준일", "1D(%)", "1W(%)", "1M(%)", "3M(%)", "YTD(%)"]]
        st.dataframe(style_pct_df(df_m7, ["1D(%)", "1W(%)", "1M(%)", "3M(%)", "YTD(%)"], "전일종가"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("데이터 없음")

    st.markdown("## 미국 섹터별 일간 수익률")
    with st.spinner("섹터 데이터 로딩 중..."):
        sec_rows = []
        for ticker, sector in US_SECTOR_ETF.items():
            try:
                df_s = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
                if isinstance(df_s.columns, pd.MultiIndex):
                    df_s.columns = df_s.columns.get_level_values(0)
                if len(df_s) < 2:
                    continue
                cur = float(df_s["Close"].iloc[-1])
                prev = float(df_s["Close"].iloc[-2])
                sec_rows.append({
                    "Sector": sector, "ETF": ticker,
                    "전일종가(USD)": round(cur, 2),
                    "기준일": df_s.index[-1].strftime("%Y-%m-%d"),
                    "1D(%)": round((cur - prev) / prev * 100, 2),
                })
            except Exception:
                continue
    if sec_rows:
        df_sec = pd.DataFrame(sec_rows).sort_values("1D(%)", ascending=False)
        fig_sec = go.Figure(go.Bar(
            x=df_sec["1D(%)"], y=df_sec["Sector"], orientation="h",
            marker_color=[pct_color(v) for v in df_sec["1D(%)"]],
            text=[f"{v:+.2f}%" for v in df_sec["1D(%)"]],
            textposition="outside",
        ))
        fig_sec.update_layout(
            height=420, margin=dict(l=180, r=60, t=30, b=30),
            xaxis_title="1D (%)", yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        st.info("데이터 없음")

    st.markdown("## 상승 Top 5")

    st.markdown("**미국 ETF**")
    with st.spinner("ETF 로딩 중..."):
        etf_rows = [r for r in (calculate_performance(t, n, label_col="ETF명", currency="USD", as_of_date=as_of_date) for t, n in ETF_LIST.items()) if r]
    if etf_rows:
        df_etf = pd.DataFrame(etf_rows).sort_values("1D(%)", ascending=False).head(5).reset_index(drop=True)
        df_etf = df_etf[["ETF명", "전일종가", "1D(%)", "3M(%)", "YTD(%)"]]
        st.dataframe(style_pct_df(df_etf, ["1D(%)", "3M(%)", "YTD(%)"], "전일종가"),
                     use_container_width=True, hide_index=True)
    else:
        st.info("데이터 없음")

    with st.spinner("미국 종목 로딩 중..."):
        df_us_yahoo = fetch_yahoo_us_top_movers(top_n=5)
        df_us_top = enrich_with_3m_ytd(df_us_yahoo, as_of_date=as_of_date)
    if not df_us_top.empty:
        st.markdown("**미국 개별 종목**")
        st.dataframe(style_pct_df(df_us_top[["종목명", "전일종가", "1D(%)", "3M(%)", "YTD(%)"]],
                                   ["1D(%)", "3M(%)", "YTD(%)"], "전일종가"),
                     use_container_width=True, hide_index=True)
        st.caption(f"미국 상승 Top5 기준일: {date.today().strftime('%Y-%m-%d')}")

    with st.spinner("국내 종목 로딩 중..."):
        df_kospi_naver = fetch_naver_top_movers(market="kospi", top_n=5)
        df_kospi_top = enrich_with_3m_ytd(df_kospi_naver, as_of_date=as_of_date)
        df_kosdaq_naver = fetch_naver_top_movers(market="kosdaq", top_n=5)
        df_kosdaq_top = enrich_with_3m_ytd(df_kosdaq_naver, as_of_date=as_of_date)

    if not df_kospi_top.empty:
        st.markdown("**코스피 개별 종목**")
        st.dataframe(style_pct_df(df_kospi_top[["종목명", "전일종가", "1D(%)", "3M(%)", "YTD(%)"]],
                                   ["1D(%)", "3M(%)", "YTD(%)"], "전일종가"),
                     use_container_width=True, hide_index=True)

    if not df_kosdaq_top.empty:
        st.markdown("**코스닥 개별 종목**")
        st.dataframe(style_pct_df(df_kosdaq_top[["종목명", "전일종가", "1D(%)", "3M(%)", "YTD(%)"]],
                                   ["1D(%)", "3M(%)", "YTD(%)"], "전일종가"),
                     use_container_width=True, hide_index=True)

    if not df_kospi_top.empty or not df_kosdaq_top.empty:
        st.caption(f"코스피/코스닥 상승 Top5 기준일: {date.today().strftime('%Y-%m-%d')}")

    if etf_rows or not df_us_top.empty:
        st.caption("미국 ETF/종목은 시가총액 상위 대표 종목 기준입니다.")

    st.markdown("## 관심종목")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = {}

    with st.expander("관심종목 추가 또는 제거", expanded=False):
        wl_col1, wl_col2, wl_col3 = st.columns([2, 2, 1])
        with wl_col1:
            wl_new_ticker = st.text_input("티커 (예: TSLA, 005930.KS, ^KS11)", key="wl_new_ticker")
        with wl_col2:
            wl_new_name = st.text_input("표시명 (비워두면 자동으로 종목명을 가져와요)", key="wl_new_name")
        with wl_col3:
            st.write("")
            if st.button("추가", key="wl_add_btn"):
                t = wl_new_ticker.strip()
                if t:
                    if wl_new_name.strip():
                        display_name = wl_new_name.strip()
                    else:
                        with st.spinner(f"{t} 종목명 조회 중..."):
                            display_name = fetch_ticker_name(t)
                    st.session_state.watchlist[t] = display_name
                    st.rerun()

        if st.session_state.watchlist:
            st.markdown("**현재 목록**")
            for t, n in list(st.session_state.watchlist.items()):
                wc1, wc2 = st.columns([5, 1])
                wc1.markdown(format_ticker_label(t, n), unsafe_allow_html=True)
                if wc2.button("제거", key=f"wl_remove_{t}"):
                    del st.session_state.watchlist[t]
                    st.rerun()

    if st.session_state.watchlist:
        with st.spinner("관심종목 데이터 로딩 중..."):
            wl_rows = []
            for t, n in st.session_state.watchlist.items():
                r = calculate_performance(t, n, label_col="종목명", as_of_date=as_of_date)
                if r:
                    t_upper = t.upper()
                    r["코드"] = t_upper.split(".")[0] if (t_upper.endswith(".KS") or t_upper.endswith(".KQ")) else t
                    wl_rows.append(r)
        if wl_rows:
            df_wl = pd.DataFrame(wl_rows)[["종목명", "코드", "전일종가", "1D(%)", "3M(%)", "YTD(%)"]]
            st.dataframe(style_pct_df(df_wl, ["1D(%)", "3M(%)", "YTD(%)"], "전일종가"),
                         use_container_width=True, hide_index=True)
        else:
            st.info("관심종목 데이터를 가져오지 못했습니다.")

    st.markdown("## 금리 추세 분석")
    rate_years = st.selectbox("금리 차트 기간(년)", [1, 2, 3, 5, 10], index=3, key="rate_years")

    ref_now = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.now()

    with st.spinner("금리 데이터 로딩 중..."):
        bond_rows = [r for r in (calculate_performance(t, n, label_col="채권", currency="%", as_of_date=as_of_date) for t, n in BONDS_YF.items()) if r]
        df_bond = pd.DataFrame(bond_rows)[["채권", "전일종가", "기준일", "1D(%)", "1W(%)", "1M(%)", "YTD(%)"]] if bond_rows else pd.DataFrame()
        if not df_bond.empty:
            df_bond = df_bond.rename(columns={"전일종가": "금리(%)"})

        ecos_start = (ref_now - pd.DateOffset(years=rate_years)).strftime("%Y%m%d")
        ecos_end = ref_now.strftime("%Y%m%d") if as_of_date else None
        df_kr3y_ecos, ecos_debug_3y = fetch_ecos_market_rate(ECOS_ITEM_CODE_KR3Y, "국고채(3년)", start=ecos_start, end=ecos_end)
        df_kr10y_ecos, ecos_debug_10y = fetch_ecos_market_rate(ECOS_ITEM_CODE_KR10Y, "국고채(10년)", start=ecos_start, end=ecos_end)

        kr_bond_rows = []
        for label, df_k in [("한국 3년", df_kr3y_ecos), ("한국 10년", df_kr10y_ecos)]:
            if df_k.empty:
                continue
            cur = float(df_k["value"].iloc[-1])
            prev = float(df_k["value"].iloc[-2]) if len(df_k) > 1 else cur
            ref_1w = df_k[df_k.index >= ref_now - pd.DateOffset(weeks=1)]
            ref_1m = df_k[df_k.index >= ref_now - pd.DateOffset(months=1)]
            ref_ytd = df_k[df_k.index >= pd.Timestamp(f"{ref_now.year}-01-01")]
            base_1w = float(ref_1w["value"].iloc[0]) if not ref_1w.empty else cur
            base_1m = float(ref_1m["value"].iloc[0]) if not ref_1m.empty else cur
            base_ytd = float(ref_ytd["value"].iloc[0]) if not ref_ytd.empty else cur
            kr_bond_rows.append({
                "채권": label, "금리(%)": round(cur, 3),
                "기준일": df_k.index[-1].strftime("%Y-%m-%d"),
                "1D(%)": round(cur - prev, 3),
                "1W(%)": round(cur - base_1w, 3),
                "1M(%)": round(cur - base_1m, 3),
                "YTD(%)": round(cur - base_ytd, 3),
            })
        if kr_bond_rows:
            df_bond = pd.concat([df_bond, pd.DataFrame(kr_bond_rows)], ignore_index=True)

    if not df_bond.empty:
        st.dataframe(style_pct_df(df_bond, PCT_BASE, "금리(%)"), use_container_width=True, hide_index=True)
    else:
        st.info("데이터 없음")

    if df_kr3y_ecos.empty or df_kr10y_ecos.empty:
        with st.expander("한국 국채 데이터 조회 결과 보기 (문제 진단용)"):
            st.write("국고채 3년:", "조회됨" if not df_kr3y_ecos.empty else "조회 안 됨")
            st.json(ecos_debug_3y)
            st.write("국고채 10년:", "조회됨" if not df_kr10y_ecos.empty else "조회 안 됨")
            st.json(ecos_debug_10y)

    with st.spinner("금리 차트 로딩 중..."):
        df_fedfunds = fetch_fred("FEDFUNDS")
        df_dgs2 = fetch_fred("DGS2")
        df_dgs10 = fetch_fred("DGS10")
        cutoff = ref_now - pd.DateOffset(years=rate_years)

        fig_rate = go.Figure()
        for label, df_s, color in [
            ("연준 기준금리", df_fedfunds, "#EF553B"),
            ("미국 2년 국채", df_dgs2, "#636EFA"),
            ("미국 10년 국채", df_dgs10, "#2471a3"),
        ]:
            if not df_s.empty:
                sl = df_s[df_s.index >= cutoff]
                fig_rate.add_trace(go.Scatter(x=sl.index, y=sl.iloc[:, 0], name=label,
                                               line=dict(color=color, width=1.8)))
        for label, df_k, color in [
            ("한국 3년 국채", df_kr3y_ecos, "#00CC96"),
            ("한국 10년 국채", df_kr10y_ecos, "#FFA15A"),
        ]:
            if not df_k.empty:
                sl = df_k[df_k.index >= cutoff]
                if not sl.empty:
                    fig_rate.add_trace(go.Scatter(
                        x=sl.index, y=sl["value"], name=label,
                        line=dict(color=color, width=1.8, dash="dot"),
                    ))
        fig_rate.add_hline(y=5.0, line=dict(color="red", dash="dot", width=1), annotation_text="5% 경계")
        fig_rate.update_layout(height=420, hovermode="x unified",
                                legend=dict(orientation="h", y=1.1),
                                margin=dict(t=50, b=20), yaxis_title="금리 (%)")
    st.plotly_chart(fig_rate, use_container_width=True)

    st.markdown("## 환율")
    with st.spinner("환율 데이터 로딩 중..."):
        fx_rows = [r for r in (calculate_performance(t, n, label_col="통화명", currency="", as_of_date=as_of_date) for t, n in FX.items()) if r]
    if fx_rows:
        df_fx = pd.DataFrame(fx_rows)[["통화명", "전일종가", "기준일", "1D(%)", "1W(%)", "1M(%)", "YTD(%)"]]
        st.dataframe(style_pct_df(df_fx, PCT_BASE, "전일종가"), use_container_width=True, hide_index=True)
        st.caption("'+' 달러 강세 / '-' 달러 약세")
    else:
        st.info("데이터 없음")

    st.markdown("## 원자재 (단위: USD)")
    with st.spinner("원자재 데이터 로딩 중..."):
        com_rows = [r for r in (calculate_performance(t, n, label_col="원자재", currency="USD", as_of_date=as_of_date) for t, n in COMMODITIES.items()) if r]
    if com_rows:
        df_com = pd.DataFrame(com_rows)[["원자재", "전일종가", "기준일", "1D(%)", "1W(%)", "1M(%)", "YTD(%)"]]
        st.dataframe(style_pct_df(df_com, PCT_BASE, "전일종가"), use_container_width=True, hide_index=True)
    else:
        st.info("데이터 없음")


# ============================================================
# 탭 2: 신용등급 변동
# ============================================================
def render_tab_rating():
    st.markdown("### 조회 기간 설정")
    col1, col2, col3 = st.columns([1, 1, 2])
    today = date.today()
    with col1:
        start_date = st.date_input("시작일", value=today - timedelta(days=7), key="rating_start")
    with col2:
        end_date = st.date_input("종료일", value=today, key="rating_end")
    with col3:
        st.write("")
        st.write("")
        st.button("조회", key="rating_search", type="primary",
                   help="날짜를 바꾸면 자동으로 다시 조회됩니다. 이 버튼은 새로고침용입니다.")

    if start_date > end_date:
        st.error("시작일은 종료일보다 늦을 수 없습니다.")
        return

    start_dot = start_date.strftime("%Y.%m.%d")   # KIS용
    end_dot = end_date.strftime("%Y.%m.%d")
    start_dash = start_date.strftime("%Y-%m-%d")   # KHR, NICE용
    end_dash = end_date.strftime("%Y-%m-%d")
    today_dash = today.strftime("%Y-%m-%d")

    st.markdown(
        f'## 발행사 신용등급 변동 내역 <span style="font-size:0.6em; color:#888; font-weight:normal;">'
        f'({start_dot} ~ {end_dot})</span>',
        unsafe_allow_html=True,
    )

    st.markdown("### 한신평")
    with st.spinner("조회 중..."):
        df_kis, kis_err = fetch_kis_issuer_rating(start_dot, end_dot)
    if kis_err:
        st.markdown(f'<p class="warn-note">조회 실패: {kis_err}</p>', unsafe_allow_html=True)
    elif df_kis.empty:
        st.info("해당 기간 내 변동 내역이 없습니다.")
    else:
        st.dataframe(df_kis.drop(columns=["kiscd", "리포트파일", "기준통화"], errors="ignore"),
                     use_container_width=True, hide_index=True)

    st.markdown("### 한기평")
    with st.spinner("조회 중..."):
        df_khr, khr_err, khr_debug = fetch_khr_icr_rating(start_dash, end_dash)
    if khr_err:
        st.markdown(f'<p class="warn-note">조회 실패: {khr_err}</p>', unsafe_allow_html=True)
    elif df_khr.empty:
        st.info("해당 기간 내 변동 내역이 없습니다.")
    else:
        st.dataframe(df_khr, use_container_width=True, hide_index=True)

    if khr_err or df_khr.empty:
        with st.expander("한기평 조회 결과 보기 (문제 진단용)"):
            st.json(khr_debug)

    st.markdown("### 나신평")
    with st.spinner("조회 중..."):
        df_nice, nice_err = fetch_nice_icr_rating(start_dash, end_dash, today=today_dash)
    if nice_err:
        st.markdown(f'<p class="warn-note">조회 실패: {nice_err}</p>', unsafe_allow_html=True)
    elif df_nice.empty:
        st.info("해당 기간 내 변동 내역이 없습니다.")
    else:
        st.dataframe(df_nice, use_container_width=True, hide_index=True)


# ============================================================
# 탭 3: 닷컴버블 비교
# ============================================================
BUBBLE_TICKERS = {
    "^KS11": "KOSPI (KRW)", "^KQ11": "KOSDAQ (KRW)",
    "^GSPC": "S&P 500 (USD)", "^IXIC": "NASDAQ (USD)",
}
BUBBLE_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#FFA15A"]


def render_tab_bubble():
    st.markdown("### 비교 기간 설정")
    col1, col2, col3 = st.columns(3)
    with col1:
        bubble_start = st.date_input("과거(닷컴버블) 시작일", value=date(1998, 5, 1), key="bubble_start")
    with col2:
        bubble_end = st.date_input("과거(닷컴버블) 종료일", value=date(2000, 6, 1), key="bubble_end")
    with col3:
        current_start = st.date_input("현재 시기 시작일", value=date(2025, 1, 1), key="bubble_current_start")

    # 두 시기를 나란히 비교하기 쉽도록, 현재 구간 길이를 과거 구간과 동일하게 맞춤
    period_length = pd.Timestamp(bubble_end) - pd.Timestamp(bubble_start)
    current_end = pd.Timestamp(current_start) + period_length

    st.markdown("## 닷컴버블 시기 vs 현재 지수 비교")
    st.caption(f"지수를 100으로 정규화 | 두 구간 모두 약 {period_length.days}일로 길이를 맞췄습니다 | 출처: Yahoo Finance")

    with st.spinner("지수 데이터 로딩 중..."):
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=[f"과거 ({bubble_start} ~ {bubble_end})",
                                            f"현재 ({current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')})"],
                            vertical_spacing=0.12)
        excluded_from_past = []
        for idx, (ticker, name) in enumerate(BUBBLE_TICKERS.items()):
            color = BUBBLE_COLORS[idx % len(BUBBLE_COLORS)]
            df_past = fetch_history_range(ticker, bubble_start.strftime("%Y-%m-%d"))
            has_past_data = False
            if not df_past.empty:
                dp = df_past[df_past.index <= pd.Timestamp(bubble_end)]
                # 과거 구간 시작 시점 근처(첫 30일 이내) 데이터가 있어야 정규화 비교가 의미 있음
                if not dp.empty and (dp.index[0] - pd.Timestamp(bubble_start)).days <= 30:
                    base = float(dp["Close"].iloc[0])
                    fig.add_trace(go.Scatter(x=dp.index, y=dp["Close"] / base * 100,
                                              name=name, line=dict(color=color, width=1.5),
                                              legendgroup=name), row=1, col=1)
                    has_past_data = True
            if not has_past_data:
                excluded_from_past.append(name)

            df_cur = fetch_history_range(ticker, current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d"))
            if not df_cur.empty:
                base = float(df_cur["Close"].iloc[0])
                fig.add_trace(go.Scatter(x=df_cur.index, y=df_cur["Close"] / base * 100,
                                          name=f"{name} (현재)", line=dict(color=color, width=1.5, dash="dash"),
                                          legendgroup=name, showlegend=has_past_data is False), row=2, col=1)
        fig.update_layout(height=780, hovermode="x unified",
                           legend=dict(orientation="h", y=-0.08),
                           margin=dict(t=50, b=60))
        fig.update_yaxes(title_text="정규화 지수 (시작=100)", row=1, col=1)
        fig.update_yaxes(title_text="정규화 지수 (시작=100)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    if excluded_from_past:
        st.caption(f"{', '.join(excluded_from_past)}는 해당 과거 시점 데이터가 없어 위쪽 차트에서 제외했습니다.")


# ============================================================
# 탭 4: HP 필터
# ============================================================
DEFAULT_HP_PORTFOLIO = {
    "005930.KS": "삼성전자", "NVDA": "엔비디아",
    "AAPL": "애플", "035420.KS": "NAVER",
    "^KS11": "코스피", "^IXIC": "나스닥",
}


def render_tab_hp():
    st.markdown("### 분석 설정")

    if "hp_portfolio" not in st.session_state:
        st.session_state.hp_portfolio = dict(DEFAULT_HP_PORTFOLIO)

    col1, col2, col3 = st.columns(3)
    with col1:
        hp_lambda = st.selectbox(
            "람다(λ) 값", options=[100, 1600, 6400, 129600], index=1,
            format_func=lambda x: {100: "100 (연간)", 1600: "1,600 (일간 권장)",
                                    6400: "6,400 (주간)", 129600: "129,600 (월간)"}[x],
            key="hp_lambda",
        )
    with col2:
        hp_period = st.selectbox(
            "조회 기간", options=["6mo", "1y", "2y", "3y"], index=1,
            format_func=lambda x: {"6mo": "6개월", "1y": "1년", "2y": "2년", "3y": "3년"}[x],
            key="hp_period",
        )
    with col3:
        st.write("")

    with st.expander("종목/지수 추가 또는 제거", expanded=False):
        add_col1, add_col2, add_col3 = st.columns([2, 2, 1])
        with add_col1:
            new_ticker = st.text_input("티커 (예: ^KS11, TSLA, 005930.KS)", key="new_ticker")
        with add_col2:
            new_name = st.text_input("표시명 (비워두면 자동으로 종목명을 가져와요)", key="new_name")
        with add_col3:
            st.write("")
            if st.button("추가", key="add_ticker_btn"):
                t = new_ticker.strip()
                if t:
                    if new_name.strip():
                        display_name = new_name.strip()
                    else:
                        with st.spinner(f"{t} 종목명 조회 중..."):
                            display_name = fetch_ticker_name(t)
                    st.session_state.hp_portfolio[t] = display_name
                    st.rerun()

        st.markdown("**현재 목록**")
        for t, n in list(st.session_state.hp_portfolio.items()):
            c1, c2 = st.columns([5, 1])
            c1.markdown(format_ticker_label(t, n), unsafe_allow_html=True)
            if c2.button("제거", key=f"remove_{t}"):
                del st.session_state.hp_portfolio[t]
                st.rerun()

    portfolio = st.session_state.hp_portfolio

    st.markdown("## HP 필터 추세 분리 & 매매 시그널")
    st.markdown(
        '<div class="info-note"><b>HP 필터란?</b> 가격을 <b>추세선</b>과 그 위아래로 움직이는 <b>순환(변동)</b>으로 나눠서 봅니다.<br>'
        '현재가가 추세선보다 많이 높으면 과열(매도 고려), 많이 낮으면 과매도(매수 고려)로 판단합니다.</div>',
        unsafe_allow_html=True,
    )

    for ticker, name in portfolio.items():
        with st.spinner(f"{format_ticker_label(ticker, name, html=False)} 분석 중..."):
            hist = fetch_history(ticker, hp_period)
        if hist.empty:
            st.warning(f"{format_ticker_label(ticker, name, html=False)} 데이터 없음")
            continue

        df_hp = compute_hp_signal(hist["Close"], hp_lambda)
        last = df_hp.iloc[-1]

        buy_df = df_hp[df_hp["signal"] == "과매도(매수)"]
        sell_df = df_hp[df_hp["signal"] == "과열(매도)"]

        price_val = float(last["price"])
        trend_val = float(last["trend"])
        cycle_val = float(last["cycle"])
        sigma_val = float(last["sigma"])

        gap_pct = (price_val - trend_val) / trend_val * 100  # 추세선 대비 현재가 위치(%)
        cycle_ratio = cycle_val / sigma_val if sigma_val else 0  # 기준선(±1σ) 대비 현재 순환 강도(배수)

        if last["signal"] == "과열(매도)":
            summary = f"현재가가 추세선보다 {gap_pct:+.1f}% 높아요. 기준선(±1σ) 대비 {abs(cycle_ratio):.1f}배 벗어난 과열 구간입니다."
        elif last["signal"] == "과매도(매수)":
            summary = f"현재가가 추세선보다 {gap_pct:+.1f}% 낮아요. 기준선(±1σ) 대비 {abs(cycle_ratio):.1f}배 벗어난 과매도 구간입니다."
        else:
            summary = f"현재가가 추세선과 비교해 {gap_pct:+.1f}% 차이로, 기준선(±1σ) 안쪽의 중립 구간입니다."

        st.markdown(
            f"### {format_ticker_label(ticker, name)} <span class='signal-badge'>{last['signal']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<p class='caption-note'>{summary}</p>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("현재가", f"{price_val:,.2f}")
        m2.metric("추세선 대비", f"{gap_pct:+.1f}%", help="현재가가 HP 추세선보다 몇 % 위/아래에 있는지")
        m3.metric("과열·과매도 정도", f"{cycle_ratio:+.1f}배", help="기준선(±1σ)을 1배로 볼 때 현재 순환(Cycle)의 강도")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.65, 0.35],
                             subplot_titles=[f"{format_ticker_label(ticker, name, html=False)} 종가 & HP 추세선",
                                             "순환 성분 (Cycle) | 점선: ±1σ"],
                             vertical_spacing=0.08)
        fig.add_trace(go.Scatter(x=df_hp.index, y=df_hp["price"], name="종가",
                                  line=dict(color="#636EFA", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_hp.index, y=df_hp["trend"], name="HP 추세",
                                  line=dict(color="#EF553B", width=2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=buy_df.index, y=buy_df["price"], mode="markers",
                                  name="매수 시그널",
                                  marker=dict(color="#00CC96", size=8, symbol="triangle-up")), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_df.index, y=sell_df["price"], mode="markers",
                                  name="매도 시그널",
                                  marker=dict(color="#EF553B", size=8, symbol="triangle-down")), row=1, col=1)
        fig.add_trace(go.Bar(x=df_hp.index, y=df_hp["cycle"], name="Cycle",
                              marker_color=np.where(df_hp["cycle"] > 0, "#EF553B", "#636EFA")), row=2, col=1)
        fig.add_hline(y=sigma_val, line=dict(color="red", dash="dot", width=1.2), row=2, col=1)
        fig.add_hline(y=-sigma_val, line=dict(color="blue", dash="dot", width=1.2), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="gray", width=0.5), row=2, col=1)
        fig.update_layout(height=520, hovermode="x unified",
                           legend=dict(orientation="h", y=1.02),
                           margin=dict(t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")


# ============================================================
# 메인 페이지
# ============================================================
st.title("Market Morning Call")
st.caption(f"조회일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  데이터: Yahoo Finance / FRED / ECOS / KIS / KHR / NICE")

tab_names = ["글로벌 증시", "신용등급 변동", "닷컴버블 비교", "HP 필터 분석"]
tab1, tab2, tab3, tab4 = st.tabs(tab_names)

with tab1:
    render_tab_global()

with tab2:
    render_tab_rating()

with tab3:
    render_tab_bubble()

with tab4:
    render_tab_hp()

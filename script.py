import time
import urllib.parse
import re
from datetime import datetime, timedelta
import os
import sys
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import kss

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

import openai
import pandas as pd
import numpy as np
import pandas_ta_classic as pdt
from pykrx import stock

OPENAI_API_KEY = ""

THEME_DB = defaultdict(list)
STOCK_TO_THEMES_DB = defaultdict(list)
TICKER_NAME_CACHE = {}
NAME_TICKER_CACHE = {}


def load_theme_db(csv_file="naver_themes_bs4.csv"):
    global THEME_DB, STOCK_TO_THEMES_DB
    try:
        df = pd.read_csv(csv_file, header=0)
        print(f"--- 테마 DB 로드 중: {csv_file} ---")
        theme_col = df.columns[0]

        for index, row in df.iterrows():
            theme_name = row[theme_col]
            if pd.isna(theme_name):
                continue
            stock_names = [name for name in row[1:].dropna() if name.strip()]
            THEME_DB[theme_name] = stock_names
            for name in stock_names:
                STOCK_TO_THEMES_DB[name].append(theme_name)
        print(f"✅ 테마 DB 로드 완료: 총 {len(THEME_DB)}개 테마, {len(STOCK_TO_THEMES_DB)}개 종목 매핑")

    except FileNotFoundError:
        print(f"❌ 치명적 오류: 테마 DB 파일 '{csv_file}'을(를) 찾을 수 없습니다.")
        sys.exit()
    except Exception as e:
        print(f"❌ 치명적 오류: 테마 DB 로드 중 오류 발생: {e}")
        sys.exit()


def load_sentiment_model():
    print("--- KR-FinBert-SC 모델 로드 중 (최초 실행 시 시간이 소요될 수 있습니다) ---")
    try:
        model_name = "snunlp/KR-FinBert-SC"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        device_name = next(model.parameters()).device
        print(f"✅ KR-FinBert-SC 모델 로드 완료. (실행 장치: {device_name})")

        return tokenizer, model
    except Exception as e:
        print(f"❌ 치명적 오류: KR-FinBert-SC 모델 로드 실패: {e}")
        sys.exit()


def calculate_finbert_sentiment(news_text, keywords, tokenizer, model):
    if not news_text or not keywords:
        return np.nan, []

    text = news_text
    text = re.sub(r'(Copyright © .*? All rights reserved\.?|무단 전재.*?금지\.?|© .*? 무단 전재.*?)', '', text, flags=re.DOTALL)
    text = re.sub(r'\[.*? 제공\.?.*?\]', '', text)
    text = re.sub(r'\([가-힣\w]+=[가-힣\w]+\)', '', text)
    text = re.sub(r'\[.{1,15}\]$', '', text)
    text = re.sub(r'▶ .*? 바로가기|▶ .*? 구독하기|\s{0,}ⓒ .*? 금지', '', text)
    text = re.sub(r'[가-힣\w]+ 기자 \(.*?@.*?\..*?\)', '', text)
    text = re.sub(r'\([가-힣\w]+=[가-힣\w]+\) .*? 기자[)]?', '', text)
    text = re.sub(r'---\s?\[기사 \d+\]\s?제목:', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\[ⓒ.*?\]', '', text)
    text = re.sub(r'<저작권자>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    try:
        sentences = kss.split_sentences(text)
    except Exception as e:
        print(f"⚠️ KSS 문장 분리 오류: {e}")
        return np.nan, []

    filtered_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) < 20:
            continue
        if any(kw in s_clean for kw in keywords if kw):
            filtered_sentences.append(s_clean)

    if not filtered_sentences:
        return np.nan, []

    try:
        label2id = model.config.label2id
        pos_id = label2id.get('positive', 2)
        neg_id = label2id.get('negative', 0)

        inputs = tokenizer(
            filtered_sentences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        try:
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        pos_probs = probs[:, pos_id].tolist()
        neg_probs = probs[:, neg_id].tolist()

        sentence_scores = []
        total_net_score = 0.0

        for i in range(len(filtered_sentences)):
            individual_score = pos_probs[i] - neg_probs[i]
            sentence_scores.append((filtered_sentences[i], individual_score))
            total_net_score += individual_score

    except Exception as e:
        print(f"❌ FinBert 배치 분석 중 오류: {e}")
        return np.nan, []

    total_analyzed = len(sentence_scores)
    if total_analyzed == 0:
        return np.nan, []

    avg_score = total_net_score / total_analyzed

    return avg_score, sentence_scores


def parse_openai_theme_output(text, candidate_themes):
    try:
        cleaned_text = text.strip().replace("'", "").replace('"', "").replace(",", "")
        found_theme = None
        for theme in candidate_themes:
            if theme in cleaned_text:
                found_theme = theme
                break

        if not found_theme:
            first_line = cleaned_text.split('\n')[0].strip()
            if first_line in candidate_themes:
                found_theme = first_line
            else:
                parts = first_line.split()
                if parts and parts[0] in candidate_themes:
                    found_theme = parts[0]
                else:
                    print(f"⚠️ OpenAI 테마 파싱 경고: 응답에서 유효한 테마를 찾지 못함. '{text}'")
                    return None, np.nan

        remaining_text = cleaned_text.replace(found_theme, "").strip()
        match = re.search(r'[-+]?\d*\.?\d+', remaining_text)

        if match:
            try:
                scale_value = float(match.group(0))
                return found_theme, scale_value
            except ValueError:
                return found_theme, np.nan
        else:
            return found_theme, np.nan

    except Exception as e:
        print(f"⚠️ OpenAI 테마/규모 파싱 중 오류: {e}")
        return None, np.nan


def call_openai_for_theme(news_content_text, leader_name, candidate_themes):
    max_text_length = 20000
    if len(news_content_text) > max_text_length:
        print(f"⚠️ OpenAI 입력 텍스트가 너무 깁니다. {max_text_length}자로 제한합니다.")
        news_content_text = news_content_text[:max_text_length]

    if OPENAI_API_KEY == "":
        print("❌ OpenAI API 키가 설정되지 않았습니다.")
        return None, np.nan
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"❌ OpenAI API 설정 오류: {e}")
        return None, np.nan

    system_instruction_text = (
        "You are a stock market theme analyzer. You will receive up to 10 news headlines, a lead stock ('주도주'), and a list of candidate themes ('후보 테마 리스트').\n\n"
        "Your tasks are:\n"
        "1. Select the **single most relevant theme** from the candidate list that the lead stock belongs to, based on the news headlines.\n"
        "2. Scan the headlines for any positive or negative financial figures related to the scale of transactions, contracts, sales, or investments.\n"
        "3. If multiple figures exist, select the one that *best* matches the theme you chose.\n"
        "4. You **must** convert this financial figure into units of **10 million KRW (ten million Korean Won)**.\n"
        "5. If the figure is negative (e.g., a loss, deficit, or fine), you **must** prepend a minus sign (-) to the number.\n\n"
        "Respond *only* with the theme name, followed by the number (if found). Do not add any other explanation.\n\n"
        "Example (Positive): A headline states 'Samsung achieved a 5 billion KRW contract.' (5,000,000,000 / 10,000,000 = 500). Your output: 반도체 500\n"
        "Example (Negative): A headline states 'Company posts 300 million KRW operating loss.' (300,000,000 / 10,000,000 = 30). Your output: 실적부진 -30\n"
        "Example (No Number): If no relevant financial figure is found, respond *only* with the theme name. Your output: AI"
    )

    themes_as_string = ", ".join(candidate_themes)

    user_prompt = (
        f"Lead Stock: {leader_name}\n\n"
        f"--- News Headlines (up to 10) ---\n{news_content_text}\n\n"
        f"--- Candidate Themes ---\n[{themes_as_string}]\n\n"
        f"Select the single best theme and the corresponding financial figure (in 10M KRW units)."
    )

    print("\n--- OpenAI API 호출 중 (gpt-5-nano) ---")
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": system_instruction_text},
                {"role": "user", "content": user_prompt}
            ],
        )

        text_output = response.choices[0].message.content
        return parse_openai_theme_output(text_output, candidate_themes)

    except Exception as e:
        print(f"❌ OpenAI API 오류 발생: {e}")
        return None, np.nan


def get_ticker_by_name(name, base_date):
    global NAME_TICKER_CACHE

    cache_key = (name, base_date)
    if cache_key in NAME_TICKER_CACHE:
        return NAME_TICKER_CACHE[cache_key]

    try:
        tickers_df = stock.get_market_ticker_list(base_date, market="ALL")

        if not TICKER_NAME_CACHE.get(base_date):
            print(f"--- {base_date} 기준 Ticker-Name 맵 캐싱 중... (시간 소요) ---")
            TICKER_NAME_CACHE[base_date] = {
                ticker: stock.get_market_ticker_name(ticker) for ticker in tickers_df
            }

        names_series = pd.Series(TICKER_NAME_CACHE[base_date])
        ticker = names_series[names_series == name].index[0]

        NAME_TICKER_CACHE[cache_key] = ticker
        return ticker
    except IndexError:
        NAME_TICKER_CACHE[cache_key] = None
        return None
    except Exception as e:
        print(f"❌ pykrx Ticker 조회 오류: {name}, {e}")
        return None


def get_name_by_ticker(ticker, base_date):
    global TICKER_NAME_CACHE

    if base_date not in TICKER_NAME_CACHE:
        tickers_df = stock.get_market_ticker_list(base_date, market="ALL")
        TICKER_NAME_CACHE[base_date] = {
            ticker: stock.get_market_ticker_name(ticker) for ticker in tickers_df
        }
    return TICKER_NAME_CACHE[base_date].get(ticker, None)


def get_followers(theme, leader_name):
    if theme not in THEME_DB:
        return []
    followers = [name for name in THEME_DB[theme] if name != leader_name]
    return followers


def calculate_follower_features(ticker, end_date_ymd):
    if not ticker:
        return None, None, None, np.nan
    try:
        end_date = datetime.strptime(end_date_ymd, '%Y%m%d')
        start_date = (end_date - timedelta(days=100)).strftime('%Y%m%d')
        df = stock.get_market_ohlcv(start_date, end_date_ymd, ticker)
        if df.empty:
            return None, None, None, np.nan

        t_day_timestamp = pd.to_datetime(end_date_ymd)
        if t_day_timestamp not in df.index:
            t_day_timestamp = df.index[-1]
        actual_date_ymd = t_day_timestamp.strftime('%Y%m%d')
        today_data_for_close = df.loc[t_day_timestamp]
        t_day_close = today_data_for_close.get('종가', np.nan)
    except Exception as e:
        return None, None, None, np.nan

    tech_features = {}
    try:
        df['RVOL5'] = df['거래량'] / df['거래량'].rolling(window=5, min_periods=1).mean()
        rsi_series = pdt.rsi(df['종가'], length=9)
        macd_df = pdt.macd(df['종가'], fast=12, slow=26, signal=15)
        sma_series = pdt.sma(df['종가'], length=5)
        df['RSI_9'] = rsi_series
        df = pd.concat([df, macd_df], axis=1)
        df['MA5'] = sma_series
        df['Disparity5'] = (df['종가'] / df['MA5']) * 100
        today_data = df.loc[t_day_timestamp]
        tech_features = {
            '후보주_RSI_9': today_data.get('RSI_9', np.nan),
            '후보주_MACDs_12_26_15': today_data.get('MACDs_12_26_15', np.nan),
            '후보주_이격도_5': today_data.get('Disparity5', np.nan),
            '후보주_RVOL_5': today_data.get('RVOL5', np.nan)
        }
    except Exception as e:
        pass

    funda_features = {}
    try:
        f_df = stock.get_market_fundamental(actual_date_ymd, actual_date_ymd, ticker)
        if not f_df.empty:
            today_funda = f_df.iloc[0]
            funda_features = {
                '후보주_DIV': today_funda.get('DIV', np.nan),
                '후보주_BPS': today_funda.get('BPS', np.nan),
                '후보주_PER': today_funda.get('PER', np.nan),
                '후보주_EPS': today_funda.get('EPS', np.nan),
                '후보주_PBR': today_funda.get('PBR', np.nan),
            }
    except Exception as e:
        pass

    return tech_features, funda_features, t_day_timestamp, t_day_close


def get_next_day_label(ticker, t_day_timestamp, t_day_close):
    if pd.isna(t_day_close) or t_day_timestamp is None:
        return np.nan
    try:
        start_search_date = (t_day_timestamp + timedelta(days=1)).strftime('%Y%m%d')
        end_search_date = (t_day_timestamp + timedelta(days=10)).strftime('%Y%m%d')
        df_future = stock.get_market_ohlcv(start_search_date, end_search_date, ticker)
        if df_future.empty:
            return np.nan
        t_plus_1_close = df_future.iloc[0]['종가']

        # t_plus_1_close > t_day_close (상승=1)
        return 1 if t_plus_1_close > t_day_close else 0

    except Exception as e:
        return np.nan


def get_correlation(leader_ticker, follower_ticker, end_date_ymd, window=7):
    try:
        end_date = datetime.strptime(end_date_ymd, '%Y%m%d')
        start_date = (end_date - timedelta(days=window + 3)).strftime('%Y%m%d')
        end_date_minus_1 = (end_date - timedelta(days=1)).strftime('%Y%m%d')

        df_leader = stock.get_market_ohlcv(start_date, end_date_minus_1, leader_ticker)['종가']
        df_follower = stock.get_market_ohlcv(start_date, end_date_minus_1, follower_ticker)['종가']
        if df_leader.empty or df_follower.empty:
            return np.nan
        leader_ret = df_leader.pct_change().dropna()
        follower_ret = df_follower.pct_change().dropna()
        corr_df = pd.concat([leader_ret, follower_ret], axis=1, join='inner').dropna()
        if len(corr_df) < 3:
            return np.nan
        correlation = corr_df.iloc[:, 0].corr(corr_df.iloc[:, 1])
        return correlation
    except Exception as e:
        return np.nan


def get_theme_overlap(follower_name):
    if follower_name not in STOCK_TO_THEMES_DB:
        return 0, ""
    themes = STOCK_TO_THEMES_DB[follower_name]
    theme_count = len(themes)
    theme_list_str = ",".join(themes)
    return theme_count, theme_list_str


def calculate_leader_features(ticker, end_date_ymd):
    try:
        end_date = datetime.strptime(end_date_ymd, '%Y%m%d')
        start_date = (end_date - timedelta(days=100))
        start_date_ymd = start_date.strftime('%Y%m%d')

        df = stock.get_market_ohlcv(start_date_ymd, end_date_ymd, ticker)
        if df.empty:
            print(f"❌ pykrx 오류: {ticker}의 가격 정보를 T일({end_date_ymd}) 기준으로 가져올 수 없습니다.")
            return None

        df['RVOL5'] = df['거래량'] / df['거래량'].rolling(window=5, min_periods=1).mean()
        rsi_series = pdt.rsi(df['종가'], length=9)
        macd_df = pdt.macd(df['종가'], fast=12, slow=26, signal=15)
        sma_series = pdt.sma(df['종가'], length=5)
        df['RSI_9'] = rsi_series
        df = pd.concat([df, macd_df], axis=1)
        df['MA5'] = sma_series
        df['Disparity5'] = (df['종가'] / df['MA5']) * 100
        df['Change_Pct'] = df['종가'].pct_change() * 100

        t_day_timestamp = pd.to_datetime(end_date_ymd)
        if t_day_timestamp not in df.index:
            print(f"⚠️ 경고: {end_date_ymd}은(는) 휴장일입니다. 직전 거래일 데이터로 대체합니다.")
            t_day_timestamp = df.index[-1]

        today_data = df.loc[t_day_timestamp]
        actual_date_ymd = t_day_timestamp.strftime('%Y%m%d')
        cap_df = stock.get_market_cap(actual_date_ymd, actual_date_ymd, ticker)
        market_cap = 0
        if not cap_df.empty:
            market_cap = cap_df.iloc[0]['시가총액']

        leader_features = {
            '주도주_상승률': today_data.get('Change_Pct', 0.0),
            '주도주_시가총액': market_cap,
            '주도주_RSI_9': today_data.get('RSI_9', np.nan),
            '주도주_MACDs_12_26_15': today_data.get('MACDs_12_26_15', np.nan),
            '주도주_이격도_5': today_data.get('Disparity5', np.nan),
            '주도주_RVOL_5': today_data.get('RVOL5', np.nan)
        }
        return leader_features

    except KeyError:
        return None
    except Exception as e:
        print(f"❌ 주도주 피처 계산 중 치명적 오류: {e}")
        return None


def save_news_to_file(keyword, start_ymd, end_ymd, news_results):
    output_dir = "news_scraped"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{keyword}_{start_ymd}_{end_ymd}.txt")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"--- 네이버 뉴스 스크래핑 결과 ---\n")
            f.write(f"키워드: {keyword}\n")
            f.write(f"기간: {start_ymd} ~ {end_ymd}\n")
            f.write(f"추출 개수: {len(news_results)}개\n\n")
            f.write("-" * 30 + "\n")
            for news in news_results:
                f.write(f"제목: {news['title']}\n")
                f.write(f"링크: {news.get('url', 'N/A')}\n")
                f.write(f"내용: {news['summary']}\n")
                f.write("\n")
        print(f"✅ 결과 파일 저장 완료: {filename}")
        return filename
    except Exception as e:
        print(f"❌ 파일 저장 중 오류 발생: {e}")
        return None


def save_sentiment_to_csv(keyword, follower_name, start_ymd, end_ymd, sentence_score_list, avg_score):
    if not sentence_score_list:
        return None

    output_dir = "sentiment_analyze"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{keyword}_to_{follower_name}_{start_ymd}_{end_ymd}.csv")

    try:
        data_for_df = [{'문장': sent, '감성점수': score} for sent, score in sentence_score_list]
        df = pd.DataFrame(data_for_df)

        avg_row = pd.DataFrame([{'문장': '--- 최종 평균 ---', '감성점수': avg_score}])

        final_df = pd.concat([df, avg_row], ignore_index=True)

        final_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ FinBert 분석 결과 저장: {filename}")
        return filename
    except Exception as e:
        print(f"❌ FinBert 파일 저장 중 오류 발생: {e}")
        return None


def initialize_driver():
    try:
        options = ChromeOptions()
        options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        options.add_argument("lang=ko_KR")
        options.add_argument('--headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("start-maximized")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        driver.implicitly_wait(0)

        return driver
    except Exception as e:
        print(f"❌ 드라이버 초기화 실패: {e}")
        return None


def read_keywords_from_file(filename):
    data = []
    try:
        df = pd.read_csv(filename, encoding='utf-8')
        if df.empty:
            print(f"⚠️ 입력 파일 '{filename}'이 비어있습니다.")
            return []
        if len(df.columns) < 2:
            print(f"❌ 오류: CSV 파일에 최소 2개의 열('종목명', '날짜')이 필요합니다.")
            return None

        col_keyword = df.columns[0]
        col_date = df.columns[1]
        print(f"--- 입력 CSV의 '{col_keyword}'(종목)과 '{col_date}'(날짜) 컬럼에서 데이터를 읽습니다. ---")

        for index, row in df.iterrows():
            keyword = row[col_keyword]
            date_val = row[col_date]
            if pd.isna(keyword) or pd.isna(date_val):
                continue
            keyword_str = str(keyword).strip()
            date_str = str(int(float(date_val))).strip()
            if keyword_str and re.fullmatch(r'\d{8}', date_str):
                data.append((keyword_str, date_str))
            elif keyword_str or date_str:
                print(f"⚠️ {index + 2}번째 행 스킵: 형식이 '종목명', 'YYYYMMDD'와 맞지 않습니다. (값: '{keyword_str}', '{date_str}')")
        return data
    except FileNotFoundError:
        print(f"❌ 오류: 지정된 파일 '{filename}'을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"❌ CSV 파일 읽기 중 오류 발생: {e}")
        return None


def scrape_naver_news_only_full_xpath(driver, keyword, start_date_ymd, end_date_ymd, max_count=10):
    links_to_visit = []
    results = []

    try:
        def format_date(ymd):
            return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"

        start_date = format_date(start_date_ymd)
        end_date = format_date(end_date_ymd)
        encoded_keyword = urllib.parse.quote(keyword)
        start_date_dots = start_date.replace('-', '.')
        end_date_dots = end_date.replace('-', '.')
        url = (
            f"https://search.naver.com/search.naver?where=news"
            f"&query={encoded_keyword}"
            f"&sm=tab_opt&sort=0&photo=0&field=0&pd=3"
            f"&ds={start_date_dots}&de={end_date_dots}"
            f"&nso=so:r,p:from{start_date_ymd}to{end_date_ymd},a:all"
        )
        print(f"--- 1/2. 링크 수집 시작: {url} ---")
        driver.get(url)

        wait = WebDriverWait(driver, 5)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "group_news")))

        links_to_visit_set = set()
        last_link_count = 0
        scroll_attempts = 0

        while True:
            link_selector = "a[href*='n.news.naver.com/mnews/article']"
            news_link_elements = driver.find_elements(By.CSS_SELECTOR, link_selector)

            for el in news_link_elements:
                try:
                    href = el.get_attribute('href')
                    search_title = el.text

                    if href and search_title and (search_title, href) not in links_to_visit_set:
                        links_to_visit_set.add((search_title, href))
                except Exception:
                    pass

            if max_count is not None and max_count > 0 and len(links_to_visit_set) >= max_count:
                print(f"--- {max_count}개 수집 완료. 스크롤 중단. ---")
                break

            current_link_count = len(links_to_visit_set)
            if current_link_count == last_link_count:
                scroll_attempts += 1
            else:
                scroll_attempts = 0

            if scroll_attempts >= 3:
                print("--- 페이지 끝에 도달. 스크롤 중단. ---")
                break

            last_link_count = current_link_count

            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(0.8)

        links_to_visit = [{"search_title": title, "url": url} for title, url in links_to_visit_set]

        if max_count is not None and max_count > 0 and len(links_to_visit) > max_count:
            links_to_visit = links_to_visit[:max_count]

        print(f"--- 1/2. 링크 수집 완료: 총 {len(links_to_visit)}건 ---")

        wait = WebDriverWait(driver, 5)
        print(f"--- 2/2. 수집된 링크 {len(links_to_visit)}개 본문/제목 스크래핑 시작 ---")

        for item in links_to_visit:
            try:
                driver.get(item['url'])
                body_text = ""
                article_title = ""

                user_title_xpath = "/html/body/div/div[2]/div/div[1]/div[1]/div[1]/div[2]/h2/span"
                common_title_class = "media_end_head_headline"

                try:
                    title_element = wait.until(EC.presence_of_element_located(
                        (By.XPATH, user_title_xpath)
                    ))
                    article_title = title_element.text
                except (TimeoutException, NoSuchElementException):
                    try:
                        title_element = driver.find_element(By.CLASS_NAME, common_title_class)
                        article_title = title_element.text
                    except NoSuchElementException:
                        print(f"⚠️ 본문 제목을 찾지 못했습니다. 검색 결과 제목을 대신 사용합니다: {item['url']}")
                        article_title = item['search_title']

                user_body_xpath = "/html/body/div/div[2]/div/div[1]/div[1]/div[2]/div[1]/article"
                common_id_1 = "dic_area"
                common_id_2 = "contents"

                try:
                    body_element = wait.until(EC.presence_of_element_located(
                        (By.XPATH, user_body_xpath)
                    ))
                    body_text = body_element.text
                except (TimeoutException, NoSuchElementException):
                    try:
                        body_element = driver.find_element(By.ID, common_id_1)
                        body_text = body_element.text
                    except NoSuchElementException:
                        try:
                            body_element = driver.find_element(By.ID, common_id_2)
                            body_text = body_element.text
                        except NoSuchElementException:
                            print(f"⚠️ 본문을 찾지 못했습니다 (XPath/ID 불일치): {item['url']}")
                            body_text = ""

                results.append({
                    "title": article_title,
                    "summary": body_text,
                    "url": item['url']
                })
                time.sleep(0.3)

            except Exception as e:
                print(f"❌ 기사 페이지({item['url']}) 처리 중 오류: {e}")
                continue

    except Exception as e:
        print(f"❌ 스크래핑 작업 중 예외 발생: {e}")

    print(f"--- 스크래핑 완료: 총 {len(results)}건 본문 추출 ---")

    return results


def save_intermediate_dataset(all_data_rows, output_filename):
    if not all_data_rows:
        print("--- 중간 저장: 데이터가 없어 건너뜁니다. ---")
        return

    try:
        print(f"\n--- 중간 저장 시작 (총 {len(all_data_rows)} 행) -> {output_filename} ---")

        final_df = pd.DataFrame(all_data_rows)

        theme_dummies = None
        if '테마_중복_목록' in final_df.columns:
            all_theme_names = list(THEME_DB.keys())
            theme_dummies = final_df['테마_중복_목록'].str.get_dummies(sep=',')
            theme_dummies = theme_dummies.reindex(columns=all_theme_names, fill_value=0)
            theme_dummies = theme_dummies.add_prefix('테마_')
            final_df = pd.concat([final_df, theme_dummies], axis=1)

        columns_order = [
            '주도주', '주도주_Ticker', 'T_Date', '테마', '뉴스기사_규모적지수', '뉴스기사_감성지수',
            '뉴스_파일_경로', '주도주_상승률', '주도주_시가총액', '주도주_RSI_9', '주도주_MACDs_12_26_15',
            '주도주_이격도_5', '주도주_RVOL_5', '후보주', '후보주_Ticker', '후보주_RSI_9',
            '후보주_MACDs_12_26_15', '후보주_이격도_5', '후보주_RVOL_5', '후보주_DIV', '후보주_BPS',
            '후보주_PER', '후보주_EPS', '후보주_PBR',
            '상관관계_지수', '테마_중복_개수', 'Target'
        ]

        corrected_columns_order = []
        for col in columns_order:
            if col not in final_df.columns:
                if 'MACDs_12_26_15' in col:
                    alt_col = col.replace('MACDs_12_26_15', 'MACD_15s')
                    if alt_col in final_df.columns:
                        corrected_columns_order.append(alt_col)
                    else:
                        corrected_columns_order.append(col)
                elif 'MACD_15s' in col:
                    alt_col = col.replace('MACD_15s', 'MACDs_12_26_15')
                    if alt_col in final_df.columns:
                        corrected_columns_order.append(alt_col)
                    else:
                        corrected_columns_order.append(col)
                else:
                    corrected_columns_order.append(col)
            else:
                corrected_columns_order.append(col)
        columns_order = corrected_columns_order

        final_columns = columns_order.copy()
        if theme_dummies is not None:
            for theme_col in sorted(theme_dummies.columns):
                if theme_col not in final_columns:
                    final_columns.append(theme_col)

        for col in final_columns:
            if col not in final_df.columns:
                final_df[col] = np.nan

        final_df = final_df[final_columns]

        if '테마_중복_목록' in final_df.columns:
            final_df = final_df.drop(columns=['테마_중복_목록'])

        final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

        print(f"✅ 중간 저장 완료: {output_filename} (총 {len(final_df)} 행)")

    except Exception as e:
        print(f"❌ 치명적 오류: 중간 저장 실패: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 40)
    print("  ⭐ 주도주-후보주 데이터셋 구축 시작 (하이브리드) ⭐")
    print("=" * 40)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 사용 가능 여부: {cuda_available}")
    if cuda_available:
        print(f"현재 GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA 버전: {torch.version.cuda}")

    final_output_dir = "leader_follower_datasets"
    os.makedirs(final_output_dir, exist_ok=True)

    final_output_filename = os.path.join(final_output_dir, "final_dataset_combined.csv")
    log_file_path = os.path.join(final_output_dir, "processed_log.txt")
    print(f"--- 최종 데이터셋: '{final_output_filename}' ---")
    print(f"--- 진행 상황 로그: '{log_file_path}' ---")

    load_theme_db("naver_themes_bs4.csv")
    if not THEME_DB:
        print("❌ 테마 DB가 비어있습니다. 'naver_themes_bs4.csv' 파일이 있는지 확인하세요.")
        sys.exit()

    tokenizer, model = load_sentiment_model()

    input_filename = input("스크래핑 목록 CSV 파일명 입력 (예: input_list.csv): ")
    keyword_date_list = read_keywords_from_file(input_filename)
    if not keyword_date_list:
        print("❌ 처리할 데이터가 없습니다. 프로그램을 종료합니다.")
        sys.exit()

    all_events_rows = []
    processed_events = set()

    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    processed_events.add(line.strip())
            print(f"--- [이어하기] {len(processed_events)}개의 이벤트를 로그에서 확인했습니다. ---")
        except Exception as e:
            print(f"⚠️ 로그 파일({log_file_path}) 읽기 오류: {e}. 새로 시작합니다.")
            processed_events = set()

    if os.path.exists(final_output_filename) and processed_events:
        print(f"--- [이어하기] '{final_output_filename}'에서 기존 데이터를 로드합니다... ---")
        try:
            existing_df = pd.read_csv(final_output_filename)
            cols_to_load = [col for col in existing_df.columns if not col.startswith('테마_')]
            if '테마_중복_목록' not in cols_to_load and '테마_중복_개수' in cols_to_load:
                cols_to_load.insert(cols_to_load.index('테마_중복_개수') + 1, '테마_중복_목록')

            for base_col in ['주도주', 'T_Date', '후보주', '테마_중복_목록']:
                if base_col not in existing_df.columns:
                    existing_df[base_col] = np.nan

            base_df = existing_df[cols_to_load]
            all_events_rows = base_df.to_dict('records')
            print(f"✅ {len(all_events_rows)}개의 기존 행이 로드되었습니다. ---")
        except Exception as e:
            print(f"⚠️ 경고: 기존 데이터셋 로드 실패: {e}. 새로 시작합니다.")
            all_events_rows = []
            processed_events = set()
    else:
        if processed_events:
            print(f"--- [경고] 로그 파일은 있으나 CSV 파일이 없습니다. 새로 시작합니다. ---")
            processed_events = set()

    print("\n--- 1회용 웹 드라이버 초기화 중... (종료 시까지 재사용) ---")
    driver = initialize_driver()
    if driver is None:
        print("❌ 치명적 오류: 웹 드라이버 초기화 실패. 프로그램을 종료합니다.")
        sys.exit()

    print(f"\n--- 총 {len(keyword_date_list)}개의 주도주 이벤트를 순차적으로 시작합니다 ---")

    total_start_time = time.time()
    events_processed_count = 0

    try:
        for keyword, end_date_ymd in keyword_date_list:

            event_start_time = time.time()

            current_event_rows = []

            event_key = f"{keyword},{end_date_ymd}"
            if event_key in processed_events:
                print(f"--- [SKIP] {keyword} ({end_date_ymd})는(은) 이미 처리되었습니다. ---")
                continue

            try:
                end_date_dt = datetime.strptime(end_date_ymd, '%Y%m%d')
                start_date_dt = end_date_dt - timedelta(days=2)
                start_date_ymd = start_date_dt.strftime('%Y%m%d')

                print(f"\n" + "=" * 50)
                print(f"  [1/2] 주도주 이벤트 처리 시작: {keyword} (T일: {end_date_ymd})")
                print(f"=" * 50)

                base_event_data = {'주도주': keyword, 'T_Date': end_date_ymd}

                news_results = scrape_naver_news_only_full_xpath(
                    driver, keyword, start_date_ymd, end_date_ymd, max_count=10
                )

                news_filename = None
                if news_results:
                    news_filename = save_news_to_file(keyword, start_date_ymd, end_date_ymd, news_results)
                base_event_data['뉴스_파일_경로'] = news_filename

                combined_text = ""
                if news_results:
                    combined_text = "".join(
                        f"[기사 {i + 1}] 제목: {news['title']}\n---\n" for i, news in
                        enumerate(news_results))
                else:
                    print("⚠️ 뉴스가 추출되지 않아 분석을 건너뜁니다.")
                    continue

                finbert_text = ""
                if news_results:
                    finbert_text = "".join(
                        f"[기사 {i + 1}] 제목: {news['title']}\n내용: {news['summary']}\n---\n" for i, news in
                        enumerate(news_results))

                leader_themes = STOCK_TO_THEMES_DB.get(keyword)
                if not leader_themes:
                    print(f"❌ {keyword}가(이) 로드된 테마 DB에 없습니다. 이벤트를 건너뜁니다.")
                    continue

                event_theme, event_scale = call_openai_for_theme(combined_text, keyword, leader_themes)

                if not event_theme:
                    print(f"❌ OpenAI가 뉴스와 일치하는 테마를 분류하지 못했습니다. 이벤트를 건너뜁니다.")
                    continue

                print(f"--- OpenAI 테마 분류 완료: '{event_theme}' (규모: {event_scale}) ---")
                base_event_data['테마'] = event_theme
                base_event_data['뉴스기사_규모적지수'] = event_scale

                print(f"--- [2/2] 주도주({keyword}) 피처 계산 중... ---")
                leader_ticker = get_ticker_by_name(keyword, end_date_ymd)
                base_event_data['주도주_Ticker'] = leader_ticker
                if not leader_ticker:
                    print(f"❌ 주도주 Ticker를 찾지 못해 이벤트를 건너뜁니다.")
                    continue

                leader_features = calculate_leader_features(leader_ticker, end_date_ymd)
                if leader_features:
                    base_event_data.update(leader_features)
                    print(f"✅ 주도주 피처 계산 완료: {keyword} (Ticker: {leader_ticker})")
                else:
                    print(f"❌ 주도주 피처 계산 실패. 이벤트를 건너킵니다.")
                    continue

                print(f"--- 확정된 테마 [{event_theme}] 로 후보주 탐색... ---")
                followers = get_followers(event_theme, keyword)
                if not followers:
                    print(f"⚠️ {event_theme} 테마에 후보주가 없습니다.")
                    continue

                for follower_name in followers:
                    try:
                        follower_ticker = get_ticker_by_name(follower_name, end_date_ymd)
                        if not follower_ticker:
                            continue

                        row_data = base_event_data.copy()
                        row_data['후보주'] = follower_name
                        row_data['후보주_Ticker'] = follower_ticker

                        theme_keyword_simple = event_theme.split('(')[0].strip()
                        filter_keywords = [keyword, follower_name, theme_keyword_simple]

                        sentiment_score, sentence_score_list = calculate_finbert_sentiment(
                            finbert_text,
                            filter_keywords,
                            tokenizer,
                            model
                        )

                        print(f"\n--- FinBert 분석 ({keyword} -> {follower_name}) ---")
                        if sentence_score_list:
                            print(f"  [분석 대상 문장 ({len(sentence_score_list)}개)]")
                            for sent, score in sentence_score_list:
                                print(f"    - [점수: {score:+.4f}] {sent}")
                            print(f"  [최종 평균 점수]: {sentiment_score:.4f}")
                            save_sentiment_to_csv(
                                keyword, follower_name, start_date_ymd, end_date_ymd,
                                sentence_score_list, sentiment_score
                            )
                        else:
                            print(f"  [분석 대상 문장 없음] (키워드: {', '.join(filter_keywords)})")

                        row_data['뉴스기사_감성지수'] = sentiment_score

                        tech_f, funda_f, actual_t_day_ts, t_day_close = calculate_follower_features(
                            follower_ticker, end_date_ymd
                        )
                        if tech_f: row_data.update(tech_f)
                        if funda_f: row_data.update(funda_f)

                        target_label = get_next_day_label(follower_ticker, actual_t_day_ts, t_day_close)
                        row_data['Target'] = target_label

                        correlation = get_correlation(leader_ticker, follower_ticker, end_date_ymd)
                        row_data['상관관계_지수'] = correlation

                        count, theme_list = get_theme_overlap(follower_name)
                        row_data['테마_중복_개수'] = count
                        row_data['테마_중복_목록'] = theme_list

                        current_event_rows.append(row_data)

                    except Exception as e:
                        print(f"❌ 후보주 '{follower_name}' 처리 중 오류: {e}")
                        continue

                if current_event_rows:
                    all_events_rows.extend(current_event_rows)
                    print(f"✅ {keyword}({end_date_ymd}) 이벤트 처리 완료. 누적 데이터: {len(all_events_rows)} 행.")

                    save_intermediate_dataset(all_events_rows, final_output_filename)

                    with open(log_file_path, 'a', encoding='utf-8') as f:
                        f.write(f"{event_key}\n")

                    processed_events.add(event_key)

                else:
                    print(f"--- {keyword}({end_date_ymd}) 이벤트에서 처리된 후보주가 없어, 추가된 데이터가 없습니다. ---")

            except Exception as e:
                print(f"❌ '{keyword}' 이벤트 전체 작업 중 치명적인 오류 발생: {e}")

            event_elapsed_time = time.time() - event_start_time
            events_processed_count += 1
            print(f"--- ⏱️  {keyword}({end_date_ymd}) 처리 소요 시간: {event_elapsed_time:.2f}초 ---")

            total_completed = len(processed_events) + events_processed_count
            remaining_events = len(keyword_date_list) - total_completed

            if events_processed_count > 0:
                avg_time_per_event = (time.time() - total_start_time) / events_processed_count
                eta_seconds = avg_time_per_event * remaining_events
                eta_minutes = eta_seconds / 60
                print(
                    f"--- 📊 진행 상황: {total_completed}/{len(keyword_date_list)} | 남은 예상 시간: {eta_minutes:.1f}분 ({eta_seconds:.0f}초) ---")

    finally:
        if driver:
            driver.quit()
            print("\n--- 모든 작업 완료. 웹 드라이버를 종료합니다. ---")

    if not all_events_rows:
        print("\n--- 처리된 데이터가 없어 최종 CSV 파일을 저장하지 않습니다. ---")
    else:
        total_elapsed_time = time.time() - total_start_time
        total_minutes = total_elapsed_time / 60

        print(f"\n" + "=" * 50)
        print(f"  🎉 모든 작업 완료 (이번 실행에서 {events_processed_count}개 신규 이벤트 처리)")
        print(f"  이번 실행 총 소요 시간: {total_minutes:.2f}분 ({total_elapsed_time:.2f}초)")
        print(f"  최종 파일: {final_output_filename}")
        print(f"  총 {len(all_events_rows)}개의 (주도주-후보주) 관계가 처리되었습니다.")
        print(f"=" * 50)

    print("\n=== 모든 작업 완료 ===")
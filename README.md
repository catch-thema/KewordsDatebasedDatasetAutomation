# 📈 주도주-후보주 하이브리드 데이터셋 구축 스크립트

이 스크립트는 `input_list.csv` 파일에 명시된 **주도주**와 **기준 날짜(T일)** 목록을 기반으로, 관련 후보주들의 머신러닝 학습용 데이터셋을 자동으로 구축합니다.

데이터 구축 과정은 뉴스 스크래핑, AI/감성 분석, 재무/기술적 지표 계산을 포함하며, 최종적으로 후보주의 **다음 영업일(T+1) 주가 상승 여부**를 예측하는 **Target Label**을 생성합니다.

##  주요 기능

* **하이브리드 데이터 수집**:
    * **뉴스 (정성적)**: `Selenium`을 사용하여 네이버 뉴스에서 T-2 ~ T일 사이의 주도주 관련 기사를 최대 10개까지 스크래핑합니다.
    * **재무/기술 (정량적)**: `pykrx`와 `pandas-ta-classic`을 사용하여 주도주와 후보주 각각의 기술적 지표(RSI, MACD, 이격도, RVOL 등)와 재무 지표(PER, PBR, BPS 등)를 수집합니다.

* **AI 기반 테마/규모 분석**:
    * 스크래핑한 뉴스 **제목**들을 `OpenAI (gpt-5-nano)`에 전달하여, 주도주가 속한 가장 관련성 높은 **핵심 테마**와 뉴스의 **규모적 지수**(예: 계약금액)를 자동으로 추출합니다.

* **금융 특화 감성 분석**:
    * `snunlp/KR-FinBert-SC` 모델을 사용하여 스크래핑한 뉴스 **본문 전체**에서 (주도주, 후보주, 테마) 키워드가 포함된 문장들의 감성 점수를 계산합니다.

* **Target Label 생성**:
    * 후보주의 T+1 영업일 종가를 T일 종가와 비교하여, **상승 시 `1`, 하락/보합 시 `0`**으로 라벨링합니다. (`get_next_day_label` 함수)

* **자동화 및 관리**:
    * **이어하기 지원**: `leader_follower_datasets/processed_log.txt` 파일을 통해 이미 처리된 이벤트를 건너뛰어 중단된 지점부터 다시 시작할 수 있습니다.
    * **중간 저장**: 주도주 이벤트 1개가 완료될 때마다 `final_dataset_combined.csv`에 즉시 덮어쓰기 저장하여 데이터 유실을 방지합니다.
    * **더미 변수 생성**: 최종 저장 시 `테마_중복_목록` 컬럼을 `naver_themes_bs4.csv` 기준의 원-핫 인코딩(더미 변수) 컬럼으로 자동 변환합니다.

 **주의**: 현재 이 로직은 상승한 것을 기반으로 Target Value를 구축하고 있습니다. 하락의 경우
 'return 1 if t_plus_1_close > t_day_close else 0' 부분을 수정해야 합니다.
 
## 🚀 사용 방법

### 1. 환경 준비 (Setup)

1.  Python 가상환경을 생성하고 활성화합니다.
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # Linux/Mac
    # source venv/bin/activate
    ```
2.  아래 `requirements.txt` 파일의 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

### 2. 필수 파일 설정 (Configuration)

스크립트 실행 전, `script.py`와 동일한 디렉토리에 다음 파일들을 준비해야 합니다.

1.  **OpenAI API 키 입력**
    * `script.py` 파일 상단의 `OPENAI_API_KEY = ""` 변수에 본인의 OpenAI API 키를 직접 입력합니다.

2.  **테마 DB: `naver_themes_bs4.csv`**
    *  테마DB 파일. (컬럼: `테마명`, `종목1`, `종목2`, ...)

3.  **입력 목록: `input_list.csv`** (파일명은 자유롭게 지정 가능)
    * 데이터를 수집할 주도주와 기준 날짜 목록.
    * 반드시 첫 번째 열이 **종목명**, 두 번째 열이 **날짜**여야 합니다. (예: `종목명`, `날짜` (YYYYMMDD 형식))

### 3. 스크립트 실행 (Run)

모든 설정이 완료되었으면 메인 스크립트를 실행합니다.

```bash
python script.py
```

* `스크래핑 목록 CSV 파일명 입력 (예: input_list.csv):` 프롬프트가 나타나면, 준비한 **입력 목록 CSV 파일명** (예: `input_list.csv`)을 입력하고 엔터를 누릅니다.

## 📊 결과물 (Output)

스크립트가 실행되면 다음 디렉토리 및 파일이 생성됩니다.

* `leader_follower_datasets/final_dataset_combined.csv`:
    모든 피처와 Target Label이 포함된 **최종 데이터셋**
* `leader_follower_datasets/processed_log.txt`:
    이미 처리가 완료된 (주도주, 날짜) 이벤트 목록
* `news_scraped/`:
    이벤트별로 스크래핑된 원본 뉴스 텍스트 파일 (`.txt`)
* `sentiment_analyze/`:
    (주도주-후보주) 관계별 FinBert 감성 분석 상세 결과 (`.csv`)

#

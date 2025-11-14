어떤 주식에 대한 주도주(테마주 중 시총이 높은 주식) 이름과 오르거나 내린 날짜를 입력하면 그에 맞춰서 감성점수/규모 분석, Pykrw를 이용한 주도주와 테마내의 후보주의 기술적, 정성적 평가, 주도주와 해당 후보주의 상관관계 지수를 학습 데이터로 사용하고
다음날 해당 후보주가 오르거나 내렸는지에 관한 타겟 데이터를 구축하여 데이터셋 구축을 자동화 한 스크립트

OPEN API 키가 필요합니다.
감성점수 분석에는 https://huggingface.co/snunlp/KR-FinBert-SC 해당 FinBert 모델을 사용합니다.
기본값은 상승으로 구현되어 있으나, 하락에 대한 데이터셋 구축이 필요한 경우
return 1 if t_plus_1_close > t_day_close else 0
해당 부분을
return 1 if t_plus_1_close < t_day_close else 0
로 변경해주시면 됩니다.

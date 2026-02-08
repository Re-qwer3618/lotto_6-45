import requests
import pandas as pd
import numpy as np
import datetime
from notion_client import Client
import os

# 1. 데이터 수집 자동화 (매주 토요일 데이터 추가)
def update_lotto_data(file_path):
    df = pd.read_csv(file_path, header=None)
    last_drw_no = int(df.iloc[-1, 0])
    next_drw_no = last_drw_no + 1
    
    # 동행복권 API 호출
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={next_drw_no}"
    res = requests.get(url).json()
    
    if res.get('returnValue') == 'success':
        new_row = [
            next_drw_no,
            res['drwtNo1'], res['drwtNo2'], res['drwtNo3'],
            res['drwtNo4'], res['drwtNo5'], res['drwtNo6'],
            res['bnusNo'], res['firstAccumamnt']
        ]
        df.loc[len(df)] = new_row
        df.to_csv(file_path, index=False, header=False)
        print(f"{next_drw_no}회차 데이터 업데이트 완료!")
        return True, next_drw_no
    else:
        print("아직 새로운 회차 데이터가 없습니다.")
        return False, last_drw_no

# 2. 딥러닝 예측 (보내주신 Lotto.ipynb 로직 핵심 요약)
def predict_next_lotto(file_path):
    # 실제 구현시에는 업로드하신 Lotto.ipynb의 학습/예측 로직을 함수화하여 넣습니다.
    # 여기서는 예시로 랜덤 6개 숫자를 생성합니다 (실제 연동시 본인 모델 코드로 교체)
    predicted_numbers = sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())
    return [int(x) for x in predicted_numbers]

# 3. 노션 페이지 업로드
def upload_to_notion(token, database_id, next_no, numbers):
    notion = Client(auth=token)
    new_page = {
        "parent": {"database_id": database_id},
        "properties": {
            "회차": {"title": [{"text": {"content": f"{next_no}회 예측"}}] settlements},
            "예측번호": {"rich_text": [{"text": {"content": str(numbers)}}]},
            "생성날짜": {"date": {"start": datetime.datetime.now().isoformat()}}
        }
    }
    notion.pages.create(**new_page)
    print("노션 업로드 완료!")

if __name__ == "__main__":
    CSV_FILE = "lotto.csv"
    NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
    DATABASE_ID = os.environ.get("NOTION_DATABASE_ID")

    # 데이터 업데이트
    updated, current_no = update_lotto_data(CSV_FILE)
    
    # 예측 및 업로드
    prediction = predict_next_lotto(CSV_FILE)
    upload_to_notion(NOTION_TOKEN, DATABASE_ID, current_no + 1, prediction)

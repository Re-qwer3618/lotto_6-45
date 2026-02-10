import numpy as np
import pandas as pd
import requests
import os
import datetime
from tensorflow import keras
from notion_client import Client

# 1. 새로운 당첨 데이터 수집 (매주 토요일 데이터 업데이트용)
def update_data(file_path):
    df = pd.read_csv(file_path, header=None)
    last_no = int(df.iloc[-1, 0])
    next_no = last_no + 1
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={next_no}"
    res = requests.get(url).json()
    if res.get('returnValue') == 'success':
        # 회차, 번호1~6, 보너스, 당첨금 순서
        new_row = [next_no, res['drwtNo1'], res['drwtNo2'], res['drwtNo3'], 
                   res['drwtNo4'], res['drwtNo5'], res['drwtNo6'], res['bnusNo'], res['firstAccumamnt']]
        df.loc[len(df)] = new_row
        df.to_csv(file_path, index=False, header=False)
        return True, next_no
    return False, last_no

# 2. 딥러닝 예측 로직
def predict_lotto(file_path):
    # CSV 로드 (번호 부분인 1~7열만 사용)
    df = pd.read_csv(file_path, header=None)
    numbers = df.iloc[:, 1:7].values
    
    # 원핫인코딩 변환
    def numbers2ohbin(nums):
        ohbin = np.zeros(45)
        for n in nums:
            if 1 <= int(n) <= 45:
                ohbin[int(n)-1] = 1
        return ohbin

    ohbins = np.array([numbers2ohbin(n) for n in numbers])
    x_train = ohbins[:-1].reshape(-1, 1, 45)
    y_train = ohbins[1:].reshape(-1, 45)

    # 모델 설정 및 학습
    model = keras.Sequential([
        keras.Input(shape=(1, 45)),
        keras.layers.LSTM(128, return_sequences=False),
        keras.layers.Dense(45, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    # 빠른 실행을 위해 Epoch 50회 설정
    model.fit(x_train, y_train, epochs=50, verbose=0)

    # 마지막 데이터를 넣어 다음 회차 예측
    last_xs = ohbins[-1].reshape(1, 1, 45)
    pred_prob = model.predict(last_xs, verbose=0)[0]
    
    # 확률 기반으로 6개 번호 추출
    ball_box = []
    for n in range(45):
        ball_box += [n+1] * int(pred_prob[n] * 100 + 1)
    
    result = []
    while len(result) < 6:
        pick = np.random.choice(ball_box)
        if pick not in result:
            result.append(int(pick))
    return sorted(result)

# 3. 노션 업로드 (오타 수정 완료)
def send_to_notion(numbers, next_no):
    token = os.environ.get("NOTION_TOKEN")
    db_id = os.environ.get("NOTION_DATABASE_ID")
    if not token or not db_id:
        print("노션 토큰 또는 데이터베이스 ID가 설정되지 않았습니다.")
        return
    
    notion = Client(auth=token)
    try:
        notion.pages.create(
            parent={"database_id": db_id},
            properties={
                "회차": {"title": [{"text": {"content": f"{next_no}회 예측"}}]},
                "번호": {"rich_text": [{"text": {"content": str(numbers)}}]},
                "날짜": {"date": {"start": datetime.date.today().isoformat()}}
            }
        )
        print("노션에 결과 업로드 성공!")
    except Exception as e:
        print(f"노션 업로드 에러: {e}")

if __name__ == "__main__":
    csv_file = "lotto.csv"
    # 1. 데이터 업데이트 시도
    updated, current_no = update_data(csv_file)
    # 2. 다음 회차 예측 (현재 회차 + 1)
    pred = predict_lotto(csv_file)
    # 3. 노션 전송
    send_to_notion(pred, current_no + 1)
    print(f"{current_no + 1}회차 예측 번호: {pred}")

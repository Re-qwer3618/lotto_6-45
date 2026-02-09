import numpy as np
import pandas as pd
import requests
import os
import datetime
from tensorflow import keras
from notion_client import Client

# 1. 새로운 당첨 데이터 수집
def update_data(file_path):
    df = pd.read_csv(file_path, header=None)
    last_no = int(df.iloc[-1, 0])
    next_no = last_no + 1
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={next_no}"
    res = requests.get(url).json()
    if res.get('returnValue') == 'success':
        new_row = [next_no, res['drwtNo1'], res['drwtNo2'], res['drwtNo3'], 
                   res['drwtNo4'], res['drwtNo5'], res['drwtNo6'], res['bnusNo'], 0]
        df.loc[len(df)] = new_row
        df.to_csv(file_path, index=False, header=False)
        return True, next_no
    return False, last_no

# 2. 딥러닝 예측 로직 (노트북 내용 통합)
def predict_lotto(file_path):
    rows = np.loadtxt(file_path, delimiter=",")
    numbers = rows[:, 1:7]
    
    # 원핫인코딩 변환 함수
    def numbers2ohbin(nums):
        ohbin = np.zeros(45)
        for n in nums: ohbin[int(n)-1] = 1
        return ohbin

    ohbins = np.array(list(map(numbers2ohbin, numbers)))
    x_train = ohbins[:-1].reshape(-1, 1, 45)
    y_train = ohbins[1:].reshape(-1, 45)

    # 모델 정의 및 학습
    model = keras.Sequential([
        keras.Input(batch_shape=(1, 1, 45)),
        keras.layers.LSTM(128, return_sequences=False, stateful=True),
        keras.layers.Dense(45, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    # 학습 (자동화를 위해 epoch를 조절했습니다)
    model.layers[0].reset_states()
    for _ in range(50): 
        for i in range(len(x_train)):
            model.train_on_batch(x_train[i:i+1], y_train[i:i+1])

    # 다음 회차 예측
    last_xs = ohbins[-1].reshape(1, 1, 45)
    pred_prob = model.predict_on_batch(last_xs)[0]
    
    # 확률 기반 번호 추출
    ball_box = []
    for n in range(45):
        ball_box += [n+1] * int(pred_prob[n] * 100 + 1)
    
    result = []
    while len(result) < 6:
        pick = np.random.choice(ball_box)
        if pick not in result: result.append(int(pick))
    return sorted(result)

# 3. 노션 업로드
def send_to_notion(numbers, next_no):
    token = os.environ.get("NOTION_TOKEN")
    db_id = os.environ.get("NOTION_DATABASE_ID")
    if not token or not db_id: return
    
    notion = Client(auth=token)
    notion.pages.create(
        parent={"database_id": db_id},
        properties={
            "회차": {"title": [{"text": {"content": f"{next_no}회 예측" Erik}}]},
            "번호": {"rich_text": [{"text": {"content": str(numbers)}}]},
            "날짜": {"date": {"start": datetime.date.today().isoformat()}}
        }
    )

if __name__ == "__main__":
    csv_file = "lotto.csv"
    updated, current_no = update_data(csv_file)
    pred = predict_lotto(csv_file)
    send_to_notion(pred, current_no + 1)
    print(f"Prediction for {current_no + 1}: {pred}")

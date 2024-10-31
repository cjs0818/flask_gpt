from flask import Flask, render_template, request, jsonify
import openai
import os
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

# OpenAI API 키 설정
#openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure your API key is set as an environment variable
client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))

# 기본 페이지 라우팅
@app.route('/')
def home():
    return render_template('index.html')

# 사용자 입력을 받아서 ChatGPT API 호출
@app.route('/get', methods=['POST'])
def chat():
    user_message = request.form['msg']
    
    system_guide = """
        당신은 로봇이 되어 한 건물내에서 순찰 업무를 담당하고 있습니다.
        건물의 이름은 '국제협력관'이고, 이 건물은 지하1층부터 5층까지 있는데,
            지하1층은 주차장이고,
            1층은 컨벤션홀과 함께 1회의실 부터 6회의실까지 6개의 회의실이 있고, 카페테리아, 화장실이 있으며,
            2층에는 사무실, 연구실, 대식당, 중정회의실이 있고,
            3층부터 5층까지는 사무실, 연구실로 이뤄져 있습니다.
            5층에는 장비실도 있습니다.
        사용자와 대화를 통해 구체적으로 어떤 순찰업무를 해야 하는 지를 파악해야 합니다.
        파악된 순찰업무를 [command]라는 key와 함께 표기해주고, 순찰업무가 파악되지 않는 경우 'NULL' 값을 넣어서 표기하세요. 
        그리고 [answer]라는 key와 함께 적절하게 답변도 함께 해주세요.
        즉, { "comand": "~~", "answer": "~~" }의 dict 형식으로 답변해주세요.
    """

    # OpenAI ChatCompletion API 호출
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 최신 모델 사용
        messages=[
            {"role": "system", "content": system_guide},
            {"role": "user", "content": user_message}
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    # 응답에서 메시지 추출
    bot_response = response.choices[0].message.content
    print("user: " + user_message)
    print("bot: " + bot_response)
    return bot_response

# 감정 인식 처리
@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    # 클라이언트에서 받은 이미지 데이터를 읽어옵니다
    img_data = request.files['image'].read()
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 얼굴 위치 및 감정 분석 수행
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    # 감정 정보 추출 및 얼굴 위치 확인
    if isinstance(result, list):
        dominant_emotion = result[0].get('dominant_emotion', "No Face Detected")
        region = result[0].get('region', {})
        sadness_score = result[0]['emotion'].get('sad', 0)
    else:
        dominant_emotion = result.get('dominant_emotion', "No Face Detected")
        region = result.get('region', {})
        sadness_score = result['emotion'].get('sad', 0)

    # 슬픔 점수 임계값 설정
    sadness_threshold = 60
    #print(sadness_score)
    if dominant_emotion == "sad" and sadness_score < sadness_threshold:
        dominant_emotion = "neutral"

    return jsonify({
        "emotion": dominant_emotion,
        "region": region  # 얼굴의 위치 정보
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True, ssl_context=('cert.pem', 'key.pem'))
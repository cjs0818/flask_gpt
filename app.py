from flask import Flask, render_template, request
import openai
import os

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
            2층에는 사무실, 연구실, 대식당이 있고,
            3층부터 5층까지는 사무실, 연구실로 이뤄져 있습니다.
        사용자와 대화를 통해 구체적으로 어떤 순찰업무를 해야 하는 지를 파악해야 합니다.
        파악된 순찰업무를 [command]라는 key와 함께 표기해주고, 순찰업무가 파악되지 않는 경우 'NULL' 값을 넣어서 표기해줘. 
        그리고 [answer]라는 key와 함께 적절하게 답변도 함께 해줘.
        즉, { "comand": "~~", "answer": "~~" }의 dict 형식으로 답변해줘.
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
    return bot_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True, ssl_context=('cert.pem', 'key.pem'))
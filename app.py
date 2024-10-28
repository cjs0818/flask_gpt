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
    
    # OpenAI ChatCompletion API 호출
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 최신 모델 사용
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    # 응답에서 메시지 추출
    bot_response = response.choices[0].message.content
    return bot_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
# Telegramm bot speech to text

Writing using python-telegramm-bot library and DeepSpech engine with pre-trained
models

# Dependencies

 * DeepSpeech
 * ffmpeg
 * python-telegram-bot

# Run

python3 -m venv venv
pip3 install -r requirements.txt
python3 main.py --model ./models/ --aggressive 1

# Download models

Must include .bpmp and .score file in models folder

https://github.com/mozilla/DeepSpeech
from telegram.ext import Updater, InlineQueryHandler, CommandHandler, \
MessageHandler, Filters
from telegram.ext.dispatcher import run_async
import logging
import requests
import re
import sys
import os
import argparse
import numpy as np
import wavTranscriber


model_data = None
aggressive = 1

# def get_url():
#     contents = requests.get('https://random.dog/woof.json').json()
#     url = contents['url']
#     return url

# def get_image_url():
#     allowed_extension = ['jpg','jpeg','png']
#     file_extension = ''
#     while file_extension not in allowed_extension:
#         url = get_url()
#         file_extension = re.search("([^.]*)$",url).group(1).lower()
#     return url

# @run_async
# def bop(update, context):
#     url = get_image_url()
#     chat_id = update.message.chat_id
#     context.bot.send_photo(chat_id=chat_id, photo=url)

@run_async
def speech_to_text(update, context):

    res_message = ""
    audio_data = context.bot.get_file(update.message.voice.file_id)
    audio_name = audio_data.file_unique_id + '.oga'
    audio_path = './voice_data/'
    audio_full = audio_path + audio_name

    if model_data is None:
        return

    with open(audio_full, 'wb') as f:
        f.write(requests.get(audio_data.file_path).content)
    f.close()

    new_audio = audio_full.rstrip(".oga") + ".wav"
    os.system("ffmpeg -i {}  -ar 16000 -ac 1  {}".format(audio_full, new_audio))

    title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 
                   'Model Load Time(s)', 'Scorer Load Time(s)']
    print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], 
          title_names[2], title_names[3], title_names[4]))

    inference_time = 0.0

    # Run VAD on the input file
    waveFile = new_audio
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile, aggressive)

    # logging.debug("Saving Transcript @: %s" % waveFile.rstrip(".wav") + ".txt")
    
    for i, segment in enumerate(segments):
        # Run deepspeech on the chunk that just completed VAD
        # logging.debug("Processing chunk %002d" % (i,))
        audio = np.frombuffer(segment, dtype=np.int16)
        output = wavTranscriber.stt(model_data[0], audio, sample_rate)
        inference_time += output[1]
        # logging.debug("Transcript: %s" % output[0])
        res_message += output[0] + " "

    os.remove(audio_full)
    os.remove(new_audio)

    context.bot.send_message(chat_id=update.effective_chat.id, text=res_message)

def preload_model(args):
    parser = argparse.ArgumentParser(description='Transcribe long audio files using webRTC VAD or use the streaming interface')
    parser.add_argument('--aggressive', type=int, choices=range(4), required=False,
                        help='Determines how aggressive filtering out non-speech is. (Interger between 0-3)')
    parser.add_argument('--model', required=True,
                        help='Path to directory that contains all model files (output_graph and scorer)')

    args = parser.parse_args()

    # Point to a path containing the pre-trained models & resolve ~ if used
    aggressive = args.aggressive
    dirName = os.path.expanduser(args.model)

    # Resolve all the paths of model files
    output_graph, scorer = wavTranscriber.resolve_models(dirName)

    # Load output_graph, alpahbet and scorer
    model_retval = wavTranscriber.load_model(output_graph, scorer)
    return model_retval

def main(args):

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
   
    global model_data
    model_data = preload_model(args)
    print("Model data {}".format(model_data))

    tok = open("./token", "r")
    updater = Updater(tok.read(), use_context=True)
    dp = updater.dispatcher
    # dp.add_handler(CommandHandler('bop',bop))
    dp.add_handler(MessageHandler(Filters.voice, speech_to_text))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main(sys.argv[1:])
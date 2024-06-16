import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import numpy as np
import datetime
from deepgram import Deepgram
import asyncio
from transformers import pipeline

# Initialize the Deepgram API with your key
DEEPGRAM_API_KEY = "your_api_key"
dg = Deepgram(DEEPGRAM_API_KEY)

# Sentiment analysis function
async def analyze_audio(audio_path: str):
    try:
        with open(audio_path, 'rb') as audio:
            source = {'buffer': audio, 'mimetype': 'wav' if audio_path.endswith('.wav') else 'mp3'}
            response = await dg.transcription.prerecorded(source, {'paragraphs': True, 'diarize': True})
            
            all_sentences = response['results']['channels'][0]['alternatives'][0]['paragraphs']['paragraphs']
            result = []
            for s in all_sentences:
                speaker_sentences = s['sentences']
                for sentence in speaker_sentences:
                    sentence['speaker'] = s['speaker']
                    result.append(sentence)
            
            df = pd.DataFrame(result)
            
            df['sentiment'] = df['text'].apply(analyze_statement)  # Assuming analyze_statement function exists
            df['start_time'] = df['start'].apply(lambda x: datetime.datetime.fromtimestamp(x))
            df['end_time'] = df['end'].apply(lambda x: datetime.datetime.fromtimestamp(x))
            
            return df
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None

def analyze_statement(statement: str):
    sentiment_analysis = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", top_k=3)
    sent = sentiment_analysis(statement)
    cleaned_dict = process_sentiment_res(sent)
    return calculate_compound_score(cleaned_dict)

def process_sentiment_res(sent):
    res = {}
    for s in sent[0]:
        if s['label'] == 'LABEL_2':
            label = 'POSITIVE'
        elif s['label'] == 'LABEL_0':
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        res[label] = s['score']
    return res

def calculate_compound_score(sent_res):
    sentiment_probabilities = np.array([sent_res['NEGATIVE'], sent_res['NEUTRAL'], sent_res['POSITIVE']])
    weights = np.array([-1, 0, 1], dtype=np.float32)
    return np.dot(sentiment_probabilities, weights)

# Function to generate sentiment plots
def generate_plots(df: pd.DataFrame):
    fig, axes = plt.subplots(df['speaker'].max() + 1, figsize=(10, 20))

    plt.subplots_adjust(left=0.1, bottom=0, right=0.9, top=1, wspace=0.4, hspace=0.4)

    speaker_names = ["Agent", "Customer"]  # Assuming 0 is Agent and 1 is Customer

    for speaker, ax in enumerate(axes):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Sentiment')
        ax.set_title(f'Speaker {speaker_names[speaker]} \n (Time-wise Sentiment Analysis)')

        datapoints = df[df['speaker'] == speaker]
        ax.plot(datapoints['start_time'], datapoints['sentiment'], label='Sentiment')

        ax.axhline(0, color='gray', linestyle='--', label='Neutral')
        ax.axhline(1, color='green', linestyle='--', label='Positive')
        ax.axhline(-1, color='red', linestyle='--', label='Negative')

        ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return buf

# Function to get a summary of sentiments
def get_summary(df: pd.DataFrame):
    summary = {}
    for speaker in df['speaker'].unique():
        speaker_name = "Agent" if speaker == 0 else "Customer"
        avg_sentiment = df[df['speaker'] == speaker]['sentiment'].mean()
        summary[speaker_name] = f"Average Sentiment: {'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral'}"
    return summary

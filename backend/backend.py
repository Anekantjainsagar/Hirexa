from flask import Flask, request,jsonify 
from flask_cors import CORS, cross_origin
import numpy as np
import moviepy.editor as mp
from google.cloud import speech_v1p1beta1 as speech
import nltk
from textblob import TextBlob
import speech_recognition as sr
from langdetect import detect
import cv2
import os
from fer import FER
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from scipy.io import wavfile
from twilio.rest import Client
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('vader_lexicon')
app = Flask(__name__)
CORS(app, support_credentials=True)

angry=0
disgust=0
fear=0
happy=0
sad=0
surprise=0 
neutral=0

def extract_frames(video_name):	
    # Read the video from specified path
    cam = cv2.VideoCapture(video_name)

    try:
        
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')

    # frame rate
    frame_rate = 2

    # calculate interval between frames
    interval = int(cam.get(cv2.CAP_PROP_FPS) / frame_rate)
    # frame
    currentframe = 0

    while(True):
        
        # reading from frame
        ret,frame = cam.read()

        if ret:
            if currentframe % interval == 0:
                # if video is still left continue creating images
                name = './data/frame' + str(int(currentframe/14)) + '.jpg'
                print ('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    return int(currentframe/14 - 1)

def detect_emotion(frame_path): #frame_path is a string of the path to the image
    global angry, disgust, fear, happy, sad, surprise, neutral
    # Input Image
    try:
        input_image = cv2.imread(frame_path)
        emotion_detector = FER()
        # Output image's information
        angry = angry + emotion_detector.detect_emotions(input_image)[0]["emotions"]["angry"]
        disgust = disgust + emotion_detector.detect_emotions(input_image)[0]["emotions"]["disgust"]
        fear = fear + emotion_detector.detect_emotions(input_image)[0]["emotions"]["fear"]
        happy = happy + emotion_detector.detect_emotions(input_image)[0]["emotions"]["happy"]
        sad = sad + emotion_detector.detect_emotions(input_image)[0]["emotions"]["sad"]
        surprise = surprise + emotion_detector.detect_emotions(input_image)[0]["emotions"]["surprise"]
        neutral = neutral + emotion_detector.detect_emotions(input_image)[0]["emotions"]["neutral"]
        print(emotion_detector.detect_emotions(input_image)[0]["emotions"])    
    except:
        print ('Error..Analysing next frame')

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['file']
    video.save(video.filename)  
    
    filename = video.filename
    vid = mp.VideoFileClip(filename)
    audio = vid.audio
    audio_file_name_without_ext = filename.split('.mp4')[0]
    audio_file_name = "{}.wav".format(audio_file_name_without_ext)
    audio.write_audiofile(audio_file_name)
    print(audio_file_name)
    
    recognizer = sr.Recognizer()
    audioFile = sr.AudioFile(audio_file_name)
    with audioFile as source:
        data = recognizer.record(source)
    text = recognizer.recognize_google(data, key=None)
    print(text)
    
    # Sentiment analysis
    from nltk.sentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    print("Sentiment score is : ")
    print(sentiment_scores)
    sentiment = ""
    if sentiment_scores["compound"] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores["compound"] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print("Sentiment: ", sentiment)

    # Perform language detection and English fluency analysis
    language = detect(text)
    if language == "en":
        blob = TextBlob(text)
        fluency = blob.correct().string
        print("Fluency: ", fluency)
    else:
        print("Language is not English.")
    
    number_of_frames = extract_frames(filename)

    video_data = []
    for i in range(number_of_frames):
        frame_path = "./data/frame" + str(i) + ".jpg"
        detect_emotion(frame_path)


    # Showing emotions 
    print("Emotions print karo")
    emotions = [angry, disgust, fear, happy, sad, surprise, neutral]
    emo = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    max_emotion = emotions.index(max(emotions))
    print(max_emotion)
    
    # Define the emotion values for video and audio
    video_emotions = {
        'Angry': emotions[0],
        'Disgust': emotions[1],
        'Fear': emotions[2],
        'Happy': emotions[3],
        'Sad': emotions[4],
        'Surprise': emotions[5],
        'Neutral': emotions[6]
    }

    audio_emotions = {
        'Positive': sentiment_scores['pos'],
        'Negative': sentiment_scores['neg'],
        'Neutral': sentiment_scores['neu']
    }

    # Define the weights for video and audio emotions
    video_weights = {
        'Angry': 2,
        'Disgust': 1,
        'Fear': 1,
        'Happy': 3,
        'Sad': 2,
        'Surprise': 2,
        'Neutral': 1
    }

    audio_weights = {
        'Positive': 2,
        'Negative': 2,
        'Neutral': 1
    }

    # Calculate the weighted scores for video and audio
    video_score = sum(video_weights[emotion] * video_emotions[emotion] for emotion in video_emotions)
    audio_score = sum(audio_weights[emotion] * audio_emotions[emotion] for emotion in audio_emotions)

    # Define the thresholds for each ranking category
    low_threshold = 70
    medium_threshold = 120

    audio_score = audio_score * 60
    
    course = ""
    
    # Assign ranking categories based on the scores
    if video_score > audio_score:
        if video_score <= low_threshold:
            course = 'Communication in the 21st Century Workplace'
        elif video_score <= medium_threshold:
            course = 'Communication Skills for University Success'
        else:
            course = 'Take Your English Communication Skills to the Next Level'
    elif video_score < audio_score:
        if audio_score <= low_threshold:
            course = 'Introduction to Communication Science'
        elif audio_score <= medium_threshold:
            course = 'Oral Communication for Engineering Leaders'
        else:
            course = 'Business Russian Communication. Part 3'
    
    final_data = { "audio" : sentiment_scores, "video" : emotions, "text" : text,"course":course}
    
    # # Process the video as needed
    return jsonify(final_data)

def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)

def recommend(new_df,similarity,course):
    course_index = new_df[new_df['Course Name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:7]
    result_list = []
    
    for i in course_list:
        course_name = new_df.iloc[i[0]]['Course Name']
        course_url = new_df.iloc[i[0]]['Course URL']
        course_desc = new_df.iloc[i[0]]['Course Description']
        result_list.append({"name": course_name, "url": course_url,"description": course_desc})
        
    return result_list

@app.route('/', methods=['POST'])
def result():
    data = pd.read_csv("../Essetials/Coursera.csv")

    
    data = data[['Course Name','Difficulty Level','Course Description','Skills','Course URL']]
    
    # Removing spaces between the words (Lambda funtions can be used as well)
    data['Course Name'] = data['Course Name'].str.replace(' ',',')
    data['Course Name'] = data['Course Name'].str.replace(',,',',')
    data['Course Name'] = data['Course Name'].str.replace(':','')
    data['Course Description'] = data['Course Description'].str.replace(' ',',')
    data['Course Description'] = data['Course Description'].str.replace(',,',',')
    data['Course Description'] = data['Course Description'].str.replace('_','')
    data['Course Description'] = data['Course Description'].str.replace(':','')
    data['Course Description'] = data['Course Description'].str.replace('(','')
    data['Course Description'] = data['Course Description'].str.replace(')','')

    #removing paranthesis from skills columns 
    data['Skills'] = data['Skills'].str.replace('(','')
    data['Skills'] = data['Skills'].str.replace(')','')
    
    data['tags'] = data['Course Name'] + data['Difficulty Level'] + data['Course Description'] + data['Skills']
    
    data['tags'].iloc[1]
    
    new_df = data[['Course Name','tags','Course URL','Course Description']]
    
    new_df.loc[:, 'tags'] = data['tags'].str.replace(',', ' ')
    new_df.loc[:, 'Course Name'] = data['Course Name'].str.replace(',', ' ')
    
    new_df.rename(columns={'Course Name': 'course_name'})
    
    new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower()) #lower casing the tags column
    
    cv = CountVectorizer(max_features=5000,stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    new_df.loc[:, 'tags'] = new_df['tags'].apply(stem) #applying stemming on the tags column
    similarity = cosine_similarity(vectors)
    result = recommend(new_df,similarity,request.json['course'])
    
    return jsonify({"recommand":result})
        
    

if __name__ == '__main__':
    app.run(debug=True)
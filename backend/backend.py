from flask import Flask, request, jsonify
import moviepy.editor as mp
from google.cloud import speech_v1p1beta1 as speech
import nltk
from textblob import TextBlob
import speech_recognition as sr
from langdetect import detect
import cv2
import os
from fer import FER
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
CORS(app, support_credentials=True)

# Initialize global emotion counters
emotion_counts = {
    "angry": 0,
    "disgust": 0,
    "fear": 0,
    "happy": 0,
    "sad": 0,
    "surprise": 0,
    "neutral": 0
}

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Create directory for storing frames
def create_data_directory():
    if not os.path.exists('data'):
        os.makedirs('data')

# Extract frames from the video
def extract_frames(video_name):
    cam = cv2.VideoCapture(video_name)
    create_data_directory()
    frame_rate = 2
    interval = int(cam.get(cv2.CAP_PROP_FPS) / frame_rate)
    currentframe = 0

    while True:
        ret, frame = cam.read()
        if ret:
            if currentframe % interval == 0:
                frame_filename = f'./data/frame{int(currentframe/14)}.jpg'
                print(f'Creating... {frame_filename}')
                cv2.imwrite(frame_filename, frame)
            currentframe += 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()
    return int(currentframe / 14 - 1)

# Detect emotions from a given frame
def detect_emotion(frame_path):
    global emotion_counts
    try:
        input_image = cv2.imread(frame_path)
        emotion_detector = FER()
        emotions = emotion_detector.detect_emotions(input_image)[0]["emotions"]
        for emotion, score in emotions.items():
            emotion_counts[emotion] += score
        print(emotions)
    except Exception as e:
        print(f'Error analyzing frame: {e}')

@app.route('/upload', methods=['POST'])
def upload():
    # Save uploaded video file
    video = request.files['file']
    video.save(video.filename)
    
    filename = video.filename
    vid = mp.VideoFileClip(filename)
    audio = vid.audio
    audio_file_name = f"{filename.split('.mp4')[0]}.wav"
    audio.write_audiofile(audio_file_name)
    print(audio_file_name)

    # Transcribe audio to text
    recognizer = sr.Recognizer()
    audioFile = sr.AudioFile(audio_file_name)
    with audioFile as source:
        data = recognizer.record(source)
    text = recognizer.recognize_google(data)
    print(text)
    
    # Analyze
    # Analyze sentiment of the text
    from nltk.sentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    print("Sentiment score is:", sentiment_scores)
    
    sentiment = "Neutral"
    if sentiment_scores["compound"] >= 0.05:
        sentiment = "Positive"
    elif sentiment_scores["compound"] <= -0.05:
        sentiment = "Negative"
    print("Sentiment:", sentiment)

    # Detect language and correct fluency if text is in English
    language = detect(text)
    if language == "en":
        blob = TextBlob(text)
        fluency = blob.correct().string
        print("Fluency:", fluency)
    else:
        print("Language is not English.")

    # Extract frames from the video
    number_of_frames = extract_frames(filename)

    # Detect emotions from each frame
    for i in range(number_of_frames):
        frame_path = f"./data/frame{i}.jpg"
        detect_emotion(frame_path)

    # Aggregate emotion scores
    emotions = [
        emotion_counts["angry"], emotion_counts["disgust"], emotion_counts["fear"],
        emotion_counts["happy"], emotion_counts["sad"], emotion_counts["surprise"],
        emotion_counts["neutral"]
    ]
    emo_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    max_emotion = emo_labels[emotions.index(max(emotions))]
    print("Dominant Emotion:", max_emotion)

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
    audio_score *= 60  # Adjust audio score scaling

    # Define the thresholds for each ranking category
    low_threshold = 70
    medium_threshold = 120

    # Determine the recommended course based on the scores
    course = ""
    if video_score > audio_score:
        if video_score <= low_threshold:
            course = 'Communication in the 21st Century Workplace'
        elif video_score <= medium_threshold:
            course = 'Communication Skills for University Success'
        else:
            course = 'Take Your English Communication Skills to the Next Level'
    else:
        if audio_score <= low_threshold:
            course = 'Introduction to Communication Science'
        elif audio_score <= medium_threshold:
            course = 'Oral Communication for Engineering Leaders'
        else:
            course = 'Business Russian Communication. Part 3'

    final_data = {
        "audio": sentiment_scores,
        "video": emotions,
        "text": text,
        "course": course
    }

    return jsonify(final_data)

# Stemming function
def stem(text):
    ps = PorterStemmer()
    return " ".join(ps.stem(word) for word in text.split())

# Course recommendation function
def recommend(new_df, similarity, course):
    course_index = new_df[new_df['Course Name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    result_list = []

    for i in course_list:
        course_name = new_df.iloc[i[0]]['Course Name']
        course_url = new_df.iloc[i[0]]['Course URL']
        course_desc = new_df.iloc[i[0]]['Course Description']
        result_list.append({
            "name": course_name,
            "url": course_url,
            "description": course_desc
        })
    
    return result_list

@app.route('/', methods=['POST'])
def result():
    data = pd.read_csv("../Essentials/Coursera.csv")

    data = data[['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Course URL']]

    # Clean and preprocess text data
    for col in ['Course Name', 'Course Description']:
        data[col] = data[col].str.replace(' ', ',').str.replace(',,', ',').str.replace(':', '').str.replace('_', '').str.replace('(', '').str.replace(')', '')

    data['Skills'] = data['Skills'].str.replace('(', '').str.replace(')', '')

    # Combine relevant columns into a single 'tags' column
    data['tags'] = data['Course Name'] + data['Difficulty Level'] + data['Course Description'] + data['Skills']
    data['tags'] = data['tags'].str.replace(',', ' ').apply(lambda x: x.lower())

    # Create a new dataframe with necessary columns
    new_df = data[['Course Name', 'tags', 'Course URL', 'Course Description']]
    new_df = new_df.rename(columns={'Course Name': 'course_name'})

    # Apply stemming
    new_df['tags'] = new_df['tags'].apply(stem)

    # Vectorize tags and compute cosine similarity
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    # Get course recommendation
    course_name = request.json['course']
    result = recommend(new_df, similarity, course_name)

    return jsonify({"recommend": result})

if __name__ == '__main__':
    app.run(debug=True)

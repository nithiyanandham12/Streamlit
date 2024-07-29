import streamlit as st
import pandas as pd
import numpy as np
import librosa
import base64
import io

def analyze_audio(file):
    # Load audio file
    y, sr = librosa.load(file, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Split the audio into 10 segments
    segment_length = len(y) // 10
    segments = [y[i*segment_length:(i+1)*segment_length] for i in range(10)]
    
    # Manually set segment durations for demonstration
    segment_durations = np.linspace(0.5, total_duration, 10)
    
    data = {
        'Speaker': [],
        'Segment Duration': [],
        'RMS Energy': [],
        'Zero Crossing Rate': [],
        'Spectral Centroid': [],
        'Spectral Bandwidth': [],
        'Sentiment Analysis': []
    }
    
    for i, segment in enumerate(segments):
        rms = librosa.feature.rms(y=segment).mean()
        zcr = librosa.feature.zero_crossing_rate(segment).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr).mean()
        
        # Mock sentiment analysis for example purposes
        sentiment = np.random.choice(['Positive', 'Neutral', 'Negative'])
        speaker = np.random.choice(['Agent', 'Customer'])
        
        data['Speaker'].append(speaker)
        data['Segment Duration'].append(segment_durations[i])
        data['RMS Energy'].append(rms)
        data['Zero Crossing Rate'].append(zcr)
        data['Spectral Centroid'].append(spectral_centroid)
        data['Spectral Bandwidth'].append(spectral_bandwidth)
        data['Sentiment Analysis'].append(sentiment)
    
    df = pd.DataFrame(data)
    return df

def get_image_as_base64(image):
    with io.BytesIO(image) as img_file:
        return base64.b64encode(img_file.read()).decode()

# Assuming the image is in the same directory as the script
# Update the path to point to your file in GitHub or local environment
logo_path = "sba_info_solutions_logo (1).jpg"

# Read the image file and encode it as base64
with open(logo_path, "rb") as img_file:
    logo_base64 = get_image_as_base64(img_file.read())

st.sidebar.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/jpeg;base64,{logo_base64}" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.title('Upload an audio file')
uploaded_file = st.sidebar.file_uploader('Choose a file', type=['wav', 'mp3'])

st.title('Audio File Analysis - SBA Info Solutions')

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    df = analyze_audio(uploaded_file)

    st.write('Analysis Results:')
    st.dataframe(df)

    # Convert DataFrame to Excel
    excel_file = 'analysis_results.xlsx'
    df.to_excel(excel_file, index=False)

    with open(excel_file, 'rb') as f:
        st.download_button('Download Excel file', f, file_name='analysis_results.xlsx')

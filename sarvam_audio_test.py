# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 13:00:43 2024

@author: TEJA
"""
import requests
import io
import base64
def generate_audio(text):
    URL = "https://api.sarvam.ai/text-to-speech"
    key= "e0d456d9-5d0d-4e45-ae0c-92c1db82b29a"
    
    

    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }

    headers = {"Content-Type": "application/json",
               "api-subscription-key": key }

    try:
        # Call the external Sarvam API
        response = requests.post(URL, json=payload, headers=headers)

        # Check if the response is successful
        if response.status_code != 200:
            print("error")

        # Extract the base64-encoded audio data from the response
        tts_response = response.json()

        # Assuming the API response has 'audios' field with base64-encoded WAV data
        if "audios" not in tts_response or not tts_response["audios"]:
            print("no audio")
            #raise HTTPException(status_code=500, detail="No audio data received from TTS API.")

        audio_base64 = tts_response["audios"][0]

        # Decode the base64 audio data
        audio_bytes = base64.b64decode(audio_base64)
        
        audio_file_path = "output_audio.wav"
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(audio_bytes)
    except Exception as e:
        print(e)
    
generate_audio("hi teja, how are you, let's discuss the package")
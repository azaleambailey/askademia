import sys
import time
from collections import deque
import io
from PIL import Image
import os
from dotenv import load_dotenv
import numpy as np
import imagehash
import av
import azure.cognitiveservices.speech as speechsdk
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import subprocess
import threading
import requests
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField
)
load_dotenv('./keys.env')

"""
Preamble: The following code handles the backend logic for a tool that addresses a problem in education using AI. During a professor's lecture, students have limited ways to address conceptual misunderstandings. 
They could raise their hand and ask the professor a question, but this method becomes problematic in a lecture hall filled with hundreds of students. Furthermore, students might be intimidated to interupt the lecturer to ask a question.
In an ideal world, each student could have a personal tutor that is an expert of the course and understands what the professor is currently talking about during lecture to answer real-time questions. Unfortunately, 
it is unlikely that colleges can provide this support. This is a problem in education that can be solved with AI. The code below formulates an AI chatbot that has access to the course notes and real time video and audio lecture data. 
First, the student question is given to multiple keyword search vector indices that retrieve the relevent lecture video, lecturer audio, and course notes context. The most recent audio and video chunks from the lecture livestream are also computed.
This information is prompt engineered and then given to an LLM to provide the student with an answer. This code currently works for local livestreams and the next goal is to have this work for livestreamed and in-person lectures.

"""

def delete_existing_index():

    """Deletes the audio and video indices from Azure if they exist."""
    
    try:
        index_client.get_index(os.getenv('AUDIO_INDEX_NAME')) #checks if there is an index with this name stored
        index_client.delete_index(os.getenv('AUDIO_INDEX_NAME')) #deletes index
    except Exception:
        pass

    try:
        index_client.get_index(os.getenv('VIDEO_INDEX_NAME')) #checks if there is an index with this name stored
        index_client.delete_index(os.getenv('VIDEO_INDEX_NAME')) #deletes index
    except Exception:
        pass

def create_index():

    """Creates or updates both a video and audio keyword search index on Azure."""

    try:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True), #unique identifier relative to moment in live lecture
            SearchableField(name="content", type=SearchFieldDataType.String), #actual data that is searchable
        ]
        audio_index = SearchIndex(name=os.getenv('AUDIO_INDEX_NAME'), fields=fields) #creates search index object
        index_client.create_or_update_index(audio_index) #creates new index if it does not exist or updates index structure if it does exist

        video_index = SearchIndex(name=os.getenv('VIDEO_INDEX_NAME'), fields=fields) #creates search index object
        index_client.create_or_update_index(video_index) #creates new index if it does not exist or updates index structure if it does exist
    except Exception:
        pass


def upload_audio_to_index(transcription, transcription_id):

    """
    Uploads audio transcription to the Azure Search index.
    
    Parameters: 
        transcription: A string containing the text transcription of an audio segment.
        transcription_id: A unique identifier (string) for the transcription document. 

    """

    try:
        with index_lock: #ensures that multiple documents can be uploaded to the same index without causing conflicts
            document = [
                {
                    "id": transcription_id,
                    "content": transcription,
                }
            ]
            audio_search_client.upload_documents(documents=document) #uploads documents to Azure Search index
    except Exception:
        pass

def upload_video_to_index(frame, frame_id):

    """
    Uploads video textual content to Azure Search index.

    Parameters: 
        frame: A string containing text extracted from a video frame (with OCR).
        frame_id: A unique identifier (string) for the frame. 
    """
    try:
        with index_lock: #ensures that multiple documents can be uploaded to the same index without causing conflicts
            document = [
                {
                    "id": frame_id,
                    "content": frame,
                }
            ]
            video_search_client.upload_documents(documents=document) #uploads documents to Azure Search index
    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        pass

def handle_transcription(evt):
    """
    Processes transcriptions generated from an audio stream, extracts words from the transcription, stores them in a buffer, and uploads the transcription to Azure Search index.
    
    Parameters:
        evt: Event object from Azure's speech recognition service.
    
    """
    global transcription_counter #tracks counter across multiple function calls
    transcription = evt.result.text #extracts the transcription text from the event.
    words = transcription.split() #splits the transcription into individual words (by spaces)
    with index_lock:  #ensures that multiple items can be uploaded to the same index without causing conflicts
        word_buffer.extend(words) #appends words in transcript to the total of words spoken so far
    upload_audio_to_index(transcription, f"transcription-{transcription_counter}") #uploads audio transcription to Azure Search index
    transcription_counter += 1


def generate_gpt_response(prompt):

    """
    Sends a prompt to a GPT LLM hosted on Azure and retrieves the model's response.
    
    Parameters:
        prompt: A list of messages (that are dictionaries) forming the conversation history and user input.

    Returns: GPT LLM response (string).

    """

    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv('OPENAI_API_KEY')
    }
    payload = {
        "messages": prompt,
        "temperature": 0.1,
        "top_p": 0.95,
    }
    response = requests.post(os.getenv('OPENAI_ENDPOINT'), headers=headers, json=payload) #sends prompt to LLM
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'] #extracts and returns the model's response

def summarize_conversation(history):

    """
    Takes a conversation history and generates a summary of it using a GPT LLM.
    
    Parameters:
        history: String containing the conversation to be summarized.
    
    Returns: Summary of inputted conversation (string).

    """

    summary_prompt = [
        {"role": "system", "content": "You are a summarization assistant."},
        {"role": "user", "content": f"Summarize the following conversation:\n\n{history}"}
    ]
    try:
        return generate_gpt_response(summary_prompt) #prompts LLM to summarize conversation
    except Exception as e:
        print(f"Error generating conversation summary: {e}")
        return ""

def retrieve_top_search_results(query):

    """
    Queries multiple Azure Search indices (audio, notes, and video) and retrieves the top search results for a given query. Performs keyword-based searches across these indices and return the most relevant content.

    Parameters:
        query: String containing the search term or query.
    
    Returns: A list of the most relevent audio context segments, list of the most relevent course note segments, and a list of the most relevent video segments (tuple of 3 lists of strings).

    """

    try:
        audio_results = audio_search_client.search(query, top=3) #retrieves the top 3 most relevent audio segments
        notes_results = notes_search_client.search(query, top=3) #retrieves the top 3 most relevent course note segments
        video_results = video_search_client.search(query, top=2) #retrieves the top 2 most relevent video segments
        return [result['content'] for result in audio_results], [result['content'] for result in notes_results], [result['content'] for result in video_results]
    except Exception as e:
        return [], [], []

def image_to_stream(pil_image):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def perform_ocr(frame_pil):

    """
    Uses Azure Cognitive Services' Computer Vision API to perform Optical Character Recognition (OCR) on a given image.
    
    Parameters:
        frame_pil: Image in Pillow format (PIL.Image object).

    Returns: Extracted OCR text from given image (string).

    """

    image_stream = image_to_stream(frame_pil) #image is converted into a binary stream
    read_response = computervision_client.read_in_stream(image_stream, raw=True) #image stream is sent to Azure Computer Vision's Read API for OCR processing
    read_operation_location = read_response.headers["Operation-Location"] #gets URL for operation status
    operation_id = read_operation_location.split("/")[-1] #extracts the operation ID from this URL 

    while True: #exits loop once OCR is complete or failed
        read_result = computervision_client.get_read_result(operation_id) #gets status of operation
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1) #adds time between calls to avoid reaching request limit
    extracted_text = ""
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results: #concatenates the entire result into a string
            for line in text_result.lines:
                extracted_text += line.text + "\n"
    return extracted_text

def process_video_stream(stream_url, hash_threshold=2, n=5):

    """
    Processes a video stream frame by frame, extracts text from unique frames using OCR, and uploads the extracted text to an Azure Search index.
    
    Parameters:
        stream_url: The URL of the video stream to be processed.
        hash_threshold: Threshold value used to determine whether two frames are considered similar.
        n: Maximum number of unique frame hashes to keep in memory.

    """

    saved_frame_hashes = [] #stores perceptual hashes of processed frames
    try:
        container = av.open(stream_url) #opens stream and prepares it for decoding
        counter = 0
        for frame in container.decode(video=0): #iterates through all the video frames from a live stream
            counter += 1
            if counter % 120 == 0: #processes a frame every 4 seconds
                frame_array = frame.to_ndarray(format="rgb24") #converts video frame to an array of pixel data
                frame_pil = Image.fromarray(frame_array) #converts frame to PIL format
                frame_hash = imagehash.phash(frame_pil) #calculates perceptual hash to determine if the current frame is similar to previously processed frames
                is_unique = True

                for saved_hash in saved_frame_hashes:
                    if abs(frame_hash - saved_hash) < hash_threshold: #if the difference between hashes is less than hash_threshold then the frame is not unique
                        is_unique = False
                        break

                if is_unique:
                    extracted_text = perform_ocr(frame_pil) #extracts text from the frame using OCR
                    upload_video_to_index(extracted_text, f"frame-{counter}") #uploads the extracted text to Azure Search index
                    saved_frame_hashes.append(frame_hash) #adds the frame’s hash to saved_frame_hashes
                    video_buffer.append(extracted_text) #appends extracted text to entire OCR text from video stream

                    if len(saved_frame_hashes) > n: #limits the comparison of perceptual hashes to only the most recent n video frames
                        saved_frame_hashes.pop(0)
    except av.AVError as e:
        print(f"Failed to process stream: {e}")


def process_audio_stream():

    """Processes an audio stream by reading chunks of audio data from a live feed and writing it to a push audio stream to be uploaded to an Azure Search index in the future."""

    try:
        while not stop_event.is_set():
            chunk = ffmpeg_proc.stdout.read(4096) #continuously reads audio data in chunks of 4096 bytes
            if not chunk:
                break
            push_stream.write(chunk) #writes chunk to push_stream
    except KeyboardInterrupt:
        stop_event.set()
    finally: #releases all resources used during audio processing
        recognizer.stop_continuous_recognition() #stops the speech recognition process
        push_stream.close() #closes the push audio stream to free resources
        ffmpeg_proc.terminate() #stops the FFmpeg process that is providing the audio data

def user_input_thread(user_input=None):

    """
    Handles user interactions with the Askademia bot. It processes user input, retrieves relevant contextual information, formats a prompt for the LLM model, and generates a response.

    Parameters:
        user_input: A string containing the user's question.
    
    """

    print('=' * 50)

    if not user_input: #initially encourages the user to ask a question
        question = input("Enter your question (or 'new' to start a new conversation, 'exit' to quit): ").strip()
    else:
        question = user_input

    with index_lock: #ensures safe access to shared resources (word_buffer and video_buffer) in this multi-threaded environment
        recent_audio = " ".join(word_buffer)  #combines the rolling window of recent audio transcriptions (word_buffer) into a single string
        recent_video = "\n".join(video_buffer) #combines the rolling window of recent video OCR results (video_buffer) into a single string

    formatted_history = "\t"+"\n\t".join( #formats recent conversation history into a readable string for inclusion in the prompt
        f'User: "{h["user"]}"\n\tAssistant: "{h["assistant"]}"' for h in list(conversation_history)
    )

    audio_vector, notes_vector, video_vector = retrieve_top_search_results(question) #searches the audio, notes, and video indices for context relevant to the user's question
    audio_vector, notes_vector, video_vector = '\n'.join(audio_vector), '\n'.join(notes_vector), '\n'.join(video_vector) #combines the retrieved results into single strings for inclusion in the prompt

    system_message = (
        "You are an AI assistant helping students understand lectures. "
        "Please use the provided context from the lecture to answer the student's question. "
        "Any 'video' context refers to information displayed on the screen. This is important. "
        "Please answer the question in the context of the conversation history provided, if any. "
        "Please limit your response to a maximum of 4 sentences."
    )

    formatted_prompt = (
        "Conversation History:\n"
        f"{formatted_history}"
        "\n\n"
        "Retrieved Notes:\n"
        f"{notes_vector}"
        "\n\n"
        "Retrieved Audio:\n"
        f"{audio_vector}"
        "\n\n"
        "Retrieved Video:\n"
        f"{video_vector}"
        "\n\n"
        "Recent Audio:\n"
        f"{recent_audio}"
        "\n\n"
        "Recent Video:\n"
        f"{recent_video}"
    )

    prompt = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context: {formatted_prompt}\n\nQuestion: {question}"},
    ]

    print(formatted_prompt)

    return generate_gpt_response(prompt)


speech_config = speechsdk.SpeechConfig(subscription=os.getenv('speech_key'), region=os.getenv('service_region'))

EMBEDDING_MODEL_DIMENSIONS = 1536
index_client = SearchIndexClient(os.getenv('SEARCH_ENDPOINT'), AzureKeyCredential(os.getenv('SEARCH_KEY')))

computervision_client = ComputerVisionClient(os.getenv('endpoint'), CognitiveServicesCredentials(os.getenv('subscription_key')))
audio_search_client = SearchClient(os.getenv('SEARCH_ENDPOINT'), os.getenv('AUDIO_INDEX_NAME'), AzureKeyCredential(os.getenv('SEARCH_KEY')))
notes_search_client = SearchClient(os.getenv('SEARCH_ENDPOINT'), os.getenv('NOTES_INDEX_NAME'), AzureKeyCredential(os.getenv('SEARCH_KEY')))
video_search_client = SearchClient(endpoint=os.getenv('SEARCH_ENDPOINT'), index_name=os.getenv('VIDEO_INDEX_NAME'),
                                   credential=AzureKeyCredential(os.getenv('SEARCH_KEY')))

#initializes storage devices
index_lock = threading.Lock() #for multiple threading
word_buffer = deque(maxlen=256) #stores rolling audio transcription
video_buffer = deque(maxlen=2) #stores rolling video frame OCR
conversation_history = deque(maxlen=5) #chatbot and user conversation history

#initializes Azure Search indices for live lecture
delete_existing_index()
create_index()

#initializes audio transcription streaming devices
push_stream = speechsdk.audio.PushAudioInputStream()
audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

#streaming info
ffmpeg_cmd = [
    "ffmpeg",
    "-i", "rtmp://localhost/live/my_stream",
    "-vn",
    "-ar", "16000",
    "-ac", "1",
    "-f", "wav",
    "pipe:1"
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

transcription_counter = 1

recognizer.recognized.connect(handle_transcription)
stop_event = threading.Event()

recognizer.start_continuous_recognition()

stream_url = "rtmp://localhost/live/my_stream"
saved_frame_hashes = []
hash_threshold = 2
n = 5

video_thread = threading.Thread(target=process_video_stream, args=(stream_url,), daemon=True)
audio_thread = threading.Thread(target=process_audio_stream, daemon=True)

video_thread.start()
audio_thread.start()

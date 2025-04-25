# import logging
# import pickle
# import string
# import sys
# import tempfile
# import os
# import gradio as gr
# import whisper
# from huggingface_hub.hf_api import HfFolder
# from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
# from llama_index.embeddings.fastembed import FastEmbedEmbedding
# from llama_index.llms.huggingface import HuggingFaceLLM
# from moviepy.editor import VideoFileClip

# # Function to process multiple video files
# def process_video_files(video_dir_path):
#     # Get a list of all the files in the directory
#     video_files = os.listdir(video_dir_path)

#     # Loop over each file in the directory
#     for video_file in video_files:
#         # Construct the full path to the video file
#         video_file_path = os.path.join(video_dir_path, video_file)

#         # Open the video file
#         video = VideoFileClip(video_file_path)

#         # Extract audio from the video
#         audio = video.audio

#         # Convert the audio to a WAV file
#         audio_file_path = f"audio_{video_file}.wav"
#         audio.write_audiofile(audio_file_path)

#         # Perform speech recognition
#         with open("model.pkl", "rb") as f:
#             model = pickle.load(f)
#         results = model.transcribe(audio_file_path, fp16=False)

#         # Define the path to the directory
#         directory = f"recognized_text_dir_"

#         # Create the directory if it doesn't exist
#         os.makedirs(directory, exist_ok=True)

#         # Define the path to the text file within the directory
#         text_file_path = os.path.join(directory, f"recognized_text_{video_file}.txt")

#         # Save the recognized text to a .txt file
#         with open(text_file_path, "w") as f:
#             f.write(results["text"])

#         # Read the text from the .txt file
#         with open(text_file_path, "r") as f:
#             text = f.read()

#         documents = SimpleDirectoryReader(input_dir=directory).load_data()

#         embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
#         Settings.embed_model = embed_model
#         Settings.chunk_size = 512

#         system_prompt = "You are a Q&A assistant. You will be provided some transcripts from YouTube videos. Please answer the query based on what is said in the videos"
#         query_wrapper_prompt = PromptTemplate("{query_str}")
#         # "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
#         HfFolder.save_token('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')

#         llm = HuggingFaceLLM(
#             context_window=8192,
#             max_new_tokens=256,
#             generate_kwargs={"temperature": 0.7, "do_sample": False},
#             system_prompt=system_prompt,
#             query_wrapper_prompt=query_wrapper_prompt,
#             tokenizer_name="google/gemma-2b-it",
#             model_name="google/gemma-2b-it",
#             device_map="cpu",
#             tokenizer_kwargs={"max_length": 4096})

#         Settings.llm = llm
#         Settings.chunk_size = 512

#         index = VectorStoreIndex.from_documents(documents)
#         print(index)
#         query_engine = index.as_query_engine()

#         def predict(input, history):
#             response = query_engine.query(input)
#             return str(response)

#     gr.ChatInterface(predict).launch(share=True)

# # Define the path to the video directory
# video_dir_path = r"C:\Users\CVHS\Desktop\Projects\gemma\videos"

# # Call the function to process the video files
# process_video_files(video_dir_path)




import os
import gradio as gr
from huggingface_hub.hf_api import HfFolder
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

def process_text_file(text_file_path):
    # Read the text from the input file
    with open(text_file_path, "r") as f:
        text = f.read()

    documents = [text]

    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    system_prompt = "You are a Q&A assistant. You will be provided with some context. Please answer the query based on the given context."
    query_wrapper_prompt = PromptTemplate("{query_str}")
    HfFolder.save_token('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')

    llm = HuggingFaceLLM(
        context_window=8192,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="google/gemma-2b-it",
        model_name="google/gemma-2b-it",
        device_map="cpu",
        tokenizer_kwargs={"max_length": 4096})

    Settings.llm = llm
    Settings.chunk_size = 512

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    def predict(input, history):
        response = query_engine.query(input)
        return str(response)

    gr.ChatInterface(predict).launch(share=True)

# Define the path to the input text file
text_file_path = r"C:\Users\CVHS\pavan\Gemini\output.txt"

# Call the function to process the text file
process_text_file(text_file_path)











# #GRADIO
# import logging
# import pickle
# import sys
# import os
# import gradio as gr
# from huggingface_hub.hf_api import HfFolder
# from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
# from llama_index.embeddings.fastembed import FastEmbedEmbedding
# from llama_index.llms.huggingface import HuggingFaceLLM
# from moviepy.editor import VideoFileClip

# # Define the path to the video file
# video_file_path = r"C:\Users\CVHS\Desktop\Projects\gemma\videos"

# # Open the video file
# video = VideoFileClip(video_file_path)

# # Extract audio from the video
# audio = video.audio

# # Convert the audio to a WAV file
# audio_file_path = "audio.wav"
# audio.write_audiofile(audio_file_path)

# # # Load the whisper model
# # model = whisper.load_model("base")

# # Perform speech recognition
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)
# results = model.transcribe(audio_file_path, fp16=False)

# # Define the path to the directory
# directory = "recognized_text_dir"

# # Create the directory if it doesn't exist
# os.makedirs(directory, exist_ok=True)

# # Define the path to the text file within the directory
# text_file_path = os.path.join(directory, "recognized_text.txt")

# # Save the recognized text to a .txt file
# with open(text_file_path, "w") as f:
#     f.write(results["text"])

# # Read the text from the .txt file
# with open(text_file_path, "r") as f:
#     text = f.read()


# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# documents = SimpleDirectoryReader(input_dir=r"C:\Users\CVHS\Desktop\Projects\gemma\recognized_text_dir").load_data()

# embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = embed_model
# Settings.chunk_size = 512

# system_prompt = "You are a Q&A assistant. You will be provided some transcripts from YouTube videos. Please answer the query based on what is said in the videos"
# query_wrapper_prompt = PromptTemplate("{query_str}")
# # "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
# HfFolder.save_token('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')

# llm = HuggingFaceLLM(
#     context_window=8192,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "do_sample": False},
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="google/gemma-2b-it",
#     model_name="google/gemma-2b-it",
#     device_map="cpu",
#     tokenizer_kwargs={"max_length": 4096})

# Settings.llm = llm
# Settings.chunk_size = 512

# index = VectorStoreIndex.from_documents(documents)
# print(index)
# query_engine = index.as_query_engine()


# def predict(input, history):
#     response = query_engine.query(input)
#     return str(response)

# gr.ChatInterface(predict).launch(share=True)



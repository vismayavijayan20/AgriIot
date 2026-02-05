
import os
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyAI7tik6xg8YYO_l2pkX5857NtGcepVwPU')

if GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY':
    print("Please set your API key in the script or environment variables.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Listing available models...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Error: {e}")

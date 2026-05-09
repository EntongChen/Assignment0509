import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import re

# 1. Model Caching 
@st.cache_resource
def load_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    # Using SmolLM: Intelligent, logical, and fast on CPU
    return pipeline("text-generation", model="HuggingFaceTB/SmolLM-360M-Instruct")

@st.cache_resource
def load_audio_model():
    return pipeline("text-to-speech", model="Matthijs/mms-tts-eng")

# 2. Functional Part

# img2text
def img2text(url):
    """Converts the uploaded image into a text description."""
    image_to_text_model = load_img2text_model()
    if not isinstance(url, str):
        url = Image.open(url)
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    """Generates a 50-100 word story based on the image description."""
    story_pipe = load_story_model()
    
    # Prompt optimized for children using Instruct format
    prompt = (
        f"<|user|>\n"
        f"Write a fun, magical 60-word story for a 7-year-old kid based ONLY on: {text}. "
        f"Start with 'Once upon a time'.\n"
        f"<|assistant|>\n"
        f"Once upon a time, "
    )
    
    story_results = story_pipe(
        prompt, 
        min_new_tokens=60, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.7,
        top_p=0.9,
        eos_token_id=story_pipe.tokenizer.eos_token_id,
        pad_token_id=story_pipe.tokenizer.eos_token_id
    )
    
    full_text = story_results[0]['generated_text']
    story = full_text.split("<|assistant|>")[-1].strip() if "<|assistant|>" in full_text else full_text.strip()
    
    # Smart truncation to ensure the story ends with a full sentence
    last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
    if last_punc != -1:
        story = story[:last_punc + 1]
    return story

# text2audio
def text2audio(story_text):
    """Converts the story text into audio data."""
    audio_pipe = load_audio_model()
    # Basic cleaning: remove newlines and special characters
    clean_text = re.sub(r'[^a-zA-Z0-9\s,.!?\']', '', story_text.replace("\n", " "))
    # Truncate to 450 characters to prevent VITS model from crashing
    safe_text = clean_text[:450]
    return audio_pipe(safe_text)

# 3. Main Part (including UI design)

def main():
    # Set page title and icon
    st.set_page_config(page_title="Magic Storybook", page_icon="✨")

    # Custom CSS for a colorful and friendly look
    st.markdown("""
        <style>
        .main { background-color: #f0f2f6; }
        .stHeader { color: #ff4b4b; }
        .story-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #ff4b4b;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            color: #31333F;
            font-size: 18px;
            line-height: 1.6;
        }
        </style>
    """, unsafe_allow_html=True)

    # Top Title and Subheader
    st.title("✨ My Magic Storybook ✨")
    st.subheader("🎨 Turn your pictures into wonderful audio stories!")
    
    # Sidebar instructions for parents
    with st.sidebar:
        st.header("Parent's Guide 👨‍👩‍👧")
        st.write("1. Upload a clear photo.")
        st.write("2. Wait for the AI magic to happen.")
        st.write("3. Listen to the story together!")

    # Upload Area
    uploaded_file = st.file_uploader("📸 Choose a picture to start the magic...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Layout: Image on the left, processing/story on the right
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Your Magic Picture", use_container_width=True)

        with col2:
            # Use Status component to show the "Magic Process"
            with st.status("🪄 Casting magic spells...", expanded=True) as status:
                st.write("👀 Looking at your picture...")
                scenario = img2text(uploaded_file)
                
                st.write("✍️ Writing a special story...")
                story = text2story(scenario)
                
                st.write("🎙️ Preparing the storyteller's voice...")
                audio_result = text2audio(story)
                
                status.update(label="✅ Magic Complete!", state="complete", expanded=False)

            # Display the Story Card
            st.markdown("### 📖 Your Story")
            st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)
            
            # Audio Playback
            st.markdown("### 🎧 Listen Now")
            st.audio(audio_result["audio"], sample_rate=audio_result["sampling_rate"])
            
            # Success Effect
            st.balloons()

if __name__ == "__main__":
    main()

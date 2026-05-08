import streamlit as st
from transformers import pipeline

# --- Function Part (带缓存且完整实现) ---

@st.cache_resource
def get_img2text_pipe():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def get_story_pipe():
    # 也可以用 gpt2，如果这个模型在云端太慢的话
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

@st.cache_resource
def get_audio_pipe():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

def img2text(uploaded_file):
    pipe = get_img2text_pipe()
    text = pipe(uploaded_file)[0]["generated_text"]
    return text

def text2story(scenario):
    pipe = get_story_pipe()
    # 优化 Prompt，让它更像给小孩讲故事
    prompt = f"Write a fun and simple story for a 5-year-old kid based on this: {scenario}. The story should be around 60 words."
    # 控制长度在 50-100 词左右
    story_results = pipe(prompt, max_new_tokens=150, min_new_tokens=60, do_sample=True)
    return story_results[0]['generated_text']

def text2audio(story_text):
    pipe = get_audio_pipe()
    return pipe(story_text)

# --- Main Part ---
st.set_page_config(page_title="Kids Storyteller", page_icon="🦜")
st.header("Turn Your Image to Audio Story")
uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 显示图片
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Stage 1: Image to Text
    with st.spinner('Processing image...'):
        scenario = img2text(uploaded_file)
        st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story
    with st.spinner('Generating story...'):
        story = text2story(scenario)
        st.subheader("The Story")
        st.write(story)

    # Stage 3: Story to Audio
    with st.spinner('Generating audio...'):
        audio_data = text2audio(story)
        # 直接显示播放器，用户体验更好
        st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])

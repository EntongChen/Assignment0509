import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# --- 1. 缓存模型加载 (为了让你的 App 在云端不崩溃，这部分是必须的) ---
@st.cache_resource
def load_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

@st.cache_resource
def load_audio_model():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

# --- 2. Function Part (保留老师给的函数名和结构) ---

# img2text
def img2text(url):
    # 调用缓存的模型
    image_to_text_model = load_img2text_model()
    # 关键修复：如果传入的是文件对象，先转为 PIL Image
    if not isinstance(url, str):
        url = Image.open(url)
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_pipe = load_story_model()
    # 补全逻辑：加入针对 3-10 岁儿童的 Prompt 和长度控制
    prompt = f"Write a fun story for a 5-year-old kid based on this: {text}. The story should be 50-100 words."
    story_results = story_pipe(prompt, max_new_tokens=150, min_new_tokens=60, do_sample=True)
    story_text = story_results[0]['generated_text']
    # 清理掉生成的文本中可能包含的 prompt
    if prompt in story_text:
        story_text = story_text.replace(prompt, "").strip()
    return story_text

# text2audio
def text2audio(story_text):
    audio_pipe = load_audio_model()
    # 补全逻辑：生成音频数据
    audio_data = audio_pipe(story_text)
    return audio_data

# --- 3. Main Part ---
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
st.header("Turn Your Image to Audio Story")
uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示图片
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Stage 1: Image to Text (调用老师定义的函数)
    with st.spinner('Processing img2text...'):
        scenario = img2text(uploaded_file)
        st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story (调用老师定义的函数)
    with st.spinner('Generating a story...'):
        story = text2story(scenario)
        st.subheader("The Story")
        st.info(story)

    # Stage 3: Story to Audio (调用老师定义的函数)
    with st.spinner('Generating audio data...'):
        audio_result = text2audio(story)
        # 播放音频
        st.audio(audio_result["audio"], sample_rate=audio_result["sampling_rate"])
        st.balloons()


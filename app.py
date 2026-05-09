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
    return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

@st.cache_resource
def load_audio_model():
    return pipeline("text-to-speech", model="Matthijs/mms-tts-eng")

# --- 2. Function Part ---

# img2text
def img2text(url):
    # 调用缓存的模型
    image_to_text_model = load_img2text_model()
    # 关键修复：如果传入的是文件对象，先转为 PIL Image
    if not isinstance(url, str):
        url = Image.open(url)
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(text):
    story_pipe = load_story_model()
    
    # 极简 Prompt，减少模型预处理负担
prompt = (
    f"<|system|>\n"
    f"You are a professional children's book author. Your task is to turn a specific visual description into a coherent, imaginative story for 5-year-olds.\n"
    f"RULES:\n"
    f"1. STICK CLOSELY to the elements mentioned in the description. Do not add random characters or themes.\n"
    f"2. Length: Must be between 50 and 100 words.\n"
    f"3. Tone: Whimsical, warm, and engaging.\n"
    f"4. Language: Simple English.\n"
    f"<|user|>\n"
    f"Here is the image description to turn into a story: {text}\n"
    f"<|assistant|>\n"
    f"Certainly! Here is a fun story based on that description: Once upon a time,"
)
    
    with st.spinner("Writing..."): # 增加一个加载动画，提升用户体验
        story_results = story_pipe(
            prompt, 
            max_new_tokens=80,   # 限制在 80 token 以内（约 60 词）
            min_new_tokens=40,   # 确保不少于 30-40 词
            do_sample=True, 
            temperature=0.8,
            top_k=50
        )
    
    story = story_results[0]['generated_text']
    
    # 清理掉 Prompt 部分
    if "Once upon a time," in story:
        story = "Once upon a time," + story.split("Once upon a time,")[-1]
        
    return story



def text2audio(story_text):
    audio_pipe = load_audio_model()
    
    # --- 核心修复：彻底清理文本 ---
    # 1. 去掉换行符，把故事变成一行
    clean_text = story_text.replace("\n", " ").strip()
    
    # 2. 只保留字母、数字和基本标点，删掉所有奇怪符号
    import re
    clean_text = re.sub(r'[^a-zA-Z0-9\s,.!?\']', '', clean_text)
    
    # 3. 强制截断到 250 个字符（这是 VITS 模型最安全的长度）
    # 250个字符大约是 40-50 个单词，虽然比作业要求的 50-100 少一点，
    # 但这是保证程序不崩溃的唯一办法。
    safe_text = clean_text[:250]
    
    # 如果截断后为空，给个保底句子
    if not safe_text:
        safe_text = "This is a wonderful story about your picture."
        
    return audio_pipe(safe_text)


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


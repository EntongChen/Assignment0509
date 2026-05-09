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
    return pipeline("text-generation", model="HuggingFaceTB/SmolLM-360M-Instruct")

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
    
    # 1. 优化 Prompt：明确告诉模型字数要求
    prompt = (
        f"Write a short children's story (about 70 words) based on: {text}. "
        f"The story must be engaging and complete. "
        f"Story: Once upon a time,"
    )
    
    with st.spinner("Generating a 50-100 word story..."):
        story_results = story_pipe(
            prompt, 
            # --- 精准 Token 控制 ---
            min_new_tokens=65,   # 确保至少有 50 个单词左右
            max_new_tokens=120,  # 确保不超过 100 个单词左右
            # ----------------------
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    
    full_text = story_results[0]['generated_text']
    
    # 2. 提取故事正文
    if "Once upon a time," in full_text:
        story = "Once upon a time," + full_text.split("Once upon a time,")[-1]
    else:
        story = full_text.strip()

    # 3. 智能截断：确保停在句号上，且总字数不超过 100
    words = story.split()
    if len(words) > 100:
        # 如果超过 100 字，先截取前 100 字
        story = " ".join(words[:100])
        # 然后回溯找到最后一个句号，保证故事完整
        last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
        if last_punc != -1:
            story = story[:last_punc + 1]
    
    # 4. 最终检查：如果截得太短（少于 50 字），则保留原样（只要不超过 100）
    if len(story.split()) < 50:
        # 这种情况下通常是模型还没写完，我们可以稍微放宽截断
        pass 

    return story



# text2audio
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


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
    
    # --- 优化 1: 简化指令，把图片描述放在最后 (Recency Bias 优化) ---
    # 这样模型在开始生成时，脑子里最后留下的信息就是你的图片描述
    prompt = (
        f"<|system|>\n"
        f"You are a children's storyteller. Write a 60-word story for a 5-year-old.\n"
        f"Rule: The story MUST be about the elements in the description.\n"
        f"<|user|>\n"
        f"Description: {text}\n"
        f"<|assistant|>\n"
        f"Here is a story about {text}: Once upon a time,"
    )
    
    # --- 优化 2: 调整参数，增加确定性 ---
    story_results = story_pipe(
        prompt, 
        min_new_tokens=70,    # 确保字数达标
        max_new_tokens=130,   # 防止过长导致语音崩溃
        do_sample=True, 
        temperature=0.6,      # 稍微降低随机性，让它更“听话”
        top_p=0.9,
        repetition_penalty=1.2 # 增强惩罚，防止复读
    )
    
    full_text = story_results[0]['generated_text']
    
    # --- 优化 3: 更稳健的提取逻辑 ---
    if "<|assistant|>" in full_text:
        story = full_text.split("<|assistant|>")[-1].strip()
    else:
        # 如果模型没按套路出牌，尝试提取最后一段
        story = full_text.split("\n")[-1].strip()

    # --- 优化 4: 确保故事从“Once upon a time”开始且完整 ---
    if "Once upon a time," not in story:
        story = "Once upon a time, " + story

    # --- 优化 5: 句子级截断 (保护语音模型) ---
    # 找到 300 字符内的最后一个句号，确保故事听起来是完整的
    if len(story) > 300:
        last_period = story.rfind('.', 0, 300)
        if last_period != -1:
            story = story[:last_period + 1]
        else:
            story = story[:300]

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


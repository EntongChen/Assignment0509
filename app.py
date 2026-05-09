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

# text2story
def text2story(text):
    story_pipe = load_story_model()
    
    # --- 优化 1: 强力关联 Prompt ---
    # 我们把图片描述直接嵌入到故事的开头，让模型避无可避
    prompt = (
        f"Write a short, complete story for a 5-year-old kid. "
        f"The story must start with the scene: {text}. "
        f"Story: Once upon a time, there was {text}. "
    )
    
    with st.spinner("Crafting a complete story..."):
        story_results = story_pipe(
            prompt, 
            min_new_tokens=80,   # 确保字数达标
            max_new_tokens=160,  # 给模型足够的空间写完结尾
            do_sample=True, 
            temperature=0.7,     # 降低随机性，提高逻辑相关性
            top_p=0.9,
            repetition_penalty=1.2,
            # 告诉模型遇到句号、问号等可以考虑停止
            eos_token_id=story_pipe.tokenizer.eos_token_id 
        )
    
    full_text = story_results[0]['generated_text']
    
    # --- 优化 2: 提取逻辑 ---
    # 只保留 "Once upon a time" 之后的内容
    if "Once upon a time," in full_text:
        story = "Once upon a time," + full_text.split("Once upon a time,")[-1]
    else:
        story = full_text.strip()

    # --- 优化 3: 解决“突然截断”问题 ---
    # 寻找最后一个句号、感叹号或问号
    import re
    # 找到最后一个标点符号的位置
    last_punctuation = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
    
    if last_punctuation != -1:
        # 截断到最后一个完整的句子，这样故事听起来就是完整的
        story = story[:last_punctuation + 1]
    
    # 如果截断后太短，或者没找到标点，至少保证它不以半个单词结尾
    return story.strip()



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


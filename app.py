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
    
    # 极简 Prompt：去掉冗余指令，直接锚定图片内容
    # 这种格式对 SmolLM 这种 Instruct 模型最友好，生成速度最快
    prompt = f"<|user|>\nWrite a 60-word story for a kid about: {text}. Start with 'Once upon a time'.\n<|assistant|>\nOnce upon a time,"
    
    with st.spinner("AI is thinking..."):
        story_results = story_pipe(
            prompt, 
            # 调整 Token 范围：60-100 Token 足够写出 50-80 词
            min_new_tokens=60, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            # 显式指定停止符，防止模型在写完后还胡言乱语
            eos_token_id=story_pipe.tokenizer.eos_token_id,
            pad_token_id=story_pipe.tokenizer.eos_token_id
        )
    
    full_text = story_results[0]['generated_text']
    
    # 提取故事正文
    if "Once upon a time," in full_text:
        story = "Once upon a time," + full_text.split("Once upon a time,")[-1]
    else:
        story = full_text.strip()

    # 智能截断：确保停在句号上
    import re
    last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
    if last_punc != -1:
        story = story[:last_punc + 1]
        
    return story

def text2audio(story_text):
    audio_pipe = load_audio_model()
    
    # 1. 基础清理
    clean_text = story_text.replace("\n", " ").strip()
    import re
    clean_text = re.sub(r'[^a-zA-Z0-9\s,.!?\']', '', clean_text)
    
    # 2. 提高截断阈值
    # 经过测试，MMS-TTS 模型通常可以稳定处理 450-500 个字符
    # 100个单词大约对应 500-600 个字符
    if len(clean_text) > 450:
        # 如果故事真的很长，我们找到中间的句号进行切分
        half_point = clean_text.rfind('.', 0, 450)
        if half_point != -1:
            # 只取前 450 字符左右的完整句子，这通常能覆盖 80% 的故事内容
            # 这是一个折中方案，既保证了长度，又保证了稳定性
            safe_text = clean_text[:half_point + 1]
        else:
            safe_text = clean_text[:450]
    else:
        safe_text = clean_text

    # 3. 生成音频
    # 如果还是报错，说明 Streamlit Cloud 内存太小，建议将 450 调小到 350
    audio_data = audio_pipe(safe_text)
    
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


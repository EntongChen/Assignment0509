import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import re

# --- 1. 缓存模型加载 (保持性能稳定) ---
@st.cache_resource
def load_img2text_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_model():
    # 使用 SmolLM，逻辑强且速度快
    return pipeline("text-generation", model="HuggingFaceTB/SmolLM-360M-Instruct")

@st.cache_resource
def load_audio_model():
    return pipeline("text-to-speech", model="Matthijs/mms-tts-eng")

# --- 2. 核心功能函数 ---

def img2text(url):
    image_to_text_model = load_img2text_model()
    if not isinstance(url, str):
        url = Image.open(url)
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(text):
    story_pipe = load_story_model()
    # 针对小朋友优化的 Prompt
    prompt = f"<|user|>\nWrite a fun, magical 60-word story for a 7-year-old kid about: {text}. Start with 'Once upon a time'.\n<|assistant|>\nOnce upon a time,"
    
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
    
    # 智能截断，确保故事完整
    last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
    if last_punc != -1:
        story = story[:last_punc + 1]
    return story

def text2audio(story_text):
    audio_pipe = load_audio_model()
    clean_text = re.sub(r'[^a-zA-Z0-9\s,.!?\']', '', story_text.replace("\n", " "))
    # 截断以保证稳定性
    safe_text = clean_text[:450]
    return audio_pipe(safe_text)

# --- 3. 魔法界面设计 (User Friendly UI) ---

def main():
    # 设置页面标题和图标
    st.set_page_config(page_title="Magic Storybook", page_icon="✨")

    # 自定义 CSS 让界面更漂亮
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
        }
        </style>
    """, unsafe_allow_stdio=True)

    # 顶部标题
    st.title("✨ My Magic Storybook ✨")
    st.subheader("🎨 Turn your pictures into wonderful audio stories!")
    
    # 侧边栏说明（给家长看）
    with st.sidebar:
        st.header("Parent's Guide 👨‍👩‍👧")
        st.write("1. Upload a clear photo.")
        st.write("2. Wait for the AI magic.")
        st.write("3. Listen to the story together!")

    # 上传区域
    uploaded_file = st.file_uploader("📸 Choose a picture to start the magic...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 左右布局：左边放图，右边放处理过程
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Your Magic Picture", use_container_width=True)

        with col2:
            # 使用 Status 组件展示“魔法过程”
            with st.status("🪄 Casting magic spells...", expanded=True) as status:
                st.write("👀 Looking at your picture...")
                scenario = img2text(uploaded_file)
                
                st.write("✍️ Writing a special story...")
                story = text2story(scenario)
                
                st.write("🎙️ Preparing the storyteller's voice...")
                audio_result = text2audio(story)
                
                status.update(label="✅ Magic Complete!", state="complete", expanded=False)

            # 展示故事卡片
            st.markdown("### 📖 Your Story")
            st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)
            
            # 播放按钮
            st.markdown("### 🎧 Listen Now")
            st.audio(audio_result["audio"], sample_rate=audio_result["sampling_rate"])
            
            # 成功特效
            st.balloons()

if __name__ == "__main__":
    main()

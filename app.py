import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# --- 1. 缓存模型加载（自动使用 GPU 如果可用）---
@st.cache_resource
def load_img2text_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=device)

@st.cache_resource
def load_story_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-generation", 
        model="HuggingFaceTB/SmolLM-360M-Instruct",
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # GPU 半精度加速
    )

@st.cache_resource
def load_audio_model():
    # TTS 模型通常 CPU 就够了，也可以指定设备
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-to-speech", model="Matthijs/mms-tts-eng", device=device)

# --- 2. Function Part ---

def img2text(url):
    image_to_text_model = load_img2text_model()
    if not isinstance(url, str):
        url = Image.open(url)
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(text):
    story_pipe = load_story_model()
    
    # 优化后 Prompt：要求更短的故事（约 40~50 个单词）
    prompt = (
        f"Write a very short children's story (about 40-50 words) based on: {text}. "
        f"The story must be engaging and complete. "
        f"Use 4 sentences at most. Begin with 'Once upon a time,'. "
        f"Story: Once upon a time,"
    )
    
    with st.spinner("Generating a short story (40-60 words)..."):
        story_results = story_pipe(
            prompt,
            max_new_tokens=70,          # 先前 100 → 70
            min_new_tokens=40,          # 先前 50 → 40
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True         # 生成完整句子后尽早停止
        )
    
    full_text = story_results[0]['generated_text']
    
    # 提取故事正文
    if "Once upon a time," in full_text:
        story = "Once upon a time," + full_text.split("Once upon a time,")[-1]
    else:
        story = full_text.strip()

    # 智能截断（保留完整句子）
    words = story.split()
    if len(words) > 70:                 # 最多保留 70 个单词（安全边界）
        story = " ".join(words[:70])
        last_punc = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
        if last_punc != -1:
            story = story[:last_punc + 1]
    
    return story

def text2audio(story_text):
    audio_pipe = load_audio_model()
    
    clean_text = story_text.replace("\n", " ").strip()
    import re
    clean_text = re.sub(r'[^a-zA-Z0-9\s,.!?\']', '', clean_text)
    
    # 降低输入长度上限，加快 TTS 速度（同时避免模型崩溃）
    if len(clean_text) > 350:           # 先前 450 → 350
        half_point = clean_text.rfind('.', 0, 350)
        if half_point != -1:
            safe_text = clean_text[:half_point + 1]
        else:
            safe_text = clean_text[:350]
    else:
        safe_text = clean_text
    
    audio_data = audio_pipe(safe_text)
    return audio_data

# --- 3. Main Part ---
st.set_page_config(page_title="Faster Image to Audio Story", page_icon="⚡")
st.header("Turn Your Image to Audio Story (Fast Version)")
uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner('Image → Text...'):
        scenario = img2text(uploaded_file)
        st.write(f"**Caption:** {scenario}")

    with st.spinner('Text → Short Story...'):
        story = text2story(scenario)
        st.subheader("The Story")
        st.info(story)

    with st.spinner('Story → Audio...'):
        audio_result = text2audio(story)
        st.audio(audio_result["audio"], sample_rate=audio_result["sampling_rate"])
        st.balloons()

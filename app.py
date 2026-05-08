import streamlit as st
from transformers import pipeline
from PIL import Image  # 导入图片处理库
import torch

# --- 1. 缓存模型加载 ---
@st.cache_resource
def load_image_to_text_pipe():
    return pipeline(model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_gen_pipe():
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

@st.cache_resource
def load_tts_pipe():
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

# --- 2. 模块化功能函数 ---

def img2text(image_file):
    """将上传的图片转换为文字描述"""
    pipe = load_image_to_text_pipe()
    # 修复：将 Streamlit 上传对象转换为 PIL Image
    img = Image.open(image_file)
    results = pipe(img)
    return results[0]["generated_text"]

def text2story(scenario):
    """生成 50-100 字的故事"""
    pipe = load_story_gen_pipe()
    prompt = f"Write a fun, simple story for a 5-year-old kid based on this: {scenario}. The story must be between 50 and 100 words."
    story_results = pipe(prompt, max_new_tokens=150, min_new_tokens=60, do_sample=True, temperature=0.7)
    story = story_results[0]['generated_text']
    # 清理 Prompt 内容
    if prompt in story:
        story = story.replace(prompt, "").strip()
    return story

def text2audio(story_text):
    """文字转语音"""
    pipe = load_tts_pipe()
    return pipe(story_text)

# --- 3. 主程序界面 ---

def main():
    st.set_page_config(page_title="Kids Storyteller", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    st.markdown("### 👶 A magical storytelling app for kids")

    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Your Uploaded Image", use_container_width=True)

        with st.status("✨ Magic is happening...", expanded=True) as status:
            # 步骤 1
            st.write("🔍 Reading the image...")
            scenario = img2text(uploaded_file)
            st.write(f"**Scenario:** {scenario}")
            
            # 步骤 2
            st.write("✍️ Crafting a story...")
            story = text2story(scenario)
            
            # 步骤 3
            st.write("🎙️ Converting to audio...")
            audio_data = text2audio(story)
            status.update(label="✅ All done!", state="complete", expanded=False)

        st.subheader("📖 The Story")
        st.info(story)
        
        st.subheader("🎧 Listen to the Story")
        st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
        st.balloons()

if __name__ == "__main__":
    main()

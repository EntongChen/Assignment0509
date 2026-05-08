import streamlit as st
from transformers import pipeline
import torch

# --- 1. 缓存模型加载 (解决性能问题和云端崩溃) ---
@st.cache_resource
def load_image_to_text_pipe():
    # 自动推断任务，避免 KeyError
    return pipeline(model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_story_gen_pipe():
    # 使用专门的故事生成模型
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

@st.cache_resource
def load_tts_pipe():
    # 使用标准的 text-to-speech 任务名
    return pipeline("text-to-speech", model="facebook/mms-tts-eng")

# --- 2. 模块化功能函数 (符合评分标准) ---

def img2text(image_file):
    """将上传的图片转换为文字描述"""
    pipe = load_image_to_text_pipe()
    # 直接处理上传的文件对象
    results = pipe(image_file)
    return results[0]["generated_text"]

def text2story(scenario):
    """根据描述为 3-10 岁儿童生成 50-100 字的故事"""
    pipe = load_story_gen_pipe()
    
    # 优化 Prompt：明确受众和风格
    prompt = f"Write a fun, simple, and engaging story for a 5-year-old kid based on this scenario: {scenario}. The story must be between 50 and 100 words."
    
    # 修改参数：确保长度符合要求 (50-100 words)
    # max_new_tokens 设为 150 左右以确保故事完整，min_new_tokens 确保不低于 50
    story_results = pipe(prompt, max_new_tokens=150, min_new_tokens=60, do_sample=True, temperature=0.7)
    
    story = story_results[0]['generated_text']
    
    # 如果模型把 prompt 也返回了，可以尝试清理（取决于具体模型表现）
    if prompt in story:
        story = story.replace(prompt, "").strip()
        
    return story

def text2audio(story_text):
    """将故事文字转换为语音数据"""
    pipe = load_tts_pipe()
    return pipe(story_text)

# --- 3. 主程序界面 (Streamlit UI) ---

def main():
    st.set_page_config(page_title="Kids Storyteller", page_icon="🦜")
    st.header("Turn Your Image to Audio Story")
    st.markdown("### 👶 A magical storytelling app for kids (3-10 years old)")

    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 显示图片
        st.image(uploaded_file, caption="Your Uploaded Image", use_container_width=True)

        # 阶段 1: 图像转文字
        with st.status("🔍 Reading the image...", expanded=True) as status:
            scenario = img2text(uploaded_file)
            st.write(f"**Scenario:** {scenario}")
            
            # 阶段 2: 生成故事
            st.write("✍️ Crafting a story for you...")
            story = text2story(scenario)
            
            # 阶段 3: 生成语音
            st.write("🎙️ Converting story to audio...")
            audio_data = text2audio(story)
            status.update(label="✅ All done!", state="complete", expanded=False)

        # 展示结果
        st.subheader("📖 The Story")
        st.info(story)
        
        # 检查字数（方便你确认是否达标）
        word_count = len(story.split())
        st.caption(f"Word count: {word_count} words")

        st.subheader("🎧 Listen to the Story")
        st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
        
        st.balloons() # 成功后放个气球，增加童趣

if __name__ == "__main__":
    main()

import time
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import PyPDF2
import docx

st.set_page_config(page_title="Умный AI Анализатор", page_icon="🧠", layout="wide")
st.title("Умный анализатор ⚡")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API ключ не найден.")
    st.stop()

# Клиент создаётся один раз и переиспользуется
@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)

client = get_client()

# --- ВЫБОР МОДЕЛИ (в боковой панели) ---
st.sidebar.header("⚙️ Настройки")
model_name = st.sidebar.selectbox(
    "Модель",
    ["gemini-3.5-flash-lite", "gemini-3.5-flash", "Своя..."],
    help="flash-lite — самая быстрая. flash — умнее, но чуть медленнее."
)
if model_name == "Своя...":
    model_name = st.sidebar.text_input("Точное название модели", "gemini-3.5-flash")

fast_mode = st.sidebar.toggle("⚡ Быстрый режим (без долгих размышлений)", value=True)

def build_config():
    """Отключаем или уменьшаем 'размышления' модели."""
    if not fast_mode:
        return None
    if "gemini-3" in model_name:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
    if "2.5" in model_name:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    return None

# --- ПАМЯТЬ ---
for key, val in {"extracted_text": "", "image_data": None, "processed_file_name": ""}.items():
    if key not in st.session_state:
        st.session_state[key] = val

uploaded_file = st.file_uploader(
    "Загрузите файл (TXT, PDF, DOCX, JPG, PNG)",
    type=["txt", "pdf", "docx", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if st.session_state.processed_file_name != uploaded_file.name:
        with st.spinner("Извлекаем данные..."):
            st.session_state.extracted_text = ""
            st.session_state.image_data = None
            try:
                if file_type == "txt":
                    st.session_state.extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
                elif file_type == "pdf":
                    reader = PyPDF2.PdfReader(uploaded_file)
                    st.session_state.extracted_text = "\n".join(
                        (p.extract_text() or "") for p in reader.pages
                    )
                elif file_type == "docx":
                    d = docx.Document(uploaded_file)
                    st.session_state.extracted_text = "\n".join(p.text for p in d.paragraphs)
                else:
                    img = Image.open(uploaded_file)
                    img.thumbnail((1024, 1024))
                    st.session_state.image_data = img
                st.session_state.processed_file_name = uploaded_file.name
            except Exception as e:
                st.error(f"Ошибка при чтении: {e}")
                st.stop()

    if st.session_state.image_data:
        st.image(st.session_state.image_data, caption="Превью", width=300)
    else:
        with st.expander("Посмотреть извлечённый текст"):
            st.text(st.session_state.extracted_text[:1000] + "...")

    default_prompt = ("Опиши, что на этой картинке."
                      if st.session_state.image_data
                      else "Сделай краткую выжимку (саммари) этого текста.")
    user_prompt = st.text_input("Ваш запрос нейросети:", default_prompt)

    if st.button("🚀 Проанализировать"):
        st.markdown("### 🤖 Ответ нейросети:")

        if st.session_state.image_data:
            contents = [user_prompt, st.session_state.image_data]
        else:
            contents = f"{user_prompt}\n\nТекст файла:\n{st.session_state.extracted_text[:30000]}"

        start = time.time()

        def stream_text(config):
            first = True
            for chunk in client.models.generate_content_stream(
                model=model_name, contents=contents, config=config
            ):
                if first:
                    st.toast(f"Первое слово через {time.time() - start:.1f} сек.", icon="⚡")
                    first = False
                if chunk.text:
                    yield chunk.text

        try:
            st.write_stream(stream_text(build_config()))
        except Exception as e:
            # Если модель не поддерживает такую настройку, пробуем без неё
            if "thinking" in str(e).lower():
                st.write_stream(stream_text(None))
            else:
                st.error(f"Ошибка API: {e}")

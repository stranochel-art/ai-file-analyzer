import streamlit as st
import google.generativeai as genai
from PIL import Image
import PyPDF2
import docx
import time  # Добавили библиотеку для таймера

st.set_page_config(page_title="Умный AI Анализатор", page_icon="🧠", layout="wide")
st.title("Умный анализатор ⚡")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API ключ не найден.")
    st.stop()

# --- САМОЕ ВАЖНОЕ ИЗМЕНЕНИЕ: УКАЗЫВАЕМ transport="rest" ---
# Это убирает двухминутное зависание на бесплатных серверах!
genai.configure(api_key=api_key, transport="rest")
model = genai.GenerativeModel('gemini-1.5-flash')

if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'image_data' not in st.session_state:
    st.session_state.image_data = None
if 'processed_file_name' not in st.session_state:
    st.session_state.processed_file_name = ""

uploaded_file = st.file_uploader(
    "Загрузите файл (TXT, PDF, DOCX, JPG, PNG)", 
    type=['txt', 'pdf', 'docx', 'jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if st.session_state.processed_file_name != uploaded_file.name:
        with st.spinner("Извлекаем данные..."):
            st.session_state.extracted_text = ""
            st.session_state.image_data = None
            
            try:
                if file_type == 'txt':
                    st.session_state.extracted_text = uploaded_file.read().decode("utf-8")
                elif file_type == 'pdf':
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        st.session_state.extracted_text += page.extract_text() + "\n"
                elif file_type == 'docx':
                    doc = docx.Document(uploaded_file)
                    for para in doc.paragraphs:
                        st.session_state.extracted_text += para.text + "\n"
                elif file_type in ['jpg', 'jpeg', 'png']:
                    image = Image.open(uploaded_file)
                    image.thumbnail((1024, 1024))
                    st.session_state.image_data = image
                    
                st.session_state.processed_file_name = uploaded_file.name
            except Exception as e:
                st.error(f"Ошибка при чтении: {e}")
                st.stop()

    if st.session_state.image_data:
        st.image(st.session_state.image_data, caption="Превью (оптимизировано)", width=300)
    else:
        with st.expander("Посмотреть извлеченный текст"):
            st.text(st.session_state.extracted_text[:1000] + "...")

    default_prompt = "Опиши, что на этой картинке." if st.session_state.image_data else "Сделай краткую выжимку (саммари) этого текста."
    user_prompt = st.text_input("Ваш запрос нейросети:", default_prompt)
    
    if st.button("🚀 Проанализировать"):
        st.markdown("### 🤖 Ответ нейросети:")
        
        # Засекаем время
        start_time = time.time()
        
        try:
            if st.session_state.image_data:
                response = model.generate_content([user_prompt, st.session_state.image_data], stream=True)
            else:
                safe_text = st.session_state.extracted_text[:30000]
                response = model.generate_content(f"{user_prompt}\n\nТекст файла:\n{safe_text}", stream=True)
            
            # Функция вывода + расчет времени
            def stream_text():
                first_chunk = True
                for chunk in response:
                    if first_chunk:
                        # Считаем, сколько времени прошло до ПЕРВОГО слова
                        ping_time = time.time() - start_time
                        st.toast(f"Отклик Google: {ping_time:.1f} сек.", icon="⚡")
                        first_chunk = False
                    yield chunk.text
                    
            st.write_stream(stream_text)
            
        except Exception as e:
            st.error(f"Ошибка API: {e}")

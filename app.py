import streamlit as st
import google.generativeai as genai
from PIL import Image
import PyPDF2
import docx

st.set_page_config(page_title="Умный AI Анализатор", page_icon="🧠", layout="wide")
st.title("Умный анализатор (Ускоренная версия) ⚡")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API ключ не найден.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- СОЗДАЕМ "ПАМЯТЬ" ДЛЯ САЙТА ---
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
    
    # Читаем файл ТОЛЬКО ЕСЛИ это новый файл (чтобы не тратить время при нажатии кнопки)
    if st.session_state.processed_file_name != uploaded_file.name:
        with st.spinner("Извлекаем данные из файла..."):
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
                    # МАГИЯ УСКОРЕНИЯ: Сжимаем огромные фото перед отправкой
                    image.thumbnail((1024, 1024))
                    st.session_state.image_data = image
                    
                st.session_state.processed_file_name = uploaded_file.name
            except Exception as e:
                st.error(f"Ошибка при чтении: {e}")
                st.stop()

    st.success(f"Файл готов к анализу!")
    
    # Показываем превью
    if st.session_state.image_data:
        st.image(st.session_state.image_data, caption="Превью (оптимизировано)", width=300)
    else:
        with st.expander("Посмотреть извлеченный текст"):
            st.text(st.session_state.extracted_text[:1000] + "...")

    # --- БЛОК АНАЛИЗА ---
    default_prompt = "Опиши, что на этой картинке." if st.session_state.image_data else "Сделай краткую выжимку (саммари) этого текста."
    user_prompt = st.text_input("Ваш запрос нейросети:", default_prompt)
    
    if st.button("🚀 Проанализировать мгновенно"):
        st.markdown("### 🤖 Ответ нейросети:")
        
        try:
            # Берем данные прямо из быстрой памяти
            if st.session_state.image_data:
                response = model.generate_content([user_prompt, st.session_state.image_data], stream=True)
            else:
                safe_text = st.session_state.extracted_text[:30000]
                response = model.generate_content(f"{user_prompt}\n\nТекст файла:\n{safe_text}", stream=True)
            
            # Потоковый вывод
            def stream_text():
                for chunk in response:
                    yield chunk.text
                    
            st.write_stream(stream_text)
            
        except Exception as e:
            st.error(f"Ошибка API: {e}")

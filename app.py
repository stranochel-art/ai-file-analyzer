import streamlit as st
import google.generativeai as genai
from PIL import Image
import PyPDF2
import docx

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="Умный AI Анализатор", page_icon="🧠", layout="wide")
st.title("Умный анализатор: Текст, PDF, Word и Картинки 📄🖼️")

# --- НАСТРОЙКА API КЛЮЧА ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API ключ не найден. Настройте его в Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3.8-flash')

# --- ИНТЕРФЕЙС ЗАГРУЗКИ (ДОБАВИЛИ НОВЫЕ ФОРМАТЫ) ---
uploaded_file = st.file_uploader(
    "Загрузите файл (TXT, PDF, DOCX, JPG, PNG)", 
    type=['txt', 'pdf', 'docx', 'jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Определяем тип файла по расширению
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    st.success(f"Файл {uploaded_file.name} успешно загружен!")
    
    # --- БЛОК ЧТЕНИЯ ФАЙЛОВ ---
    extracted_text = ""
    image_data = None
    
    try:
        # Если это текст
        if file_type == 'txt':
            extracted_text = uploaded_file.read().decode("utf-8")
            with st.expander("Посмотреть текст"):
                st.text(extracted_text[:1000] + "...")
                
        # Если это PDF
        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted_text += page.extract_text() + "\n"
            with st.expander("Посмотреть текст из PDF"):
                st.text(extracted_text[:1000] + "...")
                
        # Если это Word
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
            with st.expander("Посмотреть текст из Word"):
                st.text(extracted_text[:1000] + "...")
                
        # Если это Картинка
        elif file_type in ['jpg', 'jpeg', 'png']:
            image_data = Image.open(uploaded_file)
            st.image(image_data, caption="Загруженное изображение", use_column_width=True)
            
    except Exception as e:
        st.error(f"Ошибка при чтении файла: {e}")
        st.stop()

    # --- БЛОК АНАЛИЗА ---
    st.markdown("### Что сделать с этим файлом?")
    
    # Меняем подсказку в зависимости от файла
    default_prompt = "Опиши, что на этой картинке, и выдели главные детали." if image_data else "Сделай краткую выжимку (саммари) этого текста."
    user_prompt = st.text_input("Ваш запрос нейросети:", default_prompt)
    
    if st.button("🚀 Проанализировать"):
        with st.spinner("Нейросеть работает..."):
            try:
                # Отправляем запрос. Если есть картинка - шлем картинку, если текст - шлем текст.
                if image_data:
                    response = model.generate_content([user_prompt, image_data])
                else:
                    response = model.generate_content(f"{user_prompt}\n\nТекст файла:\n{extracted_text}")
                
                # Выводим результат
                st.markdown("### 🤖 Ответ нейросети:")
                st.info(response.text)
                
            except Exception as e:
                st.error(f"Ошибка API: {e}")

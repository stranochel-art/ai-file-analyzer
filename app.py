import streamlit as st
import requests
import json
import base64
import time
from io import BytesIO
from PIL import Image
import PyPDF2
import docx

st.set_page_config(page_title="AI Анализатор (Прямое подключение)", page_icon="⚡", layout="wide")
st.title("Умный анализатор (Прямой HTTP) 🚀")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API ключ не найден.")
    st.stop()

# --- ПАМЯТЬ СЕРВЕРА ---
for key in ["extracted_text", "image_b64", "processed_file_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

uploaded_file = st.file_uploader(
    "Загрузите файл (TXT, PDF, DOCX, JPG, PNG)",
    type=["txt", "pdf", "docx", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if st.session_state.processed_file_name != uploaded_file.name:
        with st.spinner("Быстрая обработка файла..."):
            st.session_state.extracted_text = ""
            st.session_state.image_b64 = None
            
            try:
                if file_type == "txt":
                    st.session_state.extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")
                elif file_type == "pdf":
                    reader = PyPDF2.PdfReader(uploaded_file)
                    st.session_state.extracted_text = "\n".join((p.extract_text() or "") for p in reader.pages)
                elif file_type == "docx":
                    d = docx.Document(uploaded_file)
                    st.session_state.extracted_text = "\n".join(p.text for p in d.paragraphs)
                else:
                    # Картинка: сжимаем и переводим в Base64 для прямого запроса
                    img = Image.open(uploaded_file)
                    img.thumbnail((800, 800))
                    
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    st.session_state.image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                st.session_state.processed_file_name = uploaded_file.name
            except Exception as e:
                st.error(f"Ошибка при чтении: {e}")
                st.stop()

    if st.session_state.image_b64:
        st.success("Картинка загружена и сжата!")
    else:
        with st.expander("Посмотреть извлечённый текст"):
            st.text(st.session_state.extracted_text[:1000] + "...")

    default_prompt = "Опиши эту картинку." if st.session_state.image_b64 else "Сделай выжимку этого текста."
    user_prompt = st.text_input("Запрос:", default_prompt)

    if st.button("🚀 Проанализировать (Без библиотек)"):
        st.markdown("### 🤖 Ответ:")
        
        # 1. Формируем "сырой" пакет данных для Google
        if st.session_state.image_b64:
            payload = {
                "contents": [{
                    "parts": [
                        {"text": user_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": st.session_state.image_b64}}
                    ]
                }]
            }
        else:
            safe_text = st.session_state.extracted_text[:30000]
            payload = {
                "contents": [{
                    "parts": [{"text": f"{user_prompt}\n\nТекст:\n{safe_text}"}]
                }]
            }

        # 2. Прямая ссылка на API (используем стабильную и самую быструю модель 1.5-flash)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?alt=sse&key={api_key}"
        
        start_time = time.time()

        # 3. Функция, которая слушает прямой поток от Google
        def stream_direct():
            first_chunk = True
            
            # Делаем прямой запрос
            response = requests.post(url, json=payload, stream=True)
            
            if response.status_code != 200:
                yield f"❌ Ошибка Google API: {response.text}"
                return

            # Читаем ответ построчно
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # Google присылает данные с приставкой "data: "
                    if decoded_line.startswith("data: "):
                        json_data = decoded_line[6:] # Отрезаем "data: "
                        try:
                            data = json.loads(json_data)
                            # Достаем кусок текста из сложного JSON
                            text_chunk = data["candidates"][0]["content"]["parts"][0]["text"]
                            
                            if first_chunk:
                                ping = time.time() - start_time
                                st.toast(f"Отклик без библиотек: {ping:.1f} сек.", icon="🏎️")
                                first_chunk = False
                                
                            yield text_chunk
                        except (KeyError, json.JSONDecodeError):
                            pass

        # Запускаем вывод на экран
        st.write_stream(stream_direct)

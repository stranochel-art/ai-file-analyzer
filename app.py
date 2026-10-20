import socket

# --- ХАКЕРСКИЙ ПАТЧ СЕТИ (УБИРАЕТ 20-СЕКУНДНУЮ ЗАДЕРЖКУ) ---
# Заставляем Python использовать только IPv4, чтобы не ждать таймаутов Streamlit
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo
# -----------------------------------------------------------

import streamlit as st
import requests
import json
import base64
import time
from io import BytesIO
from PIL import Image
import PyPDF2
import docx

st.set_page_config(page_title="AI Анализатор (Турбо + Фото)", page_icon="⚡", layout="wide")
st.title("Умный анализатор (Исправленный) 🚀")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API ключ не найден.")
    st.stop()

# Память
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
        with st.spinner("Чтение файла..."):
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
                    # Картинка
                    img = Image.open(uploaded_file)
                    # Если картинка в формате RGBA (PNG с прозрачностью), конвертируем в RGB
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.thumbnail((800, 800))
                    
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    st.session_state.image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                st.session_state.processed_file_name = uploaded_file.name
            except Exception as e:
                st.error(f"Ошибка при чтении: {e}")
                st.stop()

    if st.session_state.image_b64:
        st.success("Картинка загружена!")
    else:
        with st.expander("Посмотреть извлечённый текст"):
            st.text(st.session_state.extracted_text[:1000] + "...")

    default_prompt = "Опиши эту картинку детально." if st.session_state.image_b64 else "Сделай выжимку этого текста."
    user_prompt = st.text_input("Запрос:", default_prompt)

    if st.button("🚀 Проанализировать мгновенно"):
        st.markdown("### 🤖 Ответ:")
        
        # 1. Формируем правильный JSON (исправлен регистр букв: inlineData, mimeType)
        if st.session_state.image_b64:
            payload = {
                "contents": [{
                    "parts": [
                        {"text": user_prompt},
                        {"inlineData": {"mimeType": "image/jpeg", "data": st.session_state.image_b64}}
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

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?alt=sse&key={api_key}"
        
        start_time = time.time()

        def stream_direct():
            first_chunk = True
            
            try:
                # Отправляем запрос
                response = requests.post(url, json=payload, stream=True, timeout=10) # Добавил жесткий таймаут
                
                if response.status_code != 200:
                    yield f"❌ Ошибка: {response.text}"
                    return

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            json_data = decoded_line[6:]
                            try:
                                data = json.loads(json_data)
                                text_chunk = data["candidates"][0]["content"]["parts"][0]["text"]
                                
                                if first_chunk:
                                    ping = time.time() - start_time
                                    st.toast(f"Отклик сети: {ping:.1f} сек.", icon="🏎️")
                                    first_chunk = False
                                    
                                yield text_chunk
                            except (KeyError, json.JSONDecodeError, IndexError):
                                pass
            except requests.exceptions.RequestException as e:
                yield f"❌ Ошибка сети: {e}"

        st.write_stream(stream_direct)

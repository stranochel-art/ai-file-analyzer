import socket
import time

old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

import streamlit as st
import requests
import json
import base64
from io import BytesIO
from PIL import Imageа
import PyPDF2
import docx

st.set_page_config(page_title="Анализатор", layout="centered")
st.title("Анализатор документов")

if "http_session" not in st.session_state:
    st.session_state.http_session = requests.Session()

api_keys = st.secrets.get("GEMINI_API_KEYS", [])

if not api_keys:
    single_key = st.secrets.get("GEMINI_API_KEY")
    if single_key:
        api_keys = [single_key]
    else:
        st.error("API ключи не найдены в настройках.")
        st.stop()

if "current_key_index" not in st.session_state:
    st.session_state.current_key_index = 0

for key in ["extracted_text", "image_b64", "processed_file_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

uploaded_file = st.file_uploader(
    "Файл (TXT, PDF, DOCX, JPG, PNG)",
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
                    img = Image.open(uploaded_file)
                    if img.mode == 'RGBA':
                        img = img.convert('RGB')
                    img.thumbnail((512, 512))
                    
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    st.session_state.image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                st.session_state.processed_file_name = uploaded_file.name
            except Exception as e:
                st.error(f"Ошибка при чтении: {e}")
                st.stop()

    if st.session_state.image_b64:
        st.success("Изображение готово.")
    else:
        with st.expander("Извлечённый текст"):
            st.text(st.session_state.extracted_text[:1000] + "...")

    default_prompt = "Опиши изображение." if st.session_state.image_b64 else "Сделай выжимку текста."
    
    with st.form("analyze_form"):
        user_prompt = st.text_input("Запрос:", default_prompt)
        submitted = st.form_submit_button("Проанализировать")

    if submitted:
        st.markdown("### Ответ:")
        
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
            safe_text = st.session_state.extracted_text[:15000]
            payload = {
                "contents": [{
                    "parts": [{"text": f"{user_prompt}\n\nТекст:\n{safe_text}"}]
                }]
            }

        current_key = api_keys[st.session_state.current_key_index]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:streamGenerateContent?alt=sse&key={current_key}"
        
        # Заголовки для принудительного стриминга (отключают буферизацию)
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache"
        }
        
        start_time = time.time()
        
        def stream_direct():
            first_chunk = True
            try:
                response = st.session_state.http_session.post(url, json=payload, headers=headers, stream=True)
                
                if response.status_code == 429:
                    st.session_state.current_key_index += 1
                    if st.session_state.current_key_index >= len(api_keys):
                        st.session_state.current_key_index = 0
                        yield "Ошибка: Лимиты на всех ключах исчерпаны."
                    else:
                        yield f"Лимит исчерпан. Переключение на ключ №{st.session_state.current_key_index + 1}. Нажмите кнопку еще раз."
                    return
                elif response.status_code == 503:
                    yield "Серверы Google сейчас перегружены. Пожалуйста, нажмите кнопку 'Проанализировать' еще раз."
                    return
                elif response.status_code != 200:
                    yield f"Ошибка API ({response.status_code}): {response.text}"
                    return

                # Ручная расшифровка в UTF-8 убирает ломанный текст
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            json_data = decoded_line[6:]
                            try:
                                data = json.loads(json_data)
                                text_chunk = data["candidates"][0]["content"]["parts"][0]["text"]
                                
                                if first_chunk:
                                    elapsed = time.time() - start_time
                                    st.toast(f"Отклик: {elapsed:.1f} сек.")
                                    first_chunk = False
                                    
                                yield text_chunk
                            except (KeyError, json.JSONDecodeError, IndexError):
                                pass
            except requests.exceptions.RequestException as e:
                yield f"Сетевая ошибка: {e}"

        st.write_stream(stream_direct)

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
from PIL import Image
import PyPDF2
import docx

# ==================== НАСТРОЙКИ ====================
MODEL             = "gemini-3.8-flash"   # было: gemini-3.6-flash
THINKING_DEFAULT  = "low"                # ГЛАВНОЕ ИЗМЕНЕНИЕ (было не задано = medium)
MAX_CHARS         = 60000                # было: 15000 (у Gemini 3.x контекст 1M токенов)
MAX_OUTPUT_TOKENS = 2048                 # не ставьте <1024: лимит режет и thinking тоже
# requirements.txt не изменился: streamlit, requests, Pillow, PyPDF2, python-docx
# ===================================================

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
        col_a, col_b = st.columns([3, 1])
        with col_a:
            thinking_level = st.selectbox(
                "Глубина размышлений",
                ["low", "medium", "high"],
                index=["low", "medium", "high"].index(THINKING_DEFAULT),
                help="low = быстро и умно (для выжимок, фактов, перевода). "
                     "medium/high — для сложных рассуждений, но заметно дольше."
            )
        with col_b:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Проанализировать")

    if submitted:
        st.markdown("### Ответ:")

        generation_config = {
            "thinkingConfig": {"thinkingLevel": thinking_level},
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        }

        if st.session_state.image_b64:
            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": st.session_state.image_b64}},
                        {"text": user_prompt}
                    ]
                }],
                "generationConfig": generation_config
            }
        else:
            safe_text = st.session_state.extracted_text[:MAX_CHARS]
            payload = {
                "contents": [{
                    "parts": [
                        # Документ ПЕРВЫМ, вопрос ПОСЛЕДНИМ: префикс запроса совпадает
                        # между разными вопросами к одному файлу -> работает implicit caching.
                        {"text": f"Текст:\n{safe_text}"},
                        {"text": f"\n\nЗадача: {user_prompt}"}
                    ]
                }],
                "generationConfig": generation_config
            }

        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{MODEL}:streamGenerateContent?alt=sse")

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "x-goog-api-key": ""   # подставляется для каждого ключа (ключ больше не в URL)
        }

        start_time = time.time()

        def stream_direct():
            first_chunk = True
            total_keys = len(api_keys)
            start_index = st.session_state.current_key_index

            # Перебор ключей ВНУТРИ одного запуска: при 429 сразу берём следующий,
            # вместо «нажмите кнопку ещё раз».
            for i in range(total_keys):
                key_index = (start_index + i) % total_keys
                headers["x-goog-api-key"] = api_keys[key_index]

                try:
                    response = st.session_state.http_session.post(
                        url, json=payload, headers=headers, stream=True
                    )
                except requests.exceptions.RequestException as e:
                    yield f"Сетевая ошибка: {e}"
                    return

                if response.status_code in (429, 403):
                    st.session_state.current_key_index = (key_index + 1) % total_keys
                    continue

                if response.status_code == 503:
                    yield "Серверы Google сейчас перегружены. Пожалуйста, нажмите кнопку 'Проанализировать' еще раз."
                    return

                if response.status_code != 200:
                    yield f"Ошибка API ({response.status_code}): {response.text[:300]}"
                    return

                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded_line = line.decode('utf-8')
                    if not decoded_line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(decoded_line[6:])
                    except json.JSONDecodeError:
                        continue

                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    for part in parts:
                        if part.get("thought"):        # внутренние рассуждения — не в ответ
                            continue
                        text_chunk = part.get("text")
                        if not text_chunk:
                            continue
                        if first_chunk:
                            elapsed = time.time() - start_time
                            st.toast(f"Отклик: {elapsed:.1f} сек.")
                            first_chunk = False
                        yield text_chunk

                st.session_state.current_key_index = key_index
                return

            yield "Ошибка: Лимиты на всех ключах исчерпаны. Повторите примерно через минуту."

        st.write_stream(stream_direct)

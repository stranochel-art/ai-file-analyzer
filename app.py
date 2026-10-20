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
# Модель по умолчанию — 3.6-flash: на вашем ключе она быстрая и поддерживает
# уровень "minimal", который реально убирает задержку на рассуждения.
# 3.8-flash оставлен в списке — включайте, когда нужен именно глубокий разбор.
MODELS = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]

# Какие уровни размышлений поддерживает каждая модель
THINKING_LEVELS = {
    "gemini-3.6-flash":    ["minimal", "low", "medium", "high"],
    "gemini-3.8-flash":    ["low", "medium", "high"],   # minimal у 3.8 недоступен
    "gemini-3.5-flash-lite": ["minimal", "low", "medium", "high"],
}

# Сколько токенов ответа разрешить (включает и thinking-токены!)
MAX_OUTPUT_BY_THINKING = {"minimal": 1024, "low": 1024, "medium": 2048, "high": 4096}

MAX_CHARS = 15000   # вернули как было: 60000 упиралось в лимит токенов/минуту
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


def diagnose(ttfb, total, prompt_tokens, thoughts, answer_tokens, cached):
    """Короткий вердикт по цифрам, чтобы не гадать, где тормозит."""
    if ttfb and ttfb > 25 and prompt_tokens > 8000:
        return ("Упор в лимит токенов/минуту бесплатного тарифа: большой вход "
                f"({prompt_tokens} ток.) ставится в очередь. Уменьшите MAX_CHARS "
                "или добавьте второй ключ в GEMINI_API_KEYS.")
    if thoughts > 800:
        return (f"Модель сожгла {thoughts} ток. на внутренние рассуждения до первого "
                "символа. Поставьте уровень размышлений ниже (minimal/low).")
    if ttfb and ttfb < 6 and total > 40:
        return (f"Старт быстрый ({ttfb:.1f} с), но ответ длинный ({answer_tokens} ток.). "
                "Снизьте лимит ответа или попросите «ответь в 5 предложениях».")
    if ttfb and ttfb > 25:
        return ("Запрос маленький, а ожидание большое — очередь/троттлинг на стороне API "
                "именно для этой модели. Попробуйте другую модель или другой ключ.")
    if prompt_tokens > 4000 and cached == 0:
        return ("Работает в норме, но префикс не попадает в кэш Google: проверьте, что "
                "текст документа идёт первым партсом, а вопрос — последним.")
    return "Норма: задержка соответствует выбранной модели и уровню размышлений."


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

        col_m, col_t, col_b = st.columns([2, 1.4, 1])
        with col_m:
            model = st.selectbox("Модель", MODELS, index=0)
        with col_t:
            levels = THINKING_LEVELS[model]
            thinking_level = st.selectbox(
                "Размышления",
                levels,
                index=0,
                help="minimal/low = быстро. medium/high = дольше, но лучше на сложных задачах."
            )
        with col_b:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Проанализировать")

    if submitted:
        st.markdown("### Ответ:")

        generation_config = {
            "thinkingConfig": {"thinkingLevel": thinking_level},
            "maxOutputTokens": MAX_OUTPUT_BY_THINKING.get(thinking_level, 1024),
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
                        # между разными вопросами к одному файлу -> работает кэш Google.
                        {"text": f"Текст:\n{safe_text}"},
                        {"text": f"\n\nЗадача: {user_prompt}"}
                    ]
                }],
                "generationConfig": generation_config
            }

        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:streamGenerateContent?alt=sse")

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "x-goog-api-key": ""   # ключ в заголовке, а не в URL (не светится в логах)
        }

        start_time = time.time()
        st.session_state.ttfb = None
        st.session_state.last_usage = {}

        def stream_direct():
            first_chunk = True
            total_keys = len(api_keys)
            start_index = st.session_state.current_key_index

            # Перебор ключей ВНУТРИ одного запуска: при 429 сразу берём следующий,
            # а не просим пользователя нажать кнопку ещё раз.
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

                    # служебная статистика приходит в последнем чанке
                    if data.get("usageMetadata"):
                        st.session_state.last_usage = data["usageMetadata"]

                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    for part in parts:
                        if part.get("thought"):        # внутренние рассуждения — не в ответ
                            continue
                        text_chunk = part.get("text")
                        if not text_chunk:
                            continue
                        if first_chunk:
                            st.session_state.ttfb = time.time() - start_time
                            st.toast(f"Отклик: {st.session_state.ttfb:.1f} сек.")
                            first_chunk = False
                        yield text_chunk

                st.session_state.current_key_index = key_index
                return

            yield "Ошибка: Лимиты на всех ключах исчерпаны. Повторите примерно через минуту."

        st.write_stream(stream_direct)

        # ==================== ДИАГНОСТИКА СКОРОСТИ ====================
        u = st.session_state.last_usage or {}
        ttfb = st.session_state.ttfb or 0
        total = time.time() - start_time
        prompt_tokens = u.get("promptTokenCount", 0)
        thoughts = u.get("thoughtsTokenCount", 0)
        answer_tokens = u.get("candidatesTokenCount", 0)
        cached = u.get("cachedContentTokenCount", 0)

        with st.expander("Замеры и вердикт"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Первый токен", f"{ttfb:.1f} с")
            c2.metric("Всего", f"{total:.1f} с")
            c3.metric("Thinking", f"{thoughts} ток.")
            c4.metric("Ответ", f"{answer_tokens} ток.")
            st.caption(
                f"модель: {model} · размышления: {thinking_level} · "
                f"вход: {prompt_tokens} ток. · из кэша: {cached} ток. · "
                f"ключей в пуле: {len(api_keys)}"
            )
            if u:
                st.info(diagnose(ttfb, total, prompt_tokens, thoughts, answer_tokens, cached))
            else:
                st.warning("Статистика от API не пришла — вероятно, ответ оборвался с ошибкой.")

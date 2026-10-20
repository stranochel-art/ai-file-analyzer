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
MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]

# Запасная модель: на неё уходим, если основная стоит в очереди
FALLBACK_MODEL = "gemini-3.5-flash-lite"

THINKING_LEVELS = {
    "gemini-3.6-flash":      ["minimal", "low", "medium", "high"],
    "gemini-3.8-flash":      ["low", "medium", "high"],   # minimal у 3.8 недоступен
    "gemini-3.5-flash-lite": ["minimal", "low", "medium", "high"],
}

MAX_OUTPUT_BY_THINKING = {"minimal": 1024, "low": 1024, "medium": 2048, "high": 4096}

MAX_CHARS      = 15000
STALL_SECONDS  = 8    # сколько ждём ПЕРВЫЙ байт / следующий байт потока
MAX_ATTEMPTS   = 4    # сколько связок "ключ+модель" пробуем подряд
# requirements.txt: streamlit, requests, Pillow, PyPDF2, python-docx
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


def diagnose(ttfb, total, prompt_tokens, thoughts, answer_tokens, cached, attempts):
    if attempts > 1:
        return (f"Основной канал стоял в очереди, ответ пришёл с {attempts}-й попытки. "
                "Это троттлинг бесплатного тарифа, а не ваш код.")
    if ttfb and ttfb > 20 and prompt_tokens < 2000 and thoughts == 0:
        return ("Очередь бесплатного тарифа: вход крошечный, рассуждений ноль, "
                "а ждали до первого байта долго. Помогает запасная модель/другой ключ, "
                "а радикально — отдельный быстрый провайдер (Groq LPU, первый токен <1 с).")
    if ttfb and ttfb > 25 and prompt_tokens > 8000:
        return (f"Упор в лимит токенов/минуту: вход {prompt_tokens} ток. "
                "Уменьшите MAX_CHARS или добавьте ключей.")
    if thoughts > 800:
        return (f"{thoughts} ток. ушло на внутренние рассуждения. Поставьте уровень ниже.")
    if ttfb and ttfb < 6 and total > 40:
        return ("Старт быстрый, ответ длинный. Снизьте лимит ответа или попросите короче.")
    if prompt_tokens > 4000 and cached == 0:
        return ("Норма, но префикс не кэшируется: текст документа должен идти первым партсом.")
    return "Норма."


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
                "Размышления", levels, index=0,
                help="minimal/low = быстро. medium/high = дольше, но лучше на сложных задачах."
            )
        with col_b:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Проанализировать")

    if submitted:
        st.markdown("### Ответ:")

        if st.session_state.image_b64:
            base_parts = [
                {"inlineData": {"mimeType": "image/jpeg", "data": st.session_state.image_b64}},
                {"text": user_prompt}
            ]
        else:
            safe_text = st.session_state.extracted_text[:MAX_CHARS]
            base_parts = [
                # Документ ПЕРВЫМ, вопрос ПОСЛЕДНИМ — так работает кэш префикса Google.
                {"text": f"Текст:\n{safe_text}"},
                {"text": f"\n\nЗадача: {user_prompt}"}
            ]

        payload = {"contents": [{"parts": base_parts}]}

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "x-goog-api-key": ""
        }

        # Порядок попыток: выбранная модель на всех ключах, затем запасная модель.
        fallback = FALLBACK_MODEL if FALLBACK_MODEL != model else "gemini-3.6-flash"
        plan = []
        for m in [model, fallback]:
            for i in range(len(api_keys)):
                plan.append((m, (st.session_state.current_key_index + i) % len(api_keys)))
        plan = plan[:MAX_ATTEMPTS]

        start_time = time.time()
        st.session_state.ttfb = None
        st.session_state.last_usage = {}
        attempt_log = []

        def stream_direct():
            first_chunk = True
            got_text = False
            used_model = model

            for attempt, (m, key_index) in enumerate(plan, start=1):
                # уровень размышлений мог не поддерживаться запасной моделью
                lvl = thinking_level if thinking_level in THINKING_LEVELS[m] else THINKING_LEVELS[m][0]
                payload["generationConfig"] = {
                    "thinkingConfig": {"thinkingLevel": lvl},
                    "maxOutputTokens": MAX_OUTPUT_BY_THINKING.get(lvl, 1024),
                }
                headers["x-goog-api-key"] = api_keys[key_index]
                url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                       f"{m}:streamGenerateContent?alt=sse")
                tag = f"{m} / ключ №{key_index + 1}"

                try:
                    # timeout=(подключение, ожидание байта): STALL_SECONDS — детектор очереди.
                    response = st.session_state.http_session.post(
                        url, json=payload, headers=headers, stream=True,
                        timeout=(10, STALL_SECONDS)
                    )
                except requests.exceptions.Timeout:
                    attempt_log.append(f"{tag}: тишина {STALL_SECONDS} с — очередь, пробуем дальше")
                    continue
                except requests.exceptions.RequestException as e:
                    attempt_log.append(f"{tag}: сетевая ошибка {e}")
                    yield f"Сетевая ошибка: {e}"
                    return

                if response.status_code in (429, 403):
                    response.close()
                    attempt_log.append(f"{tag}: {response.status_code} — лимит ключа")
                    st.session_state.current_key_index = (key_index + 1) % len(api_keys)
                    continue

                if response.status_code == 503:
                    response.close()
                    attempt_log.append(f"{tag}: 503 перегрузка")
                    continue

                if response.status_code != 200:
                    body = response.text[:300]
                    response.close()
                    yield f"Ошибка API ({response.status_code}): {body}"
                    return

                used_model = m
                attempt_log.append(f"{tag}: принят, жду первый токен")

                try:
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

                        if data.get("usageMetadata"):
                            st.session_state.last_usage = data["usageMetadata"]

                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if part.get("thought"):
                                continue
                            text_chunk = part.get("text")
                            if not text_chunk:
                                continue
                            if first_chunk:
                                st.session_state.ttfb = time.time() - start_time
                                st.toast(f"Отклик: {st.session_state.ttfb:.1f} сек. ({m})")
                                first_chunk = False
                            got_text = True
                            yield text_chunk
                except requests.exceptions.Timeout:
                    response.close()
                    if got_text:
                        attempt_log.append(f"{tag}: поток оборвался на середине")
                        yield "\n\n_…поток оборвался, ответ неполный._"
                        return
                    attempt_log.append(f"{tag}: поток встал — пробуем дальше")
                    continue
                finally:
                    response.close()

                st.session_state.current_key_index = key_index
                st.session_state.used_model = used_model
                return

            st.session_state.used_model = used_model
            yield ("Ошибка: все ключи и запасная модель сейчас в очереди или в лимите. "
                   "Повторите примерно через минуту.")

        st.write_stream(stream_direct)

        # ==================== ДИАГНОСТИКА ====================
        u = st.session_state.last_usage or {}
        ttfb = st.session_state.ttfb or 0
        total = time.time() - start_time
        prompt_tokens = u.get("promptTokenCount", 0)
        thoughts = u.get("thoughtsTokenCount", 0)
        answer_tokens = u.get("candidatesTokenCount", 0)
        cached = u.get("cachedContentTokenCount", 0)
        attempts = len(attempt_log)

        with st.expander("Замеры и вердикт", expanded=attempts > 1):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Первый токен", f"{ttfb:.1f} с")
            c2.metric("Всего", f"{total:.1f} с")
            c3.metric("Thinking", f"{thoughts} ток.")
            c4.metric("Ответ", f"{answer_tokens} ток.")
            st.caption(
                f"модель: {st.session_state.get('used_model', model)} · "
                f"размышления: {thinking_level} · вход: {prompt_tokens} ток. · "
                f"из кэша: {cached} ток. · попыток: {attempts}"
            )
            for line in attempt_log:
                st.caption(f"• {line}")
            if u:
                st.info(diagnose(ttfb, total, prompt_tokens, thoughts,
                                 answer_tokens, cached, attempts))
            else:
                st.warning("Статистика от API не пришла — ответ оборвался с ошибкой.")

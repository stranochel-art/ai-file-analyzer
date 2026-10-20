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

# ========================= НАСТРОЙКИ =========================
# --- GROQ: быстрый текстовый канал. LPU-железо, очереди как у Google тут нет.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
# Бесплатный лимит Groq = 8 000 токенов В МИНУТУ на все запросы.
# 9000 символов ≈ 2500 токенов — влезает вместе с ответом.
GROQ_MAX_CHARS  = 9000
GROQ_MAX_TOKENS = 1200
GROQ_STALL      = 6      # секунд тишины = канал лёг, идём дальше

# --- GEMINI: умный канал и единственный, который видит картинки
GEMINI_LADDER = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]
THINKING_LEVELS = {
    "gemini-3.5-flash-lite": ["minimal", "low", "medium", "high"],
    "gemini-3.6-flash":      ["minimal", "low", "medium", "high"],
    "gemini-3.8-flash":      ["low", "medium", "high"],   # minimal у 3.8 нет
}
MAX_OUTPUT_BY_THINKING = {"minimal": 1024, "low": 1024, "medium": 2048, "high": 4096}
MAX_CHARS    = 15000
GEMINI_STALL = 5
MAX_ATTEMPTS = 6

ENGINES = [
    "Авто (Groq → Gemini)",
    "Только Groq (быстро)",
    "Только Gemini (умно)",
    "Gemini deep (максимум ума, медленно)",
]
# requirements.txt НЕ менялся: streamlit, requests, Pillow, PyPDF2, python-docx
# =============================================================

st.set_page_config(page_title="Анализатор", layout="centered")
st.title("Анализатор документов")

if "http_session" not in st.session_state:
    st.session_state.http_session = requests.Session()

# ---------- ключи ----------
api_keys = st.secrets.get("GEMINI_API_KEYS", [])
if not api_keys:
    single_key = st.secrets.get("GEMINI_API_KEY")
    if single_key:
        api_keys = [single_key]

groq_keys = st.secrets.get("GROQ_API_KEYS", [])
if not groq_keys:
    gk = st.secrets.get("GROQ_API_KEY")
    if gk:
        groq_keys = [gk]

if not api_keys and not groq_keys:
    st.error("Нет ни одного ключа. Добавьте GROQ_API_KEY и/или GEMINI_API_KEY в Secrets.")
    st.stop()

# ---------- состояние ----------
if "current_key_index" not in st.session_state:
    st.session_state.current_key_index = 0
if "current_groq_index" not in st.session_state:
    st.session_state.current_groq_index = 0
if "key_cooldown" not in st.session_state:
    st.session_state.key_cooldown = {}      # {номер ключа Gemini: время до}
if "groq_cooldown" not in st.session_state:
    st.session_state.groq_cooldown = {}     # {модель или номер ключа: время до}
if "groq_param" not in st.session_state:
    st.session_state.groq_param = "max_completion_tokens"
for key in ["extracted_text", "image_b64", "processed_file_name"]:
    if key not in st.session_state:
        st.session_state[key] = None


def diagnose(engine, ttfb, total, prompt_tokens, thoughts, answer_tokens, attempts):
    if attempts > 1:
        return (f"Первый канал не ответил, сработал запасной (попыток: {attempts}). "
                "Это нормальная работа лестницы, а не поломка.")
    if engine == "groq":
        if ttfb and ttfb > 4:
            return "Groq ответил медленнее обычного — вероятно, упёрлись в 8 000 ток/мин. Подождите минуту или уменьшите GROQ_MAX_CHARS."
        return "Норма: Groq — самый быстрый канал, очередь тут почти всегда пустая."
    if ttfb and ttfb > 20 and prompt_tokens < 2000 and thoughts == 0:
        return ("Очередь бесплатного Gemini: вход крошечный, рассуждений ноль, "
                "а ждали долго. Для текста используйте «Авто» или «Только Groq», "
                "Gemini оставьте для картинок и режима deep.")
    if ttfb and ttfb > 20 and prompt_tokens > 8000:
        return f"Упор в лимит токенов/минуту Gemini: вход {prompt_tokens} ток. Уменьшите MAX_CHARS."
    if thoughts > 800:
        return f"{thoughts} ток. ушло на внутренние рассуждения — понизьте уровень размышлений."
    if ttfb and ttfb < 6 and total > 40:
        return "Старт быстрый, ответ длинный. Попросите короче или снизьте лимит ответа."
    return "Норма."


# ---------- загрузка файла ----------
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
        st.success("Изображение готово. Картинки анализирует только Gemini.")
    else:
        with st.expander("Извлечённый текст"):
            st.text(st.session_state.extracted_text[:1000] + "...")

    default_prompt = "Опиши изображение." if st.session_state.image_b64 else "Сделай выжимку текста."

    with st.form("analyze_form"):
        user_prompt = st.text_input("Запрос:", default_prompt)
        engine_choice = st.selectbox("Движок", ENGINES, index=0)

        col_m, col_t, col_b = st.columns([2, 1.4, 1])
        with col_m:
            gemini_model = st.selectbox("Модель Gemini (если понадобится)", GEMINI_LADDER, index=0)
        with col_t:
            levels = THINKING_LEVELS[gemini_model]
            thinking_level = st.selectbox("Размышления", levels, index=0)
        with col_b:
            st.write("")
            st.write("")
            submitted = st.form_submit_button("Проанализировать")

    if submitted:
        st.markdown("### Ответ:")

        is_image = bool(st.session_state.image_b64)
        doc_text = st.session_state.extracted_text or ""
        now = time.time()
        key_cooldown = st.session_state.key_cooldown
        groq_cooldown = st.session_state.groq_cooldown
        attempt_log = []

        # Картинки Groq не понимает — только Gemini.
        if is_image and engine_choice in (ENGINES[0], ENGINES[1]):
            engine_choice = ENGINES[2]
            attempt_log.append("Изображение: Groq пропущен, работает Gemini")

        def groq_attempts(models):
            out = []
            for m in models:
                if groq_cooldown.get(m, 0) > now:
                    continue
                for i in range(len(groq_keys)):
                    ki = (st.session_state.current_groq_index + i) % len(groq_keys)
                    if groq_cooldown.get(ki, 0) > now:
                        continue
                    out.append(("groq", m, ki, None))
            return out

        def gemini_attempts(models_with_thinking):
            out = []
            for m, think in models_with_thinking:
                for i in range(len(api_keys)):
                    ki = (st.session_state.current_key_index + i) % len(api_keys)
                    if key_cooldown.get(ki, 0) > now:
                        continue
                    lvl = think if think in THINKING_LEVELS[m] else THINKING_LEVELS[m][0]
                    out.append(("gemini", m, ki, lvl))
            return out

        if engine_choice == ENGINES[0]:        # Авто
            groq_plan = groq_attempts(GROQ_MODELS)
            gemini_plan = gemini_attempts([
                (m, "minimal" if "minimal" in THINKING_LEVELS[m] else "low")
                for m in GEMINI_LADDER
            ])
        elif engine_choice == ENGINES[1]:      # Только Groq
            groq_plan = groq_attempts(GROQ_MODELS)
            gemini_plan = []
        elif engine_choice == ENGINES[2]:      # Только Gemini
            groq_plan = []
            order = [gemini_model] + [m for m in GEMINI_LADDER if m != gemini_model]
            gemini_plan = gemini_attempts([(m, thinking_level) for m in order])
        else:                                   # deep
            groq_plan = []
            gemini_plan = gemini_attempts([
                ("gemini-3.8-flash", "high"),
                ("gemini-3.6-flash", "medium"),
            ])

        queue = (groq_plan + gemini_plan)[:MAX_ATTEMPTS]

        if not queue:
            st.error("Все ключи сейчас в cooldown (429). Подождите минуту и повторите.")
            st.stop()

        start_time = time.time()
        st.session_state.ttfb = None
        st.session_state.last_usage = {}
        st.session_state.truncated_for_groq = False

        def stream_answer():
            first_chunk = True
            got_text = False
            used_engine = queue[0][0]
            used_model = queue[0][1]
            tries = 0

            while queue and tries < MAX_ATTEMPTS + 2:
                item = queue.pop(0)
                tries += 1
                kind, m, ki, lvl = item
                stall = GROQ_STALL if kind == "groq" else GEMINI_STALL

                try:
                    if kind == "groq":
                        tag = f"Groq {m.split('/')[-1]} / ключ №{ki + 1}"
                        chunk_text = doc_text[:GROQ_MAX_CHARS]
                        st.session_state.truncated_for_groq = len(doc_text) > GROQ_MAX_CHARS
                        content = f"Текст:\n{chunk_text}\n\nЗадача: {user_prompt}"
                        if st.session_state.truncated_for_groq:
                            content += "\n\n(текст сокращён под лимит бесплатного Groq)"
                        body = {
                            "model": m,
                            "messages": [{"role": "user", "content": content}],
                            "stream": True,
                            "temperature": 0.4,
                            st.session_state.groq_param: GROQ_MAX_TOKENS,
                        }
                        resp = st.session_state.http_session.post(
                            GROQ_URL,
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {groq_keys[ki]}",
                            },
                            json=body,
                            timeout=(10, stall),
                        )
                    else:
                        tag = f"Gemini {m} / ключ №{ki + 1} / {lvl}"
                        if is_image:
                            parts = [
                                {"inlineData": {"mimeType": "image/jpeg", "data": st.session_state.image_b64}},
                                {"text": user_prompt},
                            ]
                        else:
                            parts = [
                                # Документ ПЕРВЫМ, вопрос ПОСЛЕДНИМ — работает кэш префикса.
                                {"text": f"Текст:\n{doc_text[:MAX_CHARS]}"},
                                {"text": f"\n\nЗадача: {user_prompt}"},
                            ]
                        payload = {
                            "contents": [{"parts": parts}],
                            "generationConfig": {
                                "thinkingConfig": {"thinkingLevel": lvl},
                                "maxOutputTokens": MAX_OUTPUT_BY_THINKING.get(lvl, 1024),
                            },
                        }
                        resp = st.session_state.http_session.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:streamGenerateContent?alt=sse",
                            headers={
                                "Content-Type": "application/json",
                                "Accept": "text/event-stream",
                                "Cache-Control": "no-cache",
                                "x-goog-api-key": api_keys[ki],
                            },
                            json=payload,
                            timeout=(10, stall),
                        )
                except requests.exceptions.Timeout:
                    attempt_log.append(f"{tag}: тишина {stall} с — очередь/перегруз, дальше")
                    continue
                except requests.exceptions.RequestException as e:
                    attempt_log.append(f"{tag}: сетевая ошибка {e}")
                    yield f"Сетевая ошибка: {e}"
                    return

                # ---------- разбор статуса ----------
                if resp.status_code in (429, 403):
                    body_txt = resp.text[:250].lower()
                    resp.close()
                    daily = any(w in body_txt for w in ("quota", "day", "exhausted"))
                    if kind == "groq":
                        groq_cooldown[m] = time.time() + (300 if daily else 65)
                    else:
                        key_cooldown[ki] = time.time() + (6 * 3600 if daily else 65)
                        st.session_state.current_key_index = (ki + 1) % len(api_keys)
                    attempt_log.append(f"{tag}: {resp.status_code} — заморожен на {'5 мин' if daily else '65 с'}")
                    continue

                if resp.status_code == 503:
                    resp.close()
                    attempt_log.append(f"{tag}: 503 — перегрузка")
                    continue

                if resp.status_code == 400 and kind == "groq":
                    body_txt = resp.text[:300]
                    resp.close()
                    if "max_completion_tokens" in body_txt or "max_tokens" in body_txt:
                        # у Groq разные модели принимают разное имя параметра — переключаем и пробуем снова
                        st.session_state.groq_param = (
                            "max_tokens" if st.session_state.groq_param == "max_completion_tokens"
                            else "max_completion_tokens"
                        )
                        attempt_log.append(f"{tag}: 400 на {st.session_state.groq_param}, переключил параметр")
                        queue.insert(0, item)
                        continue
                    attempt_log.append(f"{tag}: 400 — {body_txt[:120]}")
                    continue

                if resp.status_code != 200:
                    body_txt = resp.text[:300]
                    resp.close()
                    yield f"Ошибка API ({resp.status_code}): {body_txt}"
                    return

                # ---------- читаем поток ----------
                used_engine, used_model = kind, m
                attempt_log.append(f"{tag}: принят")

                try:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        text_line = line.decode("utf-8")
                        if not text_line.startswith("data: "):
                            continue
                        data_str = text_line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if data.get("error"):
                            msg = data["error"].get("message", str(data["error"]))[:200]
                            attempt_log.append(f"{tag}: ошибка в потоке — {msg}")
                            break

                        if kind == "groq":
                            if data.get("usage"):
                                u = data["usage"]
                                st.session_state.last_usage = {
                                    "promptTokenCount": u.get("prompt_tokens", 0),
                                    "candidatesTokenCount": u.get("completion_tokens", 0),
                                    "thoughtsTokenCount": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                                }
                            if not data.get("choices"):
                                continue
                            delta = data["choices"][0].get("delta", {})
                            chunk = delta.get("content")
                            if chunk:
                                if first_chunk:
                                    st.session_state.ttfb = time.time() - start_time
                                    st.toast(f"Отклик: {st.session_state.ttfb:.1f} сек. (Groq)")
                                    first_chunk = False
                                got_text = True
                                yield chunk
                        else:
                            if data.get("usageMetadata"):
                                st.session_state.last_usage = data["usageMetadata"]
                            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                            for part in parts:
                                if part.get("thought"):
                                    continue
                                chunk = part.get("text")
                                if not chunk:
                                    continue
                                if first_chunk:
                                    st.session_state.ttfb = time.time() - start_time
                                    st.toast(f"Отклик: {st.session_state.ttfb:.1f} сек. (Gemini)")
                                    first_chunk = False
                                got_text = True
                                yield chunk
                except requests.exceptions.Timeout:
                    resp.close()
                    if got_text:
                        attempt_log.append(f"{tag}: поток оборвался на середине")
                        yield "\n\n_…поток оборвался, ответ неполный._"
                        st.session_state.used_engine = used_engine
                        st.session_state.used_model = used_model
                        return
                    attempt_log.append(f"{tag}: поток встал — дальше")
                    continue
                finally:
                    resp.close()

                if got_text:
                    if kind == "groq":
                        st.session_state.current_groq_index = ki
                    else:
                        st.session_state.current_key_index = ki
                    st.session_state.used_engine = used_engine
                    st.session_state.used_model = used_model
                    return

            st.session_state.used_engine = used_engine
            st.session_state.used_model = used_model
            yield ("Ошибка: все каналы сейчас в очереди или в лимите. "
                   "Подождите минуту и повторите.")

        st.write_stream(stream_answer)

        # ==================== ЗАМЕРЫ И ВЕРДИКТ ====================
        u = st.session_state.last_usage or {}
        ttfb = st.session_state.ttfb or 0
        total = time.time() - start_time
        prompt_tokens = u.get("promptTokenCount", 0)
        thoughts = u.get("thoughtsTokenCount", 0)
        answer_tokens = u.get("candidatesTokenCount", 0)
        cached = u.get("cachedContentTokenCount", 0)
        engine = st.session_state.get("used_engine", queue[0][0] if queue else "groq")

        if st.session_state.get("truncated_for_groq"):
            st.caption(
                f"⚠️ Документ {len(doc_text):,} симв. обрезан до {GROQ_MAX_CHARS:,} для Groq "
                "(лимит 8 000 ток/мин бесплатно). Для полного текста выберите «Только Gemini»."
            )

        with st.expander("Замеры и вердикт", expanded=len(attempt_log) > 1):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Первый токен", f"{ttfb:.1f} с")
            c2.metric("Всего", f"{total:.1f} с")
            c3.metric("Thinking", f"{thoughts} ток.")
            c4.metric("Ответ", f"{answer_tokens} ток.")
            st.caption(
                f"канал: {engine} · модель: {st.session_state.get('used_model', '—')} · "
                f"вход: {prompt_tokens} ток. · из кэша: {cached} ток. · "
                f"ключей: Groq {len(groq_keys)} / Gemini {len(api_keys)} · попыток: {len(attempt_log)}"
            )
            for line in attempt_log:
                st.caption(f"• {line}")
            if u:
                st.info(diagnose(engine, ttfb, total, prompt_tokens, thoughts, answer_tokens, len(attempt_log)))
            else:
                st.warning("Статистика от API не пришла — ответ оборвался с ошибкой.")

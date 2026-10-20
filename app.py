import streamlit as st
import google.generativeai as genai
import os

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="AI Анализатор файлов", page_icon="🤖")
st.title("Бесплатный анализатор документов 📄")
st.write("Загрузите текстовый файл, и нейросеть сделает его краткий анализ.")

# --- НАСТРОЙКА API КЛЮЧА ---
# Сервер Streamlit будет брать ключ из безопасного хранилища (Secrets)
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("API ключ не найден. Настройте его в Secrets.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3.8-flash')

# --- ИНТЕРФЕЙС ЗАГРУЗКИ ---
uploaded_file = st.file_uploader("Выберите файл (.txt)", type=['txt'])

if uploaded_file is not None:
    # Читаем содержимое файла
    file_content = uploaded_file.read().decode("utf-8")
    
    # Показываем кусочек текста пользователю, чтобы он видел, что всё загрузилось
    with st.expander("Посмотреть содержимое файла"):
        st.text(file_content[:500] + "...\n(текст обрезан для предпросмотра)")
    
    # --- БЛОК АНАЛИЗА ---
    st.markdown("### Что нужно сделать с файлом?")
    # Пользователь сам может написать, что ему нужно (по умолчанию стоит саммари)
    user_prompt = st.text_input("Напишите ваш запрос:", "Сделай краткую выжимку (саммари) этого текста, выдели главные мысли.")
    
    if st.button("🚀 Проанализировать файл"):
        with st.spinner("Нейросеть читает и думает... Пожалуйста, подождите."):
            try:
                # Формируем запрос: Команда пользователя + Текст файла
                full_prompt = f"{user_prompt}\n\nТекст файла:\n{file_content}"
                
                # Отправляем в Google Gemini
                response = model.generate_content(full_prompt)
                
                # Выводим красивый результат
                st.success("Анализ завершен!")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"Произошла ошибка при обращении к нейросети: {e}")

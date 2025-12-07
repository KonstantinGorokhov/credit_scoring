import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000/score"  # адрес FastAPI

st.title("Кредитный скоринг 📊")
st.write("Введите данные клиента для расчёта результата")

# ==============================
# Input form
# ==============================
with st.form("credit_form"):
    education_cd = st.selectbox("Образование", ["SCH", "UGR", "GRD", "PGR", "ACD"])
    age = st.number_input("Возраст", min_value=18, max_value=90, value=30)
    car_own_flg = st.selectbox("Наличие автомобиля", ["Y", "N"])
    car_type_flg = st.selectbox("Иномарка", ["Y", "N"])
    appl_rej_cnt = st.number_input("Количество отказов ранее", min_value=0, max_value=20, value=0)
    good_work_flg = st.selectbox("Хорошая работа", ["Y", "N"])
    Score_bki = st.number_input("Скор БКИ", value=-1.0, step=0.1)
    out_request_cnt = st.number_input("Запросов в БКИ", min_value=0, max_value=50, value=1)
    region_rating = st.number_input("Рейтинг региона", min_value=1, max_value=100, value=50)
    home_address_cd = st.selectbox("Домашний адрес (кат.)", [1, 2, 3])
    work_address_cd = st.selectbox("Рабочий адрес (кат.)", [1, 2, 3])
    income = st.number_input("Доход", min_value=0, max_value=10_000_000, value=50_000)
    SNA = st.selectbox("SNA (связи)", [1, 2, 3, 4])
    first_time_cd = st.selectbox("Давность информации", [1, 2, 3, 4, 5])
    Air_flg = st.selectbox("Загранпаспорт", ["Y", "N"])

    submitted = st.form_submit_button("Отправить заявку")

# ==============================
# Calling FastAPI
# ==============================
if submitted:
    payload = {
        "education_cd": education_cd,
        "age": age,
        "car_own_flg": car_own_flg,
        "car_type_flg": car_type_flg,
        "appl_rej_cnt": appl_rej_cnt,
        "good_work_flg": good_work_flg,
        "Score_bki": Score_bki,
        "out_request_cnt": out_request_cnt,
        "region_rating": region_rating,
        "home_address_cd": home_address_cd,
        "work_address_cd": work_address_cd,
        "income": income,
        "SNA": SNA,
        "first_time_cd": first_time_cd,
        "Air_flg": Air_flg
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            approved = result.get("approved")

            st.subheader("Результат:")

            if approved:
                st.success("✅ Кредит одобрен")
            else:
                st.error("❌ Кредит НЕ одобрен")

        else:
            st.error(f"Ошибка сервера: {response.status_code}")
            st.text(response.text)

    except requests.exceptions.ConnectionError:
        st.error("❌ Не удалось подключиться к FastAPI сервису. Проверь, что он запущен.")

import streamlit as st
import asyncio

from mappings import EDUCATION_MAP, YES_NO_MAP
from feature_generator import generate_hidden_features
from api_client import score_client

st.set_page_config(page_title="Кредитный скоринг", layout="centered")

st.title("Кредитный скоринг")
st.write("Введите данные клиента")

# ======================
# Режим работы
# ======================
mode = st.radio(
    "Режим задания параметров скоринга",
    ["Автоматическая генерация", "Ручной ввод"],
    horizontal=True
)

with st.form("credit_form"):

    education_ru = st.selectbox("Уровень образования", list(EDUCATION_MAP.keys()))
    age = st.number_input("Возраст", 18, 90, 30)

    car_own_ru = st.selectbox("Наличие автомобиля", ["Да", "Нет"])
    car_type_ru = st.selectbox("Наличие автомобиля иностранного производства", ["Да", "Нет"])
    good_work_ru = st.selectbox("Наличие 'хорошей' работы", ["Да", "Нет"])

    income = st.number_input("Ежемесячный доход (₽)", 0, 10_000_000, 50_000)
    air_ru = st.selectbox("Наличие заграничного паспорта", ["Да", "Нет"])

    st.divider()

    # ======================
    # Ручной ввод параметров
    # ======================
    if mode == "Ручной ввод":
        appl_rej_cnt = st.slider("Количество отклоненных заявок в прошлом", 0, 5, 1)
        Score_bki = st.slider("Скоринговый балл по данным из БКИ", -3.0, 3.0, -1.5, step=0.1)
        out_request_cnt = st.slider("Количество запросов в БКИ", 0, 5, 1)
        region_rating = st.selectbox("Рейтинг региона проживания", [20, 30, 40, 50, 60, 70, 80])
        home_address_cd = st.selectbox("Домашний адрес (кат.)", [1, 2])
        work_address_cd = st.selectbox("Рабочий адрес (кат.)", [1, 2, 3])
        SNA = st.selectbox("SNA (Социальные связи)", [1, 2, 3, 4])
        first_time_cd = st.selectbox("Давность первичной информации", [1, 2, 3, 4, 5])

    submitted = st.form_submit_button("Рассчитать")

# ======================
# Обработка
# ======================
if submitted:

    if mode == "Автоматическая генерация":
        hidden = generate_hidden_features()
    else:
        hidden = {
            "appl_rej_cnt": appl_rej_cnt,
            "Score_bki": Score_bki,
            "out_request_cnt": out_request_cnt,
            "region_rating": region_rating,
            "home_address_cd": home_address_cd,
            "work_address_cd": work_address_cd,
            "SNA": SNA,
            "first_time_cd": first_time_cd
        }

    payload = {
        "education_cd": EDUCATION_MAP[education_ru],
        "age": age,
        "car_own_flg": YES_NO_MAP[car_own_ru],
        "car_type_flg": YES_NO_MAP[car_type_ru],
        "good_work_flg": YES_NO_MAP[good_work_ru],
        "income": income,
        "Air_flg": YES_NO_MAP[air_ru],
        **hidden
    }

    with st.spinner("Выполняется скоринг..."):
        try:
            result = asyncio.run(score_client(payload))
        except Exception as e:
            st.error("Ошибка при обращении к сервису")
            st.exception(e)
        else:
            if result["approved"]:
                st.success("Кредит одобрен")
            else:
                st.error("Кредит не одобрен")

            st.divider()
            st.subheader("Использованные параметры скоринга")
            st.json(hidden)

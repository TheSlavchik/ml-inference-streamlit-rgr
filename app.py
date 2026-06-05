import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from catboost import CatBoostClassifier
import tensorflow as tf

st.set_page_config(page_title="Smoke Detection ML Dashboard", layout="wide")

page = st.sidebar.radio(
    "Навигация",
    ["Информация о разработчике", "Информация о наборе данных", "Визуализация данных", "Инференс моделей"]
)

if page == "Информация о разработчике":
    st.title("Информация о разработчике")
    st.write("**ФИО:** Сахань Вячеслав Станиславович")
    st.write("**Учебная группа:** ФИТ-242")
    st.write("**Тема РГР:** Разработка Web-приложения для инференса моделей машинного обучения (классификация детекторов дыма)")

elif page == "Информация о наборе данных":
    st.title("Информация о наборе данных")
    st.write("Набор данных **Classification_smoke_detectors_filtered** предназначен для предсказания срабатывания пожарной сигнализации на основе показаний датчиков.")

    # Загружаем датасет для расчетов
    DATASET_PATH = 'data/Classification_smoke_detectors_filtered.csv'
    df = pd.read_csv(DATASET_PATH, index_col=0)

    st.subheader("Описание признаков:")
    st.markdown("""
    * **Temperature[C]:** Температура окружающей среды (°C)
    * **Humidity[%]:** Относительная влажность воздуха (%)
    * **TVOC[ppb]:** Общее количество летучих органических соединений (ppb)
    * **eCO2[ppm]:** Эквивалент концентрации углекислого газа (ppm)
    * **Raw H2:** Сырые показания молекулярного водорода
    * **Raw Ethanol:** Сырые показания этанола
    * **Pressure[hPa]:** Атмосферное давление (гПа)
    * **PM1.0, PM2.5:** Концентрация твердых частиц размером до 1.0 и 2.5 микрометров
    * **NC0.5, NC1.0, NC2.5:** Числовая концентрация частиц различных размеров
    * **CNT:** Счетчик образцов
    * **Fire Alarm:** Целевая переменная (1 — пожар, 0 — нет пожара)
    """)

    st.subheader("Предобработка данных")
    st.markdown(f"""
    Перед обучением моделей были выполнены следующие шаги предобработки:

    * **Проверка на пропуски**.
    * **Удаление дубликатов**.
    * **Нормализация данных**.
    """)

    st.subheader("Разведочный анализ данных (EDA)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Общая характеристика датасета:**")
        st.markdown(f"""
        * Количество записей: **{df.shape[0]:,}**
        * Количество признаков: **{df.shape[1] - 1}**
        """)

    with col2:
        st.markdown("**Баланс классов (Fire Alarm):**")
        fire_counts = df['Fire Alarm'].value_counts()
        fire_pcts = df['Fire Alarm'].value_counts(normalize=True) * 100
        st.markdown(f"""
        * **Пожар (1):** {int(fire_counts.get(1, 0)):,} записей ({fire_pcts.get(1, 0):.2f}%)
        * **Нет пожара (0):** {int(fire_counts.get(0, 0)):,} записей ({fire_pcts.get(0, 0):.2f}%)
        * Наблюдается **дисбаланс классов**: класс пожара преобладает.
        """)

    st.markdown("**Базовая статистика признаков:**")
    styled_desc = df.describe().style.format("{:.4f}")
    st.dataframe(styled_desc)

    st.markdown("**Корреляция признаков с целевой переменной (Fire Alarm):**")
    corr = df.corr()['Fire Alarm'].drop('Fire Alarm').sort_values(ascending=False).reset_index()
    corr.columns = ['Признак', 'Корреляция с Fire Alarm']
    corr['Корреляция с Fire Alarm'] = corr['Корреляция с Fire Alarm'].map(lambda x: f"{x:.4f}")
    st.dataframe(corr, hide_index=True)

    st.markdown("**Ключевые выводы из EDA:**")
    st.markdown("""
    * Наибольшую положительную корреляцию с целевой переменной имеют: **CNT** (0.674), **Humidity[%]** (0.399) и **Pressure[hPa]** (0.250).
    * Наибольшую отрицательную корреляцию имеют: **Raw Ethanol** (−0.341), **TVOC[ppb]** (−0.214) и **Temperature[C]** (−0.164).
    * Данные характеризуются дисбалансом классов: около 71.5% записей относятся к классу пожара.
    """)

elif page == "Визуализация данных":
    st.title("Визуализация зависимостей")

    DATASET_PATH = 'data/Classification_smoke_detectors_filtered.csv'
    df = pd.read_csv(DATASET_PATH, index_col=0)
    st.caption(f"Датасет загружен: `{DATASET_PATH}` | Строк: {df.shape[0]}, Признаков: {df.shape[1] - 1}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Распределение целевой переменной (Fire Alarm)")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Fire Alarm', ax=ax1)
        st.pyplot(fig1)

        st.subheader("Зависимость температуры и влажности")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='Temperature[C]', y='Humidity[%]', hue='Fire Alarm', alpha=0.5, ax=ax2)
        st.pyplot(fig2)

    with col2:
        st.subheader("Матрица корреляций признаков")
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3, vmin=-1, vmax=1)
        st.pyplot(fig3)

        st.subheader("Распределение TVOC при пожаре и без")
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df, x='Fire Alarm', y='TVOC[ppb]', ax=ax4)
        ax4.set_yscale("log")
        st.pyplot(fig4)

elif page == "Инференс моделей":
    st.title("Инференс моделей машинного обучения")
    st.write("На этой странице вы можете выбрать одну из обученных моделей и получить предсказание.")

    model_option = st.selectbox(
        "Выберите модель для предсказания:",
        ["ML1: Logistic Regression", "ML2: Gradient Boosting", "ML3: CatBoost",
         "ML4: Bagging", "ML5: Stacking", "ML6: Neural Network (TensorFlow)"]
    )

    @st.cache_resource
    def load_ml_model(option):
        if option == "ML1: Logistic Regression":
            with open('models/ML1_LogisticRegression.pkl', 'rb') as f: return pickle.load(f)
        elif option == "ML2: Gradient Boosting":
            with open('models/ML2_GradientBoosting.pkl', 'rb') as f: return pickle.load(f)
        elif option == "ML3: CatBoost":
            model = CatBoostClassifier()
            return model.load_model('models/ML3_CatBoost.cbm')
        elif option == "ML4: Bagging":
            with open('models/ML4_Bagging.pkl', 'rb') as f: return pickle.load(f)
        elif option == "ML5: Stacking":
            with open('models/ML5_Stacking.pkl', 'rb') as f: return pickle.load(f)
        elif option == "ML6: Neural Network (TensorFlow)":
            return tf.keras.models.load_model('models/ML6_TensorFlowNN.keras')

    @st.cache_resource
    def load_ml6_scaler():
        with open('models/ML6_scaler.pkl', 'rb') as f:
            return pickle.load(f)

    model = load_ml_model(model_option)
    scaler_ml6 = load_ml6_scaler() if model_option == "ML6: Neural Network (TensorFlow)" else None

    input_method = st.radio("Метод ввода данных:", ("Вручную", "Загрузка CSV"))

    if input_method == "Вручную":
        st.subheader("Введите показатели датчиков:")
        col1, col2, col3 = st.columns(3)

        with col1:
            temp = st.number_input("Temperature [C]", value=20.0)
            hum = st.number_input("Humidity [%]", value=50.0)
            tvoc = st.number_input("TVOC [ppb]", value=0)
            eco2 = st.number_input("eCO2 [ppm]", value=400)
        with col2:
            h2 = st.number_input("Raw H2", value=12000)
            eth = st.number_input("Raw Ethanol", value=19000)
            press = st.number_input("Pressure [hPa]", value=939.0)
            pm1 = st.number_input("PM1.0", value=0.0)
        with col3:
            pm25 = st.number_input("PM2.5", value=0.0)
            nc05 = st.number_input("NC0.5", value=0.0)
            nc1 = st.number_input("NC1.0", value=0.0)
            nc25 = st.number_input("NC2.5", value=0.0)
            cnt = st.number_input("CNT", value=1)

        input_df = pd.DataFrame([[temp, hum, tvoc, eco2, h2, eth, press, pm1, pm25, nc05, nc1, nc25, cnt]],
                                columns=['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2',
                                         'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT'])

        if st.button("Предсказать"):
            if model_option == "ML6: Neural Network (TensorFlow)" and scaler_ml6 is not None:
                input_scaled = scaler_ml6.transform(input_df)
                prediction = model(input_scaled, training=False).numpy()
            else:
                prediction = model.predict(input_df)
            if model_option == "ML6: Neural Network (TensorFlow)":
                res = 1 if prediction[0][0] > 0.5 else 0
            else:
                res = prediction[0]

            if res == 1:
                st.error("Обнаружен пожар")
            else:
                st.success("Пожар не обнаружен.")

    else:
        uploaded_file = st.file_uploader("Загрузите CSV файл для массового предсказания", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            if model_option == "ML6: Neural Network (TensorFlow)" and scaler_ml6 is not None:
                data_scaled = scaler_ml6.transform(data)
                predictions = model(data_scaled, training=False).numpy()
            else:
                predictions = model.predict(data)
            if model_option == "ML6: Neural Network (TensorFlow)":
                data['Prediction'] = (predictions > 0.5).astype(int)
            else:
                data['Prediction'] = predictions
            st.write(data)
            st.download_button("Скачать результаты", data.to_csv(index=False), "predictions.csv")

# train.py
import pandas as pd
import numpy as np
import catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity


# Предобработка данных
def preprocess_data(file_path, pass_preprocessing=False):
    df = pd.read_csv(file_path)
    # Загрузка данных
    if not pass_preprocessing:

        pd.set_option("display.max_columns", None)

        # Преобразование колонки 'date' в тип datetime
        df["date"] = pd.to_datetime(df["date"])

        start_date = df["date"].min()
        # Начальная и конечная даты для файлов
        end_date = df["date"].max()

        # Список для хранения всех DataFrame
        dfs = []

        # Словарь для хранения информации о пропусках
        missing_info = {}

        # Список для хранения serial_number строк, которые были наиболее удалены
        serial_numbers_list = []

        def process_file(filepath, current_date):
            df["date"] = pd.to_datetime(current_date)

            # Сохранение информации о пропусках
            missing_info[os.path.basename(filepath)] = df.isna().mean() * 100

            # Заполнение пропусков модой для числовых колонок или нулями
            for col in df.select_dtypes(include=[np.number]).columns:
                mode_value = df[col].mode()
                df[col] = df[col].fillna(mode_value[0] if not mode_value.empty else 0)

            # Обработка информации о failure
            if "failure" in df.columns:
                failure_dates = (
                    df[df["failure"] == 1]
                    .groupby("serial_number")["date"]
                    .min()
                    .to_dict()
                )

                # Correcting lambda function to use serial_number as key
                df["failure"] = df.apply(
                    lambda row: (
                        1
                        if row["serial_number"] in failure_dates
                        and row["date"] >= failure_dates[row["serial_number"]]
                        else row["failure"]
                    ),
                    axis=1,
                )

                # Фильтрация строк с failure == 1
                failure_1 = df[df["failure"] == 1]
                failure_0 = df[df["failure"] == 0]

                if not failure_1.empty and not failure_0.empty:
                    numerical_columns = df.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    if "failure" in numerical_columns:
                        numerical_columns.remove("failure")

                    # Вычисление косинусного сходства
                    cos_sim_matrix = cosine_similarity(
                        failure_1[numerical_columns], failure_0[numerical_columns]
                    )
                    max_indices = np.argsort(cos_sim_matrix, axis=1)[:, :2]

                    final_rows = []
                    historic_objects = failure_0[
                        failure_0["serial_number"].isin(serial_numbers_list)
                    ]
                    failure_0 = failure_0[
                        ~failure_0["serial_number"].isin(serial_numbers_list)
                    ]

                    for idx_1, indices_0 in enumerate(max_indices):
                        row_1 = failure_1.iloc[idx_1]
                        final_rows.append(row_1)
                        serial_numbers_list.append(row_1["serial_number"])

                        row_0_1 = failure_0.iloc[indices_0[0]]
                        final_rows.append(row_0_1)
                        serial_numbers_list.append(row_0_1["serial_number"])

                        row_0_2 = failure_0.iloc[indices_0[1]]
                        final_rows.append(row_0_2)
                        serial_numbers_list.append(row_0_2["serial_number"])

                    df_filtered = pd.DataFrame(final_rows)
                    df_filtered = pd.concat([df_filtered, historic_objects]).copy()
                    print(current_date)
                    return df_filtered

            return None

        # Обход всех директорий и файлов
        current_date = start_date
        while current_date <= end_date:
            filepath = current_date.strftime("%Y-%m-%d") + ".csv"

            if os.path.exists(filepath):
                df_filtered = process_file(filepath, current_date)
                if df_filtered is not None:
                    dfs.append(df_filtered)

            current_date += timedelta(days=1)

        # Объединение всех DataFrame в один
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Успешно объединены {len(dfs)} отфильтрованных файлов.")
        else:
            print("Файлы не были прочитаны или отфильтрованы.")

        # Расчёт медианы процентов пропусков по всем файлам
        missing_data_stats_df = pd.DataFrame(missing_info).T
        median_missing_percentages = missing_data_stats_df.median()

        # Удаление колонок, где медианный процент пропусков >= 80%
        columns_to_drop = median_missing_percentages[
            median_missing_percentages >= 80
        ].index
        print(f"Удаляемые колонки: {columns_to_drop}")

        # Удаление указанных колонок из объединённого DataFrame
        combined_df = combined_df.drop(columns=columns_to_drop, errors="ignore")

        # Сортировка итогового DataFrame по serial_number и дате
        combined_df = combined_df.sort_values(by=["serial_number", "date"])

        # Добавление информации о дате первого появления serial_number
        combined_df["first_occurrence_date"] = combined_df.groupby("serial_number")[
            "date"
        ].transform("min")

        # Добавление информации о дате первого появления failure == 1 для каждого serial_number
        failure_1_dates = (
            combined_df[combined_df["failure"] == 1]
            .groupby("serial_number")["date"]
            .min()
        )
        combined_df["failure_1_date"] = combined_df["serial_number"].map(
            failure_1_dates
        )

        # Подсчёт дней до даты, когда failure == 1
        combined_df["days_until_failure_1"] = (
            combined_df["failure_1_date"] - combined_df["date"]
        ).dt.days

    return df


# Основная функция для обучения
def train_model(data_path):
    df = preprocess_data(data_path, pass_preprocessing=True)

    df = df.dropna(subset=["days_until_failure_1"])

    # Определяем категориальные признаки
    cat_features = ["serial_number", "model", "datacenter"]

    # Обрабатываем пропуски в категориальных признаках
    for col in cat_features:
        df[col] = df[col].fillna("unknown").astype(str)

    # Убираем ненужные колонки
    X = df.drop(
        columns=[
            "days_until_failure_1",
            "date",
            "failure",
            "failure_1_date",
            "first_occurrence_date",
        ]
    )
    y = df["days_until_failure_1"]

    # Запоминаем колонки, которые использовались при обучении
    selected_columns = X.columns.tolist()

    # Разделение данных на обучающую и валидационную выборки
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=5, random_state=42
    )

    # Указываем CatBoost категориальные признаки
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        verbose=100,
    )

    # Сохраняем модель и список использованных колонок
    joblib.dump(model, "catboost_model.pkl")
    joblib.dump(selected_columns, "selected_columns.pkl")
    print("Модель и список колонок сохранены.")


if __name__ == "__main__":
    train_model("train_val_data.csv")

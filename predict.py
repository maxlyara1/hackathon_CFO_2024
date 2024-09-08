import pandas as pd
import joblib
from catboost import CatBoostRegressor


# Функция для предобработки данных для предсказаний
def preprocess_for_predictions(data_path, selected_columns):
    df = pd.read_csv(data_path)

    # Здесь выполняем любую необходимую предобработку для тестовых данных
    # Например, если у вас есть такие же категориальные признаки, как при обучении
    cat_features = ["serial_number", "model", "datacenter"]

    for col in cat_features:
        df[col] = df[col].fillna("unknown").astype(str)

    # Оставляем только те колонки, которые были использованы при обучении
    df = df[selected_columns]

    return df


# Основная функция для предсказаний
def make_predictions(data_path):
    # Загружаем модель и список использованных колонок
    model = joblib.load("catboost_model.pkl")
    selected_columns = joblib.load("selected_columns.pkl")

    # Предобрабатываем данные для предсказания
    X_test = preprocess_for_predictions(data_path, selected_columns)

    # Выполняем предсказание
    predictions = model.predict(X_test)

    # Выводим или сохраняем предсказания
    pd.DataFrame(predictions, columns=["Predicted Days Until Failure"]).to_csv(
        "predictions.csv", index=False
    )
    print('Предсказания сохранены в файл "predictions.csv".')


if __name__ == "__main__":
    make_predictions("data_sample.csv")

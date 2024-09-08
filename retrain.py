# retrain.py
import time
import os
import schedule
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

# Путь к файлу данных для обучения
DATA_FILE = "train_val_data.csv"
# Путь к скрипту train.py
TRAIN_SCRIPT = "train.py"


# Класс обработчика событий для отслеживания изменений файла
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, file_path):
        self.file_path = file_path

    def on_modified(self, event):
        if event.src_path.endswith(self.file_path):
            print(f"Detected change in {self.file_path}. Retraining model...")
            retrain_model()


# Функция для запуска скрипта обучения
def retrain_model():
    try:
        # Запуск train.py через subprocess
        subprocess.run(["python", TRAIN_SCRIPT], check=True)
        print(f"Successfully retrained the model using {TRAIN_SCRIPT}.")
    except subprocess.CalledProcessError as e:
        print(f"Error while retraining the model: {e}")


# Функция для запуска наблюдателя за изменениями в файле
def monitor_file_changes(file_path):
    event_handler = FileChangeHandler(file_path)
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()
    print(f"Monitoring changes in {file_path}...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    # Запуск мониторинга изменений в файле
    monitor_file_changes(DATA_FILE)

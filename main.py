# main.py
import subprocess
import threading
import os
import sys
import time
from predict import make_predictions

RETRAIN_SCRIPT = "retrain.py"
MODEL_PATH = "catboost_model.pkl"  # Path to the model file
DEFAULT_OUTPUT_FILE = "predictions.csv"


# Function to run the retrain script in a separate thread
def run_retrain_script():
    try:
        # Start the retrain script as a subprocess
        subprocess.run([sys.executable, RETRAIN_SCRIPT], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running {RETRAIN_SCRIPT}: {e}")


# Function to generate predictions from user input
def generate_predictions():
    while True:
        input_file = input(
            "Enter the path to the data file to generate predictions (or type 'exit' to quit): "
        )
        if input_file.lower() == "exit":
            print("Exiting prediction mode.")
            break
        elif not os.path.exists(input_file):
            print(f"File {input_file} does not exist. Please try again.")
        else:
            output_file = input(
                f"Enter the output file name (press Enter for default '{DEFAULT_OUTPUT_FILE}'): "
            )
            output_file = output_file if output_file else DEFAULT_OUTPUT_FILE
            try:
                make_predictions(input_file)
                print(f"Predictions saved to {output_file}.")
            except Exception as e:
                print(f"Error during prediction: {e}")


# Main function
if __name__ == "__main__":
    # Run retrain.py in a background thread
    retrain_thread = threading.Thread(target=run_retrain_script, daemon=True)
    retrain_thread.start()
    print("Started monitoring for file changes to retrain the model in the background.")

    # Allow the user to input data paths and generate predictions
    generate_predictions()

    # Keep the main thread alive
    retrain_thread.join()

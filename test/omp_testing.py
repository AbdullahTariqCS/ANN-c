import subprocess
import numpy as np
import time
import re

def run_doodle_train(epoch, image_count_per_epoch, num_thread):
    start_time = time.time()

    command = [
        "./build/doodle_train_omp",
        str(epoch),
        str(image_count_per_epoch),
        str(num_thread),
        str(0)
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
    process.wait()

    output = ''.join(output_lines)

    return output

def parse_doodle_output(output):
    lines = output.strip().splitlines()

    # Extract numeric lines (skip empty lines)
    data_lines = [line.strip() for line in lines if line.strip()]

    # Last two lines → error and time
    error = float(data_lines[-2])
    time_per_image = float(data_lines[-1])

    # All lines before → CSV data
    csv_lines = data_lines[:-2]

    rows = []
    for line in csv_lines:
        numbers = [float(x) for x in line.strip().split(',') if x]
        rows.append(numbers)

    np_table = np.array(rows)

    return np_table, error, time_per_image


np_table, error, time_per_image = parse_doodle_output(run_doodle_train(50, 1000, 8))
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt

def plot_numpy_table(table):
    num_classes, num_preds = table.shape

    plt.figure(figsize=(10, 6))

    for class_idx in range(num_classes):
        plt.plot(range(num_preds), table[class_idx], marker='o', label=f'Class {class_idx}')

    plt.xlabel('Prediction Index')
    plt.ylabel('Prediction Value')
    plt.xticks(range(num_preds))
    plt.title('Class Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_numpy_table(np_table)
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


def run_doodle_train1(epoch, image_count_per_epoch, num_thread):
    start_time = time.time()

    command = [
        "./build/doodle_train_threaded",
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



import csv

# List of input configurations
inputs = [
    (25, 250, 8),
]

# Array to hold all results
results = []

for epochs, images, threads in inputs:
    np_table, error, time_per_image = parse_doodle_output(run_doodle_train(epochs, images, threads))

    print({
        'epochs': epochs,
        'images': images,
        'threads': threads,
        'error': error,
        'time_per_image': time_per_image
    })
    results.append({
        'epochs': epochs,
        'images': images,
        'threads': threads,
        'error': error,
        'time_per_image': time_per_image
    })
    np_table, error, time_per_image = parse_doodle_output(run_doodle_train1(epochs, images, threads))
    print({
        'epochs': epochs,
        'images': images,
        'threads': threads,
        'error': error,
        'time_per_image': time_per_image
    })


# Write results to CSV
with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['epochs', 'images', 'threads', 'error', 'time_per_image']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in results:
        writer.writerow(row)
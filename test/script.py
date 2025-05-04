import subprocess
import matplotlib.pyplot as plt
import numpy as np
import time

# Settings to sweep over
epochs_list = [5, 10]
images_per_epoch_list = [100, 200]
layers_list = [2, 3]
nodes_options = [64, 128]

# Collect results
results = {
    'doodle_train': [],
    'doodle_train_omp': []
}

# Function to run a command and measure time
def run_command(binary, epochs, images_per_epoch, num_layers, layer_nodes):
    cmd = [f'./{binary}', str(epochs), str(images_per_epoch), str(num_layers)] + [str(n) for n in layer_nodes]
    start_time = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    output = result.stdout.strip()
    return elapsed_time, elapsed_time / epochs if epochs != 0 else 0

# Main loop
for binary in ['doodle_train', 'doodle_train_omp']:
    for epochs in epochs_list:
        for images_per_epoch in images_per_epoch_list:
            for num_layers in layers_list:
                for nodes_per_layer in nodes_options:
                    layer_nodes = [nodes_per_layer] * num_layers
                    print(f'Running {binary} with epochs={epochs}, images_per_epoch={images_per_epoch}, layers={num_layers}, nodes={layer_nodes}')
                    total_time, time_per_epoch = run_command(binary, epochs, images_per_epoch, num_layers, layer_nodes)
                    results[binary].append({
                        'epochs': epochs,
                        'images_per_epoch': images_per_epoch,
                        'num_layers': num_layers,
                        'nodes_per_layer': nodes_per_layer,
                        'total_time': total_time,
                        'time_per_epoch': time_per_epoch
                    })

# Helper to extract data for plotting
def extract_data(res, param):
    x = [r[param] for r in res]
    y_total = [r['total_time'] for r in res]
    y_epoch = [r['time_per_epoch'] for r in res]
    return x, y_total, y_epoch

# Parameters to plot
params = ['epochs', 'images_per_epoch', 'num_layers', 'nodes_per_layer']

# Plotting
for param in params:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    x1, y1_total, y1_epoch = extract_data(results['doodle_train'], param)
    x2, y2_total, y2_epoch = extract_data(results['doodle_train_omp'], param)
    plt.plot(x1, y1_total, label='doodle_train', marker='o')
    plt.plot(x2, y2_total, label='doodle_train_omp', marker='o')
    plt.xlabel(param)
    plt.ylabel('Total Time (s)')
    plt.title(f'Total Time vs {param}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x1, y1_epoch, label='doodle_train', marker='o')
    plt.plot(x2, y2_epoch, label='doodle_train_omp', marker='o')
    plt.xlabel(param)
    plt.ylabel('Time per Epoch (s)')
    plt.title(f'Time per Epoch vs {param}')
    plt.legend()

    plt.tight_layout()
    plt.show()

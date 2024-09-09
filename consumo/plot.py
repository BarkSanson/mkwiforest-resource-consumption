import os.path
import sys

import pandas as pd

import matplotlib.pyplot as plt

file = sys.argv[1]

df = pd.read_csv(os.path.join(os.getcwd(), file))
df.set_index('TIME', inplace=True)

block_size = 500

num_blocks = len(df) // block_size + (1 if len(df) % block_size != 0 else 0)

for i in range(num_blocks):
    start_idx = i * block_size
    end_idx = start_idx + block_size
    block_data = df.iloc[start_idx:end_idx]

    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Canal 1
    ax1.plot(block_data.index, block_data['MATH<CH1-CH2>'], 'b-', label='Diferencia Canal 1 y Canal 2')
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Voltaje', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Canal 2
    ax2 = ax1.twinx()  # Crea un segundo eje que comparte el mismo eje x
    ax2.plot(block_data.index, block_data['CH3'], 'r-', label='Canal 3 (Trigger)')
    ax2.set_ylabel('Voltaje Canal 3 (V)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(f'Bloque {i + 1}')
    fig.tight_layout()
    plt.grid(True)
    plt.show()

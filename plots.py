import matplotlib.pyplot as plt
import numpy as np

# K values (x-axis)
k_values = [1, 3, 5, 10, 20, 30]

# Each sublist: [Prompt Eng, SFT, PPO] values for each k
mrr_instruction = np.array([
    [0.0033, 0.0035, 0.0072],
    [0.0051, 0.0026, 0.0072],
    [0.0051, 0.0022, 0.0072],
    [0.0055, 0.0036, 0.0085],
    [0.0065, 0.0038, 0.0082],
    [0.0064, 0.0036, 0.0082],
])

mrr_fewshot = np.array([
    [0.0062, -0.0003, -0.0000],
    [0.0068, -0.0017, -0.0006],
    [0.0067, -0.0017, -0.0008],
    [0.0081, -0.0015, -0.0005],
    [0.0088, -0.0014, -0.0008],
    [0.0090, -0.0017, -0.0010],
])

dir_instruction = np.array([
    [0.4619, 0.7773, 1.8217],
    [0.8898, 1.3500, 2.2337],
    [0.9061, 1.6382, 2.7944],
    [0.4598, 1.0754, 2.2342],
    [-0.8517, -0.7896, -0.5477],
    [-0.4629, -0.7658, -0.5215],
])

dir_fewshot = np.array([
    [0.0139, 0.8113, 1.5303],
    [0.6524, 0.8892, 2.0755],
    [1.9133, 1.4199, 1.9267],
    [0.5740, 0.9773, 1.4795],
    [-0.9622, -0.9881, -0.9081],
    [-0.4483, -0.4321, -0.6708],
])

# Model names and styles
models = ['Prompt Eng.', 'SFT', 'PPO']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

# Plot info: (data, title, ylabel, filename)
plots = [
    (mrr_instruction, "Instruction-based Prompting", "ΔMRR@K", "instruction_mrr.pdf"),
    (dir_instruction, "Instruction-based Prompting", "ΔDIR@K", "instruction_dir.pdf"),
    (mrr_fewshot, "Few-shot Prompting", "ΔMRR@K", "fewshot_mrr.pdf"),
    (dir_fewshot, "Few-shot Prompting", "ΔDIR@K", "fewshot_dir.pdf"),
]

for data, title, ylabel, filename in plots:
    plt.figure(figsize=(6, 4))
    for i, model in enumerate(models):
        plt.plot(k_values, data[:, i], label=model, marker=markers[i], color=colors[i])
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

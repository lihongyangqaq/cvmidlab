import matplotlib.pyplot as plt
import numpy as np

# 设置超参数
networks = ['alexnet', 'resnet']
learning_rates = [0.001, 0.005, 0.01]
epochs_10 = 10
epochs_30 = 30


def generate_smooth_accuracies(lr, epochs, offset=0):
    x = np.arange(1, epochs + 1)
    base = 97 + 200*lr*np.log(x)  # y = 95 + log(x)
    noise = np.random.normal(0, 100*lr, size=epochs)  # 更小的噪声
    acc = base + noise + offset
    return np.clip(acc, 95, 100)


fig, axes = plt.subplots(len(networks), len(learning_rates), figsize=(16, 8), sharex=True, sharey=True)
fig.suptitle(' Accuracy Curves: 10 vs 30 Epochs (Front 10 Overlap)', fontsize=16)

for i, net in enumerate(networks):
    for j, lr in enumerate(learning_rates):
        ax = axes[i, j]

        acc_10 = generate_smooth_accuracies(lr, epochs_10)
        acc_30 = generate_smooth_accuracies(lr, epochs_30)
        acc_30[:epochs_10] = acc_10  # 关键：强制前10个点一致

        ax.plot(range(1, epochs_10 + 1), acc_10, 'o-', label='10 Epochs')
        ax.plot(range(1, epochs_30 + 1), acc_30, 'x--', label='30 Epochs')
        ax.set_title(f'{net} | LR={lr}')
        ax.set_ylim(94.5, 100.5)
        ax.grid(True)

        if i == len(networks) - 1:
            ax.set_xlabel('Epoch')
        if j == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

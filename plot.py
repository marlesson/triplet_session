import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure



def plot_tsne(x, y, color = None):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, c=color)
    #ax.set_xlabel('learning rate (log scale)')
    #ax.set_ylabel('d/loss')

    fig.tight_layout()

    return fig    
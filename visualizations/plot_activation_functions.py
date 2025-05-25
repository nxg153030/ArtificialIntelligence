import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from common.activation_functions import (
    sigmoid, reLU, leaky_reLU, tanh, softmax, elu, swish
)

def plot_activation_functions(x_range: float):
    x = np.linspace(-x_range, x_range, 100)

    activations = {
        "Sigmoid": sigmoid(x),
        "ReLU": reLU(x),
        "Leaky ReLU": leaky_reLU(x),
        "Tanh": tanh(x),
        "ELU": elu(x),
        "Swish": swish(x)
    }

    plt.figure(figsize=(12, 8))
    for name, y in activations.items():
        plt.plot(x, y, label=name)
    
    plt.title("Activation Functions")
    plt.xlabel("Input (x)")
    plt.ylabel("Output")
    plt.axhline(0, color="black", lw=0.5, ls="--")
    plt.axvline(0, color="black", lw=0.5, ls="--")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    interact(
        plot_activation_functions,
        x_range=FloatSlider(value=5, min=0, max=10, step=0.1, description="X Range")
    )
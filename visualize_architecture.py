import torch
import torch.nn as nn
import visualtorch
import matplotlib.pyplot as plt
from src.models import *

model = MicroResNet(num_classes=3)

# Input shape matching the dataset (used to dynamically compute the layer dimensions)
input_shape = (1, 3, 200, 300)
img = visualtorch.layered_view(model, input_shape=input_shape, legend=True)
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()

# Optional
# img.save("model_layered_view.png")
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T

plt.style.use('seaborn-v0_8-whitegrid')

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

denormalize = T.Compose([
    T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
])


def visualize_image(image):
    return to_pil(denormalize(image))


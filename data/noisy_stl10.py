from torchvision.datasets import STL10
from torchvision import transforms
import numpy as np
import torch

def noisify(dataset, train_labels, noise_type, noise_rate, random_state):
    np.random.seed(random_state)
    num_samples = len(train_labels)
    noisy_labels = train_labels.copy()

    if noise_type == 'symmetric':
        num_noisy_samples = int(noise_rate * num_samples)
        noisy_indices = np.random.choice(num_samples, num_noisy_samples, replace=False)
        noisy_labels[noisy_indices] = np.random.randint(0, dataset.num_classes, num_noisy_samples)

    # Add more cases for different noise types as needed

    return noisy_labels, noise_rate

class NoisySTL10(STL10):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        super(NoisySTL10, self).__init(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state

        if noise_type is not None and split == 'train':
            self.add_noise()

    def add_noise(self):
        labels = np.array(self.labels)
        noisy_labels, _ = noisify(dataset='stl10', train_labels=labels, noise_type=self.noise_type, noise_rate=self.noise_rate, random_state=self.random_state)
        self.labels = torch.tensor(noisy_labels)

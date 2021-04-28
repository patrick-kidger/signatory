import matplotlib
import matplotlib.pyplot as plt
import math
import signatory
import torch

matplotlib.rc('text', usetex=True)
matplotlib.rc('font', size=10)


def save(name):
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


time = torch.linspace(0, 1, 10)
path = torch.stack([torch.cos(math.pi * time), torch.sin(math.pi * time)]).T.unsqueeze(0)

depth = 11
signature = signatory.signature(path, depth)

reconstructed_path = signatory.invert_signature(signature, depth, path.shape[2], initial_position=path[:, 0, :])

plt.plot(path[0, :, 0], path[0, :, 1], marker='o', label='original')
plt.plot(reconstructed_path[0, :, 0], reconstructed_path[0, :, 1], marker='o', label='reconstruction')
plt.legend()
save('Half_circle_inversion.png')
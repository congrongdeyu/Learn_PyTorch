import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor(list(range(25)), dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


tudui = Tudui()

writer = SummaryWriter("./logs")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

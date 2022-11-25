import torchvision
from torch.utils.tensorboard import SummaryWriter

# # 下载训练数据集和测试数据集
# train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)

# # 查看测试数据集的元素
# print(test_set[0])
#
# # 查看测试数据集的分类
# print(test_set.classes)
#
# # 查看图片和标签
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# 定义图片变换函数，把图片变换为Tensor变量
dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 下载训练数据集和测试数据集
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transforms, download=True)

# 使用
writer = SummaryWriter("P10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img_tensor=img, global_step=i)
writer.close()

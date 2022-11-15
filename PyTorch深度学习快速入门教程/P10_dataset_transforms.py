import torchvision

# 下载训练数据集和测试数据集
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)

# 查看测试数据集的元素
print(test_set[0])

# 查看测试数据集的分类
print(test_set.classes)

# 查看图片和标签
img, target = test_set[0]
print(img)
print(target)
print(test_set.classes[target])
img.show()
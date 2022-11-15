from PIL import Image
from torchvision import transforms

img = Image.open("../PyTorch深度学习快速入门教程/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg")
print(img)

transfroms_tensor = transforms.ToTensor()
tensor_img = transfroms_tensor(img)
print(tensor_img.shape)

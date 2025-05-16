import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

manual_transforms = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),  # resize_size=256
    transforms.CenterCrop(224),                                       # crop_size=224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print(manual_transforms)

from torchvision import transforms

class ImageTransform:
    def __init__(self, resize, mean, std):
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),

            'val': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
        }

    def __call__(self, image, phase='train'):
        return self.transform[phase](image)

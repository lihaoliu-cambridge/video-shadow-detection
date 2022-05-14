import random
from PIL import Image
from torchvision import transforms


def get_train_joint_transform(scale=416):
    joint_transform = Compose([
        RandomHorizontallyFlip(),
        Resize((scale, scale))
    ])
    return joint_transform


def get_val_joint_transform(scale=416):
    joint_transform = Compose([
        Resize((scale, scale))
    ])
    return joint_transform


def get_img_transform():
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return img_transform


def get_target_transform():
    target_transform = transforms.ToTensor()
    return target_transform


def get_to_pil():
    to_pil = transforms.ToPILImage()
    return to_pil
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, manual_random=None):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask, manual_random)
        return img, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, manual_random=None):
        if manual_random is None:
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask
        else:
            if manual_random < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, manual_random=None):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

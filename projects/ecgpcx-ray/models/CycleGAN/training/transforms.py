from torchvision import transforms


def get_train_transforms(image_size=128):

    return transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=5
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])


def get_val_transforms(image_size=128):

    return transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
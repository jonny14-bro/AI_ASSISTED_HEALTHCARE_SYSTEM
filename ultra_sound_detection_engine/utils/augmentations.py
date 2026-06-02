import albumentations as A

def get_train_augmentations():
    """
    Ultrasound-safe augmentations.
    Applied ONLY during training.
    """
    return A.Compose([
        A.Rotate(
            limit=12,          # smaller rotation = more realistic
            interpolation=1,   # cv2.INTER_LINEAR
            border_mode=0,     # cv2.BORDER_CONSTANT
            p=0.4
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.GaussianBlur(
            blur_limit=3,
            p=0.2
        ),
    ])


def get_val_augmentations():
    """
    NO augmentation for validation/testing.
    """
    return A.Compose([])

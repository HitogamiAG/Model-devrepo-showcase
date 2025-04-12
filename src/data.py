"""Data modules for segmentation task."""

import glob
import os

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as L
from torch.utils.data import DataLoader, Dataset

import settings


def get_png_size(filepath: str) -> tuple:
    """Get the size of a PNG image parsing the header"""
    with open(filepath, "rb") as f:
        f.seek(16)
        width = int.from_bytes(f.read(4), "big")
        height = int.from_bytes(f.read(4), "big")
        return height, width


class SegmentationDataModule(L.LightningDataModule):
    """Custom lightning data module"""

    def __init__(self, data_config: dict) -> None:
        """Initializes the data module with the given configuration.

        Args:
            data_config (dict): Configuration for the data module.
        """
        super().__init__()
        self.save_hyperparameters(data_config)

    def setup(self, stage: str = None) -> None:
        """Sets up the data module for training, validation, and testing.

        Args:
            stage (str, optional): Stage of the data module. Can be "fit", "test", or None. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = CustomTrainDataset(
                data_split_path=self.hparams["train"]["path"],
                transform_config=self.hparams["train"],
            )

            self.val_dataset = CustomValDataset(
                data_split_path=self.hparams["val"]["path"],
                transform_config=self.hparams["val"],
            )

            if self.hparams["dry_run"]:
                # NOTE: For dry run we will use only 16 samples
                self.train_dataset.samples = self.train_dataset.samples[:16]
                self.val_dataset.samples = self.val_dataset.samples[:16]

        if stage == "test" or stage is None:
            # NOTE: As there are no labels for test split,
            #      we will use the same dataset as for validation
            self.test_dataset = CustomValDataset(
                data_split_path=self.hparams["val"]["path"],
                transform_config=self.hparams["val"],
            )

            if self.hparams["dry_run"]:
                # NOTE: For dry run we will use only 16 samples
                self.test_dataset.samples = self.test_dataset.samples[:16]

    def train_dataloader(self) -> DataLoader:
        """Creates the training data loader.

        Returns:
            DataLoader: Data loader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams["train"]["batch_size"],
            shuffle=self.hparams["train"]["shuffle"],
            num_workers=self.hparams["train"]["num_workers"],
            pin_memory=self.hparams["train"]["pin_memory"],
            persistent_workers=(
                True if self.hparams["train"]["num_workers"] > 0 else False
            ),
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the validation data loader.

        Returns:
            DataLoader: Data loader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams["val"]["batch_size"],
            shuffle=self.hparams["val"]["shuffle"],
            num_workers=self.hparams["val"]["num_workers"],
            pin_memory=self.hparams["val"]["pin_memory"],
            persistent_workers=(
                True if self.hparams["val"]["num_workers"] > 0 else False
            ),
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the testing data loader.

        Returns:
            DataLoader: Data loader for the validation* dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams["val"]["batch_size"],
            shuffle=self.hparams["val"]["shuffle"],
            num_workers=self.hparams["val"]["num_workers"],
            pin_memory=self.hparams["val"]["pin_memory"],
            persistent_workers=(
                True if self.hparams["val"]["num_workers"] > 0 else False
            ),
        )


class CustomTrainDataset(Dataset):
    """Custom dataset for training"""

    def __init__(self, data_split_path: str, transform_config: dict):
        """Initializes the dataset with the given data split path and transform configuration.

        Args:
            data_split_path (str): Path to the data split folder.
            transform_config (dict): Configuration for data augmentation transforms.
        """
        super().__init__()

        self.parse_split_folder(data_split_path)
        self.parse_transform_config(transform_config)

        # NOTE: Dummy cache mechanism that speeds up training x100 times
        self.cache = {}

    def parse_split_folder(self, data_split_path: str) -> None:
        """Parses the data split folder and initializes the dataset samples.

        Args:
            data_split_path (str): Path to the data split folder.
        """

        self.samples = []

        # NOTE: Exclude 400 augmented samples "shifted*.png" and "flipped*.png"
        images = glob.glob(
            os.path.join(data_split_path, "**", "Images", "000*.png")
        ) + glob.glob(os.path.join(data_split_path, "**", "Images", "file*.png"))

        for image_path in images:
            label_path = image_path.replace("/Images/", "/Labels/")

            self.samples.append((image_path, label_path))

    def parse_transform_config(self, transform_config: dict) -> None:
        """Parses the transform configuration and initializes the transformation pipeline.

        Args:
            transform_config (dict): Configuration for data augmentation transforms.
        """

        self.transform = None

        if transform_config:
            self.transform = A.from_dict(transform_config)

    def __len__(self) -> int:
        """Gets the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.samples)

    def load_image(self, image_path: str) -> np.ndarray:
        """Loads an image from the given path.

        Args:
            image_path (str): Path to the image.

        Returns:
            np.ndarray: Loaded image.
        """
        return cv2.imread(image_path)

    def load_label(self, label_path: str) -> np.ndarray:
        """Loads a label from the given path.
        The label is encoded to match the class encoding defined in settings.

        Args:
            label_path (str): Path to the label.

        Returns:
            np.ndarray: Encoded label.
        """
        label = cv2.imread(label_path)
        height, width = label.shape[:2]

        label = label.reshape(-1, 3)

        encoded_label = np.zeros((height, width), dtype=np.uint8)
        for class_idx, (class_name, pixel_value) in enumerate(
            settings.CLASS_ENCODING.items()
        ):
            encoded_class = np.all(label == pixel_value, axis=1, keepdims=True)
            encoded_class = encoded_class.reshape(height, width)

            encoded_label[encoded_class] = class_idx

        return encoded_label

    def __getitem__(self, idx: int) -> tuple:
        """Gets an item from the dataset.

        Args:
            idx (int): Index of the item to get.

        Returns:
            tuple: (image, label)
        """

        image_path, label_path = self.samples[idx]
        if image_path in self.cache:
            image, label = self.cache[image_path]
        else:
            image = self.load_image(image_path)
            label = self.load_label(label_path)
            self.cache[image_path] = (image, label)
        # image = self.load_image(image_path)
        # label = self.load_label(label_path)

        augmented = self.transform(image=image, mask=label)
        sample = augmented["image"]
        label = augmented["mask"]

        return sample, label


class CustomValDataset(Dataset):
    """Custom dataset for validation and testing"""

    def __init__(
        self,
        data_split_path: str,
        transform_config: dict,
        image_crop_size: int = settings.IMAGE_CROP_SIZE,
    ):
        """
        While the CustomTrainDataset returns random crop of N size of image,
        the CustomTestDataset is filled with sequences of sliding windows ("slices") of size N.
        So the CustomTestDataset reproduces "Slicing" inference strategy.
        """
        super().__init__()

        self.parse_split_folder(data_split_path, image_crop_size)
        self.parse_transform_config(transform_config)

        # NOTE: Dummy cache mechanism that speeds up testing x100 times
        self.cache = {}

    def generate_slice_intervals(
        self, image_height: int, image_width: int, image_crop_size: int
    ) -> list:
        """Generates the intervals for slicing the image.

        Args:
            image_height (int): Height of the image.
            image_width (int): Width of the image.
            image_crop_size (int): Size of the crop.

        Returns:
            list: List of tuples representing the intervals.
        """
        intervals = []

        # NOTE: It's ok if the slice end exceeds the image size, we will use paddning
        for i in range(0, image_height, image_crop_size):
            for j in range(0, image_width, image_crop_size):
                intervals.append(
                    (slice(i, i + image_crop_size), slice(j, j + image_crop_size))
                )

        return intervals

    def parse_split_folder(self, data_split_path: str, image_crop_size: int) -> None:
        """Parses the data split folder and initializes the dataset samples.

        Args:
            data_split_path (str): Path to the data split folder.
            image_crop_size (int): Size of the crop.
        """

        self.samples = []

        for image_path in glob.glob(
            os.path.join(data_split_path, "**", "Images", "*.png")
        ):
            label_path = image_path.replace("/Images/", "/Labels/")

            height, width = get_png_size(image_path)
            intervals = self.generate_slice_intervals(height, width, image_crop_size)

            for interval in intervals:
                self.samples.append((image_path, label_path, interval))

    def parse_transform_config(self, transform_config: dict) -> None:
        """Parses the transform configuration and initializes the transformation pipeline.

        Args:
            transform_config (dict): Configuration for data augmentation transforms.
        """

        self.transform = None

        if transform_config:
            self.transform = A.from_dict(transform_config)

    def __len__(self) -> int:
        """Gets the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.samples)

    def load_image(self, image_path: str) -> np.ndarray:
        """Loads an image from the given path.

        Args:
            image_path (str): Path to the image.

        Returns:
            np.ndarray: Loaded image.
        """
        return cv2.imread(image_path)

    def load_label(self, label_path: str) -> np.ndarray:
        """Loads a label from the given path.
        The label is encoded to match the class encoding defined in settings.

        Args:
            label_path (str): Path to the label.

        Returns:
            np.ndarray: Encoded label.
        """
        label = cv2.imread(label_path)
        height, width = label.shape[:2]

        label = label.reshape(-1, 3)

        encoded_label = np.zeros((height, width), dtype=np.uint8)
        for class_idx, (class_name, pixel_value) in enumerate(
            settings.CLASS_ENCODING.items()
        ):
            encoded_class = np.all(label == pixel_value, axis=1, keepdims=True)
            encoded_class = encoded_class.reshape(height, width)

            encoded_label[encoded_class] = class_idx

        return encoded_label

    def pad_if_needed(self, array: np.ndarray, target_size: int) -> np.ndarray:
        """Pads the array to the target size if needed.

        Args:
            array (np.ndarray): Array to pad.
            target_size (int): Target size.

        Returns:
            np.ndarray: Padded array.
        """
        height, width = array.shape[:2]

        if height < target_size or width < target_size:
            pad_height = max(0, target_size - height)
            pad_width = max(0, target_size - width)

            array = cv2.copyMakeBorder(
                array, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        return array

    def __getitem__(self, idx: int) -> tuple:
        """Gets an item from the dataset.

        Args:
            idx (int): Index of the item to get.

        Returns:
            tuple: (image, label)
        """

        image_path, label_path, interval = self.samples[idx]

        if image_path in self.cache:
            image, label = self.cache[image_path]
        else:
            image = self.load_image(image_path)
            label = self.load_label(label_path)
            self.cache[image_path] = (image, label)

        image = image[interval]
        label = label[interval]

        image = self.pad_if_needed(image, settings.IMAGE_CROP_SIZE)
        label = self.pad_if_needed(label, settings.IMAGE_CROP_SIZE)

        augmented = self.transform(image=image, mask=label)
        sample = augmented["image"]
        label = augmented["mask"]

        return sample, label


if __name__ == "__main__":
    import yaml
    from tqdm import tqdm

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_split_config = config["data"]

    data_module = SegmentationDataModule(data_split_config)
    data_module.setup(stage="fit")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    for batch in tqdm(train_loader):
        images, labels = batch
        print(images.shape, labels.shape)
        break

    for batch in tqdm(val_loader):
        images, labels = batch
        print(images.shape, labels.shape)
        break

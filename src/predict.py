"""Prediction script for image segmentation tasks"""

import datetime as dt
import os
from urllib.parse import urlparse
from warnings import filterwarnings

import cv2
import numpy as np
import onnxruntime as ort
import requests
import segmentation_models_pytorch as smp
import torch

import settings
from data import PredictionSource
from utils import create_dir_safely, get_console_logger, parse_predict_args


class PredictionEngine:
    """Prediction engine for image segmentation tasks."""

    DEFAULT_MEAN = np.array([0.485, 0.456, 0.406])  # ImageNet mean
    DEFAULT_STD = np.array([0.229, 0.224, 0.225])  # ImageNet std

    def __init__(
        self,
        model_source: str,
        num_classes: int,
        batch_size: int,
        image_height: int,
        image_width: int,
        apply_slicing: bool,
        slice_height: int,
        slice_width: int,
        slice_overlap: float,
        half: bool = False,
    ):
        """Initialize the PredictionEngine.

        Args:
            model_source (str): Path to the model file or URL.
            num_classes (int): Number of classes for segmentation.
            batch_size (int): Batch size for inference.
            image_crop_size (int): Size of the image crop.
            intersection_ratio (float): Ratio of intersection for cropping.
            half (bool, optional): Only for Pytorch Inference! Use FP16. Defaults to False.
        """

        self.model_source = model_source
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.image_crop_size = slice_height
        self.apply_slicing = apply_slicing
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.slice_overlap = slice_overlap
        self.half = half

        self.logger = get_console_logger("PredictionEngine")

        self.load_model()

    def download_model(self, url: str) -> str:
        """Download the model from a URL and save it to a temporary file.

        Args:
            url (str): URL of the model file.

        Returns:
            str: Path to the downloaded model file.
        """
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        # Create full path in /tmp directory
        filepath = os.path.join("/tmp", filename)

        # Download and save the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)

        return filepath

    def load_model(self) -> None:
        """Load the model from the specified source."""
        if self.model_source.startswith("http"):
            model_path = self.download_model(self.model_source)
            self.model_source = model_path

        if self.model_source.endswith(".ckpt"):
            self._engine = "torch"

            checkpoint = torch.load(self.model_source)
            # Remove 'model.' prefix from state_dict keys
            checkpoint["state_dict"] = {
                k[6:]: v for k, v in checkpoint["state_dict"].items()
            }

            self.model = smp.create_model(**checkpoint["hyper_parameters"]["model"])
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()

            if self.half:
                self.model.half()

        elif self.model_source.endswith(".onnx"):
            self._engine = "ort"

            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.logger.warning(
                    "CUDAExecutionProvider not available. Using CPUExecutionProvider."
                )

                if "LD_LIBRARY_PATH" not in os.environ:
                    self.logger.warning(
                        "LD_LIBRARY_PATH not set. This may cause issues with CUDAExecutionProvider."
                    )

                self._providers = ["CPUExecutionProvider"]

            self.model = ort.InferenceSession(
                self.model_source, providers=self._providers
            )
            self._input_name = self.model.get_inputs()[0].name
            self._output_name = self.model.get_outputs()[0].name

            self._dtype = (
                self.model.get_inputs()[0].type.replace("tensor(", "").replace(")", "")
            )
            try:
                self._dtype = np.dtype(self._dtype)
            except TypeError:
                self.logger.warning(
                    f"ORT expected input type not understood: {self._dtype}. Defaulting to float32."
                )
                self._dtype = np.float32

            if self.half:
                pass
        else:
            raise ValueError(f"Unsupported model format: {self.model_source}")

    def generate_slice_intervals(
        self,
        image_height: int,
        image_width: int,
        slice_height: int,
        slice_width: int,
        slice_overlap: float,
    ) -> list:
        """Generates the intervals for slicing the image.

        Args:
            image_height (int): Height of the image.
            image_width (int): Width of the image.
            slice_height (int): Height of the slice.
            slice_width (int): Width of the slice.
            slice_overlap (float): Overlap ratio for the slices.

        Returns:
            list: List of tuples representing the intervals.
        """
        intervals = []

        height_step_size = int(slice_height * (1 - slice_overlap))
        width_step_size = int(slice_width * (1 - slice_overlap))

        for i in range(0, image_height, height_step_size):
            for j in range(0, image_width, width_step_size):
                height_slice_start = i - max(0, (i + slice_height) - image_height)
                height_slice_end = i + slice_height

                width_slice_start = j - max(0, (j + slice_width) - image_width)
                width_slice_end = j + slice_width

                intervals.append(
                    (
                        slice(height_slice_start, height_slice_end),
                        slice(width_slice_start, width_slice_end),
                    )
                )

        return intervals

    def split_to_slices(self, image: np.ndarray) -> tuple:
        """Apply slicing to the image based on the specified intervals.

        Args:
            image (np.ndarray): Input image.

        Returns:
            tuple: Tuple containing the sliced images and the intervals.
        """

        intervals = self.generate_slice_intervals(
            self.image_height,
            self.image_width,
            self.slice_height,
            self.slice_width,
            self.slice_overlap,
        )

        slices = []
        for interval in intervals:
            slices.append(image[interval])

        return np.array(slices), intervals

    def concatenate_slices(
        self, slices_probs: np.ndarray, intervals: list, image_shape: tuple
    ) -> np.ndarray:
        """Concatenate the slices back to the original image shape.

        Args:
            slices_probs (np.ndarray): Segmentation masks of probs of classes.
            intervals (list): List of intervals used for slicing.
            image_shape (tuple): Original image shape.

        Returns:
            np.ndarray: Concatenated segmentation mask.
        """

        slices_probs = np.moveaxis(slices_probs, (1, 2), (-1, -2))
        mask_probs = np.zeros(
            (image_shape[0], image_shape[1], self.num_classes), dtype=np.uint8
        )

        for slice_probs, slice_interval in zip(slices_probs, intervals):
            mask_probs[slice_interval] = slice_probs

        mask = np.argmax(mask_probs, axis=-1)
        return mask

    def predict_torch(self, images: np.ndarray) -> np.ndarray:
        """Perform inference using PyTorch.

        Args:
            images (np.ndarray): Input images for prediction.

        Returns:
            np.ndarray: Predicted segmentation masks.
        """

        images_tensor = torch.from_numpy(images).float()

        if self.half:
            images_tensor = images_tensor.half()

        masks_probs = np.zeros(
            (
                images_tensor.shape[0],
                self.num_classes,
                images_tensor.shape[2],
                images_tensor.shape[3],
            ),
            dtype=np.float32,
        )
        for i in range(0, images_tensor.shape[0], self.batch_size):
            with torch.no_grad():
                masks_probs[i : i + self.batch_size] = (
                    self.model(images_tensor[i : i + self.batch_size]).cpu().numpy()
                )

        return masks_probs

    def predict_ort(self, images: np.ndarray) -> np.ndarray:
        """Perform inference using ONNX Runtime.

        Args:
            images (np.ndarray): Input images for prediction.

        Returns:
            np.ndarray: Predicted segmentation masks.
        """

        images = images.astype(self._dtype)

        masks_probs = np.zeros(
            (images.shape[0], self.num_classes, images.shape[2], images.shape[3]),
            dtype=self._dtype,
        )
        for i in range(0, images.shape[0], self.batch_size):
            masks_probs[i : i + self.batch_size] = self.model.run(
                [self._output_name], {self._input_name: images[i : i + self.batch_size]}
            )[0]

        return masks_probs

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict the segmentation mask for the given image.

        Args:
            image (np.ndarray): Input image for prediction.

        Returns:
            np.ndarray: Predicted segmentation mask.
        """

        height, width = image.shape[:2]
        image = cv2.resize(
            image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC
        )

        if self.apply_slicing:
            image_tensor, intervals = self.split_to_slices(image)
        else:
            image_tensor = np.expand_dims(image, axis=0)
            intervals = [(slice(None), slice(None))]

        image_tensor = np.moveaxis(image_tensor, (1, 2), (-1, -2))

        # Normalize the images
        image_tensor = image_tensor.astype(np.float32)
        image_tensor = image_tensor / 255.0
        image_tensor = (
            image_tensor - self.DEFAULT_MEAN.reshape(1, 3, 1, 1)
        ) / self.DEFAULT_STD.reshape(1, 3, 1, 1)

        if self._engine == "ort":
            masks_probs = self.predict_ort(image_tensor)
        elif self._engine == "torch":
            masks_probs = self.predict_torch(image_tensor)
        else:
            raise ValueError(f"Unsupported engine: {self._engine}")

        mask = self.concatenate_slices(masks_probs, intervals, image.shape[:2])

        if np.max(mask) > 255:
            self.logger.warning(
                "Number of classes exceeds 255. Converting to uint8 could lead to wrong results."
            )

        mask = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_CUBIC
        )

        return mask


if __name__ == "__main__":

    filterwarnings("ignore")
    console_logger = get_console_logger("PredictLogger")

    # --- Parse command line arguments ---
    args = parse_predict_args()

    # --- Set up run name ---
    run_name = f'{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    console_logger.info(f"Generated run name: {run_name}")

    # --- Track experiment into local folder ---
    dirpath = f"{settings.PREDICT_LOG_DIR}/{run_name}"
    create_dir_safely(dirpath)

    # --- Load source ---
    source_generator = PredictionSource(source=args.source)

    # --- Load model ---
    model = PredictionEngine(
        model_source=args.model,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        half=args.half,
        image_height=args.image_height,
        image_width=args.image_width,
        apply_slicing=args.apply_slicing,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        slice_overlap=args.slice_overlap,
    )

    # --- Iterate over source and predict ---
    for i, (image, image_filename) in enumerate(source_generator):
        console_logger.info(f"Predicting image {i + 1}/{len(source_generator)}")

        # Predict the segmentation mask
        mask = model.predict(image)

        # Save the predicted mask
        mask_path = os.path.join(dirpath, image_filename)
        cv2.imwrite(mask_path, (mask / args.num_classes * 255).astype(np.uint8))

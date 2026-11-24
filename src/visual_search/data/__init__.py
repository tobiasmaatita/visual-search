"""Dataset schema, loaders, validation and the query/gallery split."""

from visual_search.data.folder import load_folder_dataset, load_image_folder, load_mlcvtools_annotations
from visual_search.data.schema import Item, Prompt, is_match, parse_dataset
from visual_search.data.split import (
    DatasetValidationError,
    Split,
    ValidationReport,
    split_dataset,
    validate_dataset,
)
from visual_search.data.synthetic import make_synthetic_dataset

__all__ = [
    "DatasetValidationError",
    "Item",
    "Prompt",
    "Split",
    "ValidationReport",
    "is_match",
    "load_folder_dataset",
    "load_image_folder",
    "load_mlcvtools_annotations",
    "make_synthetic_dataset",
    "parse_dataset",
    "split_dataset",
    "validate_dataset",
]

import os
import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["benign", "malignant"]


class ThyroidDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, image_path


def read_xml_label(xml_path):

    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"Annotation file not found:\n{xml_path}"
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    object_node = root.find("object")

    if object_node is None:
        raise ValueError(
            f"No <object> found in annotation:\n{xml_path}"
        )

    name_node = object_node.find("name")

    if name_node is None:
        raise ValueError(
            f"No <name> found in annotation:\n{xml_path}"
        )

    label_text = name_node.text.strip()

    if label_text == "0":
        return 0

    elif label_text == "1":
        return 1

    else:
        raise ValueError(
            f"Unknown label '{label_text}' in:\n{xml_path}\n"
            f"Expected 0 (benign) or 1 (malignant)."
        )


def load_tn5000_split(root_dir, split):

    image_dir = os.path.join(root_dir, "JPEGImages")

    annotation_dir = os.path.join(root_dir, "Annotations")

    split_file = os.path.join(
        root_dir,
        "ImageSets",
        "Main",
        f"{split}.txt"
    )

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"JPEGImages folder not found:\n{image_dir}"
        )

    if not os.path.isdir(annotation_dir):
        raise FileNotFoundError(
            f"Annotations folder not found:\n{annotation_dir}"
        )

    if not os.path.isfile(split_file):
        raise FileNotFoundError(
            f"Split file not found:\n{split_file}"
        )

    samples = []

    with open(split_file, "r") as f:

        image_ids = [
            line.strip()
            for line in f
            if line.strip()
        ]

    for image_id in image_ids:

        image_id = image_id.split()[0]

        image_path = os.path.join(
            image_dir,
            image_id + ".jpg"
        )

        xml_path = os.path.join(
            annotation_dir,
            image_id + ".xml"
        )

        if not os.path.exists(image_path):
            print(
                f"Warning: image missing -> {image_path}"
            )
            continue

        if not os.path.exists(xml_path):
            print(
                f"Warning: annotation missing -> {xml_path}"
            )
            continue

        label = read_xml_label(xml_path)

        samples.append(
            (image_path, label)
        )

    if len(samples) == 0:
        raise RuntimeError(
            f"No valid samples found for split '{split}'."
        )

    return samples


def load_all_tn5000_samples(root_dir):

    image_dir = os.path.join(root_dir, "JPEGImages")
    annotation_dir = os.path.join(root_dir, "Annotations")

    samples = []

    for filename in os.listdir(image_dir):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_id = os.path.splitext(filename)[0]

        image_path = os.path.join(
            image_dir,
            filename
        )

        xml_path = os.path.join(
            annotation_dir,
            image_id + ".xml"
        )

        if not os.path.exists(xml_path):
            continue

        label = read_xml_label(xml_path)

        samples.append(
            (image_path, label)
        )

    if len(samples) == 0:
        raise RuntimeError(
            "No TN5000 samples found."
        )

    return samples
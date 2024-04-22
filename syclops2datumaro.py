import os
import json
import numpy as np
import argparse
import datumaro as dm
import datumaro.plugins.splitter as splitter
from datumaro.components.annotation import LabelCategories, AnnotationType
from collections import OrderedDict
from glob import glob
import cv2
from itertools import product
import yaml

LABEL_MAP = OrderedDict([("ground", 0), ("maize", 1), ("weed", 2)])
CLASSES_TO_SKIP = [0]
INSTANCE_PIXEL_THRESHOLD = 0.01
INSTANCE_COUNT_THRESHOLD = 5
NUM_KEYPOINTS = 1


def create_syclops_categories(label_map):
    categories = {}
    label_categories = LabelCategories()
    for label in label_map:
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories
    return categories


def load_image_files(images_dir):
    image_files = sorted(glob(os.path.join(images_dir, "*.png")))
    return [os.path.basename(item) for item in image_files]


def load_keypoints(keypoints_path):
    if os.path.exists(keypoints_path):
        with open(keypoints_path) as f:
            return json.load(f)
    return None


def filter_instances(sem_mask, inst_mask, classes_to_skip):
    mask = np.zeros(sem_mask.shape)
    for class_id in classes_to_skip:
        mask += sem_mask == class_id
    unique_instance_ids = np.unique(inst_mask[mask == 0])
    return unique_instance_ids


def process_instance(sem_mask, inst_mask, instance_id, item, keypoints_data):
    num_pixels = np.sum(inst_mask == instance_id)
    class_ids, count = np.unique(sem_mask[inst_mask == instance_id], return_counts=True)
    class_ids = class_ids[count > num_pixels * INSTANCE_PIXEL_THRESHOLD]
    count = count[count > num_pixels * INSTANCE_PIXEL_THRESHOLD]
    class_ids = class_ids[count > INSTANCE_COUNT_THRESHOLD]
    count = count[count > INSTANCE_COUNT_THRESHOLD]
    if len(class_ids) == 0:
        return

    class_id = np.min(class_ids)
    instance_idx = np.zeros(inst_mask.shape, dtype=bool)
    instance_idx[(inst_mask == instance_id) & np.isin(sem_mask, class_ids)] = True
    inst_mask_datumaro = dm.Mask(
        instance_idx, label=class_id, group=instance_id, object_id=instance_id
    )
    item.annotations.append(inst_mask_datumaro)

    if keypoints_data and str(instance_id) in keypoints_data:
        for kp_id, kp in keypoints_data[str(instance_id)].items():
            if kp_id != "class_id":
                item.annotations.append(
                    dm.Points(
                        [float(kp["x"]), float(kp["y"])],
                        label=int(kp_id),
                        group=instance_id,
                        object_id=instance_id,
                    )
                )

    bbox = inst_mask_datumaro.get_bbox()
    item.annotations.append(
        dm.Bbox(*bbox, label=class_id, group=instance_id, object_id=instance_id)
    )


def find_closest_points(contour1, contour2):
    # Convert contours to NumPy arrays if they aren't already
    contour1 = np.asarray(contour1)
    contour2 = np.asarray(contour2)

    # Compute the difference matrix between each pair of points
    diff = contour1[:, np.newaxis, :] - contour2[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Find the indices of the minimum distance in the matrix
    i, j = np.unravel_index(np.argmin(dist_matrix, axis=None), dist_matrix.shape)
    return i, j


def merge_contours(contour1, contour2, closest_points):
    i, j = closest_points
    new_contour = np.concatenate(
        (contour1[: i + 1], contour2[j:], contour2[: j + 1], contour1[i:])
    )
    return new_contour


def connect_contours(contours, contour_approx_eps=2.0):
    contours = [
        cv2.approxPolyDP(c, epsilon=contour_approx_eps, closed=True).reshape(-1, 2)
        for c in contours
    ]
    while len(contours) > 1:
        # Find pair of contours with the closest points
        closest_pair = None
        min_dist = float("inf")
        for (i, c1), (j, c2) in product(enumerate(contours), repeat=2):
            if i == j:
                continue
            cp = find_closest_points(c1, c2)
            dist = np.linalg.norm(c1[cp[0]] - c2[cp[1]])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (i, j, cp)

        # Merge the closest contours
        i, j, cp = closest_pair
        contours[i] = merge_contours(contours[i], contours[j], cp)
        del contours[j]

    return contours[0]


def convert_to_datumaro(
    images_dir,
    sem_masks_dir,
    inst_masks_dir,
    keypoints_dir,
    output_dir,
    classes_to_skip,
    flags,
):
    dataset = dm.Dataset(
        media_type=dm.components.media.Image,
        categories=create_syclops_categories(LABEL_MAP),
    )

    image_files = load_image_files(images_dir)
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(images_dir, image_name + ".png")
        sem_mask_path = os.path.join(sem_masks_dir, image_name + ".npz")
        inst_mask_path = os.path.join(inst_masks_dir, image_name + ".npz")
        keypoints_path = os.path.join(keypoints_dir, image_name + ".json")

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        sem_mask = np.load(sem_mask_path)["array"]
        inst_mask = np.load(inst_mask_path)["array"]
        keypoints_data = load_keypoints(keypoints_path)

        item = dm.DatasetItem(id=image_name, media=dm.Image.from_file(image_path))
        unique_instance_ids = filter_instances(sem_mask, inst_mask, classes_to_skip)

        for instance_id in unique_instance_ids:
            process_instance(sem_mask, inst_mask, instance_id, item, keypoints_data)

        dataset.put(item)

    if "yolo_ultralytics_seg" in flags.output_formats:
        yolo_seg_output_dir = os.path.join(output_dir, "yolo_ultralytics_seg")
        os.makedirs(yolo_seg_output_dir, exist_ok=True)

        # Split data into train and val sets
        splits = [("train", flags.train_ratio), ("val", 1 - flags.train_ratio)]
        split_dataset = dataset.transform("split", task="segmentation", splits=splits)

        for split_name, split_subset in split_dataset.subsets().items():
            split_dir_image = os.path.join(yolo_seg_output_dir, "images", split_name)
            split_dir_label = os.path.join(yolo_seg_output_dir, "labels", split_name)
            os.makedirs(split_dir_image, exist_ok=True)
            os.makedirs(split_dir_label, exist_ok=True)

            for item in split_subset:
                image_name = item.id
                image_path = os.path.join(split_dir_image, image_name + ".jpg")
                label_path = os.path.join(split_dir_label, image_name + ".txt")

                # Save image
                item.media.save(image_path)

                # Save labels
                with open(label_path, "w") as f:
                    for ann in item.annotations:
                        if ann.type == AnnotationType.mask:
                            class_id = ann.label
                            contours, _ = cv2.findContours(
                                ann.image.astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE,
                            )
                            connected_contour = connect_contours(contours)
                            if (
                                len(connected_contour) >= 6
                            ):  # Ensure at least 3 points (6 coordinates)
                                # Normalize coordinates
                                connected_contour = [
                                    coord / width if i % 2 == 0 else coord / height
                                    for i, coord in enumerate(
                                        connected_contour.flatten().tolist()
                                    )
                                ]
                                f.write(
                                    f"{class_id} {' '.join(map(str, connected_contour))}\n"
                                )

        # Save dataset YAML
        dataset_yaml = {
            "path": os.path.abspath(yolo_seg_output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(LABEL_MAP.keys())},
        }
        with open(os.path.join(yolo_seg_output_dir, "dataset.yaml"), "w") as f:
            yaml.dump(dataset_yaml, f)

    if "datumaro" in flags.output_formats:
        dataset.export(os.path.join(output_dir, "datumaro"), format="datumaro")

    if "coco" in flags.output_formats:
        splits = [("train", flags.train_ratio), ("val", 1 - flags.train_ratio)]
        task = splitter.SplitTask.segmentation.name
        new_dataset = dataset.transform("split", task=task, splits=splits, seed=42)
        new_dataset.export(os.path.join(output_dir, "coco"), format="coco")

    if "yolo_ultralytics_det" in flags.output_formats:
        splits = [("train", flags.train_ratio), ("val", 1 - flags.train_ratio)]
        task = splitter.SplitTask.detection.name
        new_dataset = dataset.transform("split", task=task, splits=splits, seed=42)
        new_dataset.export(
            os.path.join(output_dir, "yolo_ultralytics_det"),
            format="yolo_ultralytics",
            save_media=True,
        )

    if "yolo_ultralytics_pose" in flags.output_formats:
        yolo_pose_output_dir = os.path.join(output_dir, "yolo_ultralytics_pose")
        os.makedirs(yolo_pose_output_dir, exist_ok=True)

        # Split data into train and val sets
        splits = [("train", flags.train_ratio), ("val", 1 - flags.train_ratio)]
        split_dataset = dataset.transform("split", task="detection", splits=splits)

        for split_name, split_subset in split_dataset.subsets().items():
            split_dir_image = os.path.join(yolo_pose_output_dir, "images", split_name)
            split_dir_label = os.path.join(yolo_pose_output_dir, "labels", split_name)
            os.makedirs(split_dir_image, exist_ok=True)
            os.makedirs(split_dir_label, exist_ok=True)

            for item in split_subset:
                image_name = item.id
                image_path = os.path.join(split_dir_image, image_name + ".jpg")
                label_path = os.path.join(split_dir_label, image_name + ".txt")

                # Save image
                item.media.save(image_path)

                # Save labels
                with open(label_path, "w") as f:
                    for ann in item.annotations:
                        if ann.type == AnnotationType.bbox:
                            class_id = ann.label
                            bbox = ann.get_bbox()
                            x_center = (bbox[0] + bbox[2] / 2) / width
                            y_center = (bbox[1] + bbox[3] / 2) / height
                            bbox_width = bbox[2] / width
                            bbox_height = bbox[3] / height

                            keypoints = []
                            visibilities = []
                            for kp_id in range(NUM_KEYPOINTS):
                                kp_found = False
                                for kp_ann in item.annotations:
                                    if (
                                        kp_ann.type == AnnotationType.points
                                        and kp_ann.group == ann.group
                                        and kp_ann.label == kp_id
                                    ):
                                        kp_x = kp_ann.points[0] / width
                                        kp_y = kp_ann.points[1] / height
                                        keypoints.extend([kp_x, kp_y])
                                        visibilities.append(1)
                                        kp_found = True
                                        break
                                if not kp_found:
                                    keypoints.extend([0, 0])
                                    visibilities.append(0)

                            f.write(
                                f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height} "
                                f"{' '.join(map(str, keypoints))} {' '.join(map(str, visibilities))}\n"
                            )

        # Save dataset YAML
        dataset_yaml = {
            "path": os.path.abspath(yolo_pose_output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(LABEL_MAP.keys())},
            "kpt_shape": [NUM_KEYPOINTS, 3],
        }
        with open(os.path.join(yolo_pose_output_dir, "dataset.yaml"), "w") as f:
            yaml.dump(dataset_yaml, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="C:/Users/Anton/Documents/Datasets/Phenobench Keypoints",
        help="Root directory of the generated dataset",
    )
    parser.add_argument(
        "--output_formats",
        default="yolo_ultralytics_pose",
        nargs="+",
        choices=[
            "datumaro",
            "coco",
            "yolo_ultralytics_det",
            "yolo_ultralytics_seg",
            "yolo_ultralytics_pose",
        ],
        help="Currently supported output labelling formats (default: %(default)s)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of images to use for training (default: %(default)s)",
    )

    flags = parser.parse_args()

    convert_to_datumaro(
        os.path.join(flags.base_dir, "main_camera/rect"),
        os.path.join(flags.base_dir, "main_camera_annotations/semantic_segmentation"),
        os.path.join(flags.base_dir, "main_camera_annotations/instance_segmentation"),
        os.path.join(flags.base_dir, "main_camera_annotations/keypoints"),
        os.path.join(flags.base_dir, "datumaro_dataset"),
        CLASSES_TO_SKIP,
        flags=flags,
    )

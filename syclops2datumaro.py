import os
import json
import numpy as np
import argparse
import datumaro as dm
import datumaro.plugins.splitter as splitter
from datumaro.components.annotation import (
    AnnotationType,
    LabelCategories,
    MaskCategories,
)
from collections import OrderedDict
from glob import glob
from JSON2YOLO.general_json2yolo import convert_coco_json
label_map = OrderedDict([("none", 0), ("ground", 1), ("maize", 2), ("weed", 3)])
classes_to_skip = [0]


def make_syclops_categories(label_map):
    categories = {}
    label_categories = LabelCategories()
    for label in label_map:
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories
    return categories


def convert_to_datumaro(
    images_dir,
    sem_masks_dir,
    inst_masks_dir,
    keypoints_dir,
    output_dir,
    classes_to_skip,
    flags='datumaro',
):
    dataset = dm.Dataset(
        media_type=dm.components.media.Image,
        categories=make_syclops_categories(label_map),
    )

    # Assuming all directories have corresponding file names
    image_files = sorted(glob(os.path.join(images_dir, '*.png')))
    image_files = [os.path.basename(item) for item in image_files]
    for image_file in image_files:
        # if "20" in image_file:
        #     break
        image_name = os.path.splitext(image_file)[0]

        image_path = os.path.join(images_dir, image_name + ".png")
        sem_mask_path = os.path.join(sem_masks_dir, image_name + ".npz")
        inst_mask_path = os.path.join(inst_masks_dir, image_name + ".npz")
        keypoints_path = os.path.join(keypoints_dir, image_name + ".json")

        # Load RGB image
        image = dm.Image.from_file(image_path)

        # Load semantic segmentation mask
        sem_mask = np.load(sem_mask_path)["array"]

        # Load instance segmentation mask
        inst_mask = np.load(inst_mask_path)["array"]

        # Create mask of segmentation image with classes to skip
        mask = np.zeros(sem_mask.shape)
        for class_id in classes_to_skip:
            mask += sem_mask == class_id

        # Get unique instance ids from masked instance image
        unique_instance_ids = np.unique(inst_mask[mask == 0])

        # Create dataset item
        item = dm.DatasetItem(id=image_name, media=image)

        # Load and filter keypoints
        if os.path.exists(keypoints_path):
            with open(keypoints_path) as f:
                keypoints_data = json.load(f)

        for instance_id in unique_instance_ids:
            num_pixels = np.sum(inst_mask == instance_id)
            class_ids, count = np.unique(
                sem_mask[inst_mask == instance_id], return_counts=True
            )
            # Remove class_ids with count less than 1% of total pixels
            class_ids = class_ids[count > num_pixels * 0.01]
            count = count[count > num_pixels * 0.01]
            # Remove class_ids with count less than 5 pixels
            class_ids = class_ids[count > 5]
            count = count[count > 5]
            if len(class_ids) == 0:
                continue

            class_id = np.min(class_ids)
            # Get mask of pixels with instance id and one of the class ids
            instance_idx = np.zeros(inst_mask.shape, dtype=bool)
            instance_idx[(inst_mask == instance_id) & np.isin(sem_mask, class_ids)] = (
                True
            )
            # Create instance mask
            inst_mask_datumaro = dm.Mask(
                instance_idx, label=class_id, group=instance_id, object_id=instance_id
            )
            item.annotations.append(inst_mask_datumaro)
            if os.path.exists(keypoints_path):
                if str(instance_id) in keypoints_data.keys():
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

            # Add bounding box
            bbox = inst_mask_datumaro.get_bbox()
            item.annotations.append(
                dm.Bbox(*bbox, label=class_id, group=instance_id, object_id=instance_id)
            )

            # Add item to dataset
            dataset.put(item)

    
    if 'datumaro' in flags.output_formats:
        # Export dataset in Datumaro format
        dataset.export(output_dir, format="datumaro")
    # dataset.export(output_dir, format="coco")
    if 'coco' in flags.output_formats:
        splits = [("train", 0.8), ("val", 0.2)]
        task = splitter.SplitTask.segmentation.name
        new_dataset = dataset.transform("split", task=task, splits=splits, seed=42)
        new_dataset.export(output_dir, 
                        #    save_media=True, 
                        format="coco")
    if 'yolo_ultralytics_det' in flags.output_formats:
        splits = [("train", 0.8), ("val", 0.2)]
        task = splitter.SplitTask.detection.name
        new_dataset = dataset.transform("split", task=task, splits=splits, seed=42)
        new_dataset.export(output_dir, 
                        #    save_media=True, 
                           format="yolo_ultralytics")
    if 'yolo_ultralytics_seg' in flags.output_formats:
        # NOTE : YOLO segmentation format is not supported by Datumaro==1.5.2
        convert_coco_json(json_dir=os.path.join(output_dir, 'annotations'), 
                        save_dir=output_dir,  # NOTE : Yolov5 and Yolov7 (Yolov8?) expects label files in the same directory as images
                        use_segments=True,      # This is needed to get RLE masks from COCO format to YOLO format
                        cls91to80=True    # TODO Does this break anything or is even needed? 
                        )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/mnt/d/datasets/sugarbeets', help='Root directory of the generated dataset')
    parser.add_argument('--output_formats',
                    default='datumaro',
                    nargs='+',
                    choices=['datumaro', 'coco', 'yolo_ultralytics_det', 'yolo_ultralytics_seg'],
                    help='Currently supported output labelling formats (default: %(default)s)')
    flags = parser.parse_args()
    convert_to_datumaro(
        os.path.join(flags.base_dir, "main_camera/rect"),
        os.path.join(flags.base_dir, "main_camera_annotations/semantic_segmentation"),
        os.path.join(flags.base_dir, "main_camera_annotations/instance_segmentation"),
        os.path.join(flags.base_dir, "main_camera_annotations/keypoints"),
        os.path.join(flags.base_dir, "datumaro_dataset"),
        classes_to_skip,
        flags=flags,
    )
# syclops-dataconverter

This repository takes input dataset from Syclops engine and converts the labels into standard Deep learning dataset formats using Datumaro and publically available libraries. 

# Usage

```bash
git clone --recursive git@github.com:LALWeco/syclops-dataconverter.git
cd syclops-dataconverter/JSON2YOLO
python3 -m pip install -r requirements.txt
cd .. && python3 -m pip install -r requirements.txt
./scripts/syclops2datumaro.sh /path/to/data/root/dir datumaro yolo_ultralytics_seg coco yolo_ultralytics_det
```

# ToDo

- [x] Convert Datumaro to COCO (det, seg, keypoint)
- [x] Convert Datumaro to YOLOv7 segmentation
- [ ] Convert Datumaro to YOLOv7 keypoints

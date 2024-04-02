# syclops-dataconverter

This repository takes input dataset from Syclops engine and converts the labels into standard Deep learning dataset formats using Datumaro and publically available libraries. 

```bash
scripts/syclops2datumaro.sh /path/to/data/root/dir /path/to/dump/labels
```



# ToDo

- [x] Convert Datumaro to COCO (det, seg, keypoint)
- [x] Convert Datumaro to YOLOv7 segmentation
- [ ] Convert Datumaro to YOLOv7 keypoints

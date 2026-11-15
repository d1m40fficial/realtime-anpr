# Third-party notices

The MIT license in `LICENSE` covers the source code of this repository. Models,
libraries and datasets that were used to build it keep their original licenses.

| Component | Where it is used | License |
|---|---|---|
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | vehicle + plate detectors (`models/yolo11n_plate_best.pt`) | **AGPL-3.0** |
| [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) (FastPlate CCT-XS) | OCR architecture (`models/ru_plate_cct_xs.onnx`) | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | OCR inference | MIT |
| [OpenCV](https://opencv.org/) | video I/O, drawing, CLAHE | Apache-2.0 |
| [AY000554/Car_plate_OCR_dataset](https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset) | OCR fine-tuning / evaluation (45 511 crops) | see dataset card |
| [adilshamim8/license-plate-recognition](https://www.kaggle.com/datasets/adilshamim8/license-plate-recognition) | plate detector training | see dataset card |

## What this means in practice

`models/yolo11n_plate_best.pt` is a fine-tuned Ultralytics checkpoint, so it is
a derivative work of AGPL-3.0 code. If you redistribute the weights or build a
network service around them, the AGPL obligations apply to that part of the
project. The Python code in `src/` is independent of Ultralytics at import time
(post-processing, aggregation and RTSP handling have no Ultralytics dependency)
and can be reused under MIT with any detector you like.

No infrastructure details from the pilot deployment are published: camera
addresses, credentials and network topology have been removed from the code and
from the notebooks. Every frame published in `assets/` was passed through
`scripts/anonymize_clip.py`: plates are pixelated and the displayed strings are
masked, so the pipeline stays visible while the vehicles stay unidentifiable.

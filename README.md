###🤸 Fall Detection Model Training: Pseudo-Label Generation

This project focuses on leveraging the large-scale COCO dataset to generate high-quality pseudo-labels for training a specialized fall detection model. We use the YOLOv8-pose model to extract dense keypoint information from person detections and apply a heuristic-based filter to classify the detected poses as 'Fall' instances.

##🎯 Methodology: Pose-Based Pseudo-Labeling

The core process involves three steps:

Pose Estimation (YOLOv8n-pose): The pre-trained YOLOv8n-pose model is used to detect all human instances in the COCO dataset and extract 17 keypoints per person.

Ground Truth Filtering: Each person detection is compared against a small existing set of known 'fall' bounding box annotations (Ground Truth).

Pseudo-Label Creation: If a predicted person's bounding box has a high Intersection over Union (IoU > 0.5) with a known 'fall' box, the corresponding keypoint data is extracted, normalized, and saved as a new YOLO Keypoint format label (class ID 0: fall).

The resulting dataset provides a large volume of high-quality keypoint data specifically tailored for supervised fall posture classification.
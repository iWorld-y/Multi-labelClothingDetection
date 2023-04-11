yolo \
  task=detect \
  mode=train \
  model="models/yolov8s.pt" \
  data="dataset/train_1.yaml" \
  epochs=10 \
  imgsz=640 \
  batch=40 \
  save=true \
  patience=50

yolo \
  task=detect \
  mode=train \
  model="models/yolov8n.pt" \
  data="dataset/train_1.yaml" \
  epochs=100 \
  imgsz=640 \
  batch=192 \
  save=true \
  patience=50

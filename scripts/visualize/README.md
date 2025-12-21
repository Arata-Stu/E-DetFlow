```bash
python3 visualize_detection.py \
model=rvt \
+train_task=detection \
output_path=./video/output.mp4 \
gt=false \
pred=true \
dataset=SEVD \
dataset.path=/media/arata-22/AT_2TB/dataset/carla_preprocessed \
ckpt_path=/home/arata-22/Downloads/E-Det_rvt_SEVD_50ms/ff8htjq1/checkpoints/epoch_001-step_210000-val_AP_0.44.ckpt 
```
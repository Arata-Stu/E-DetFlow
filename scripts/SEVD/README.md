# SEVD data preprocess

```bash
bash 1_organize_sequences.bash
python3 2_split_sequence.py
python3 3_concat_event_npz.py /mnt/ssd-4tb/dataset/carla --downsample --workers 10
```

## フィルタなし
```bash
## フィルタなし
python3 4_convert_bbox.py /mnt/ssd-4tb/dataset/carla/

python3 5_preprocess_dataset.py /mnt/ssd-4tb/dataset/carla/ /mnt/ssd-4tb/dataset/carla_preprocessed/ \
conf_preprocess/split_SEVD.yaml \
conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml \
conf_preprocess/filter_SEVD.yaml \
-d SEVD \
--downsample \
-np 5 
```

## フィルタあり
```bash
## フィルタあり
python3 4_convert_bbox.py /mnt/ssd-4tb/dataset/carla/ --filter_static --threshold 0.1 --duration 1.0

python3 5_preprocess_dataset.py /mnt/ssd-4tb/dataset/carla/ /mnt/ssd-4tb/dataset/carla_preprocessed_filtered/ \
conf_preprocess/split_SEVD.yaml \
conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml \
conf_preprocess/filter_SEVD.yaml \
-d SEVD \
--downsample \
-np 5 \
--filtered_label 
```
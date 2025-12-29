# SEVD data preprocess

```bash
DATASET_DIR="/path/to/dataset"
bash 1_organize_sequences.bash ${DATASET_DIR}
python3 2_split_sequence.py ${DATASET_DIR}
python3 3_concat_event_npz.py ${DATASET_DIR} --downsample --num_workers 10 --orig_size 1280 960
```

## フィルタなし
```bash
DATASET_DIR="/path/to/dataset"
OUTPUT_DIR="/path/to/output"
## フィルタなし
python3 4_format_bbox.py ${DATASET_DIR}
python3 5_format_flow.py ${DATASET_DIR} --downsample --num_workers 10 --orig_size 1280 960
python3 6_preprocess_dataset.py ${DATASET_DIR} ${OUTPUT_DIR} \
conf_preprocess/split_SEVD.yaml \
conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml \
conf_preprocess/filter_SEVD.yaml \
-d SEVD \
--downsample \
-np 5 \
--ignore_yaml conf_preprocess/ignore_SEVD.yaml
```

## フィルタあり
```bash
DATASET_DIR="/path/to/dataset"
OUTPUT_DIR="/path/to/output"
## フィルタあり
python3 4_convert_bbox.py ${DATASET_DIR} --filter_static --threshold 0.1 --duration 1.0
python3 5_format_flow.py ${DATASET_DIR} --downsample --num_workers 10
python3 6_preprocess_dataset.py ${DATASET_DIR} ${OUTPUT_DIR} \
conf_preprocess/split_SEVD.yaml \
conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml \
conf_preprocess/filter_SEVD.yaml \
-d SEVD \
--downsample \
-np 5 \
--filtered_label \
--ignore_yaml conf_preprocess/ignore_SEVD.yaml
```
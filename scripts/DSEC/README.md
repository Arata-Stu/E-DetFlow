# DSEC data preprocess

## DSEC Detection
```bash
DATASET_DIR="/path/to/dataset"
OUTPUT_DIR="/path/to/output"
## DSEC-Det

python3 2_preprocess_dataset.py ${DATASET_DIR} ${OUTPUT_DIR} \
conf_preprocess/split_DSEC-det.yaml \
conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml \
conf_preprocess/filter_DSEC.yaml \
-d DSEC \
-np 5 \
--ignore_yaml conf_preprocess/ignore_DSEC.yaml
```

## DESC-Flow
```bash
DATASET_DIR="/path/to/dataset"
OUTPUT_DIR="/path/to/output"
## DSEC-Flow
python3 1_format_flow.py ${DATASET_DIR} ${DATASET_DIR} --output_dir ${OUTPUT_DIR} --num_workers 10 --orig_size 640 480 --config conf_preprocess/split_DSEC-flow.yaml 

python3 2_preprocess_dataset.py ${DATASET_DIR} ${OUTPUT_DIR} \
conf_preprocess/split_DSEC-flow.yaml \
conf_preprocess/representation/stacked_hist.yaml \
conf_preprocess/extraction/const_duration.yaml \
conf_preprocess/filter_DSEC.yaml \
-d DSEC \
-np 5 \
--ignore_yaml conf_preprocess/ignore_DSEC.yaml
```

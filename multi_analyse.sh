
# Test first video
MODEL="/home/angel/experiments/Pretrained/exp1_xtttv/checkpoints/"
VIDEO="--video_path /home/angel/HET-CAM/DSC_1098"
MAX="--max_x 1500" # 25 s

WEIGTHS="--weight_path ${MODEL}yolo_ckpt_40.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name xtttv_40

WEIGTHS="--weight_path ${MODEL}yolo_ckpt_80.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name xtttv_80

WEIGTHS="--weight_path ${MODEL}yolo_ckpt_120.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name xtttv_120


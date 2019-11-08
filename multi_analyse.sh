
# Test
MODEL="/home/angel/experiments/Pretrained/exp1_ttvxt/checkpoints/"
VIDEO="--video_path /home/angel/HET-CAM/DSC_1107.MOV"
MAX="--max_x 1500" # 25 s

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_40.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name xtttv_40

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_80.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name xtttv_80

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_120.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name xtttv_120


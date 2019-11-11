# DSC_1089
MODEL="/home/angel/experiments/Pretrained/exp1_xtttv/checkpoints/"
VIDEO="--video_path /home/angel/HET-CAM/DSC_1089.MOV"
MAX="--max_x 1500" # 25 s

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_40.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name xtttv_40

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_80.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name xtttv_80

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_120.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name xtttv_120

# DSC_1098
MODEL="/home/angel/experiments/Pretrained/exp1_vxttt/checkpoints/"
VIDEO="--video_path /home/angel/HET-CAM/DSC_1098.MOV"
MAX="--max_x 1500" # 25 s

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_40.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name vxttt_40

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_80.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name vxttt_80

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_120.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name vxttt_120

# DSC_1104
MODEL="/home/angel/experiments/Pretrained/exp1_tvxtt/checkpoints/"
VIDEO="--video_path /home/angel/HET-CAM/DSC_1104.MOV"
MAX="--max_x 1500" # 25 s

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_40.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name tvxtt_40

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_80.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name tvxtt_80

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_120.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name tvxtt_120


# DSC_1107
MODEL="/home/angel/experiments/Pretrained/exp1_ttvxt/checkpoints/"
VIDEO="--video_path /home/angel/HET-CAM/DSC_1107.MOV"
MAX="--max_x 1500" # 25 s

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_40.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name ttvxt_40

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_80.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name ttvxt_80

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_120.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name ttvxt_120

# DSC_1109
MODEL="/home/angel/experiments/Pretrained/exp1_tttvx/checkpoints/"
VIDEO="--video_path /home/angel/HET-CAM/DSC_1109.MOV"
MAX="--max_x 2100" # 35 s

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_40.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name tttvx_40

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_80.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name tttvx_80

WEIGTHS="--weights_path ${MODEL}yolov3_ckpt_120.pth"
python3 analyse.py $WEIGTHS $VIDEO $MAX --output_name tttvx_120
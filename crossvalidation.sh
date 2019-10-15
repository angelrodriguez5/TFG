echo Experiment 1 - tttvx

cp data/crossvalidation/1-tttvx/* data/custom/
python3 train.py --experiment_name tttvx_150 --epochs 150

echo Experiment 2 - xtttv

cp data/crossvalidation/2-xtttv/* data/custom/
python3 train.py --experiment_name xtttv_150 --epochs 150

echo Experiment 3 - vxttt

cp data/crossvalidation/3-vxttt/* data/custom/
python3 train.py --experiment_name vxttt_150 --epochs 150

echo Experiment 4 - tvxtt

cp data/crossvalidation/4-tvxtt/* data/custom/
python3 train.py --experiment_name tvxtt_150 --epochs 150

echo Experiment 5 - ttvxt

cp data/crossvalidation/5-ttvxt/* data/custom/
python3 train.py --experiment_name ttvxt_150 --epochs 150

# Leave dataset as with its default values
cp data/crossvalidation/1-tttvx/* data/custom/
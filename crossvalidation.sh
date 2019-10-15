echo Experiment 1 - tttvx

cp data/crossvalidation/1-tttvx/* data/custom/
python3 train.py --experiment_name tttvx_test1 --epochs 1

echo Experiment 2 - xtttv

cp data/crossvalidation/2-xtttv/* data/custom/
python3 train.py --experiment_name xtttv_test1 --epochs 1
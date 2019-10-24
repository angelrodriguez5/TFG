echo Experiment 1 - Learning rate 1e-2

python3 train.py --experiment_name lr-2 --epochs 150 --model_def config/learningRates/lr-2.cfg

echo Experiment 2 - Learning rate 1e-3

python3 train.py --experiment_name lr-3 --epochs 150 --model_def config/learningRates/lr-3.cfg

echo Experiment 3 - Learning rate 1e-4

python3 train.py --experiment_name lr-4 --epochs 150 --model_def config/learningRates/lr-4.cfg

echo Experiment 4 - Learning rate 1e-5

python3 train.py --experiment_name lr-5 --epochs 150 --model_def config/learningRates/lr-5.cfg

echo Experiment 5 - Learning rate 1e-6

python3 train.py --experiment_name lr-6 --epochs 150 --model_def config/learningRates/lr-6.cfg

# Log finish date
echo Finished at:
date

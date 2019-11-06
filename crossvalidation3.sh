NAME1="LearningRate_1e-2"
PARAMS1="--pretrained_weights Yolo/weights/darknet53.conv.74 --augmentation True --model_def config/learningRates/lr-2.cfg"
NAME2="LearningRate_1e-4"
PARAMS2="--pretrained_weights Yolo/weights/darknet53.conv.74 --augmentation True --model_def config/learningRates/lr-4.cfg"
NAME3="LearningRate_1e-5"
PARAMS3="--pretrained_weights Yolo/weights/darknet53.conv.74 --augmentation True --model_def config/learningRates/lr-5.cfg"

echo Experiment 1 - tttvx
cp data/crossvalidation/1-tttvx/* data/custom/
python3 train.py --experiment_name exp1_tttvx $PARAMS1
python3 train.py --experiment_name exp2_tttvx $PARAMS2
python3 train.py --experiment_name exp3_tttvx $PARAMS3

echo Experiment 2 - xtttv
cp data/crossvalidation/2-xtttv/* data/custom/
python3 train.py --experiment_name exp1_xtttv $PARAMS1
python3 train.py --experiment_name exp2_xtttv $PARAMS2
python3 train.py --experiment_name exp3_xtttv $PARAMS3

echo Experiment 3 - vxttt
cp data/crossvalidation/3-vxttt/* data/custom/
python3 train.py --experiment_name exp1_vxttt $PARAMS1
python3 train.py --experiment_name exp2_vxttt $PARAMS2
python3 train.py --experiment_name exp3_vxttt $PARAMS3

echo Experiment 4 - tvxtt
cp data/crossvalidation/4-tvxtt/* data/custom/
python3 train.py --experiment_name exp1_tvxtt $PARAMS1
python3 train.py --experiment_name exp2_tvxtt $PARAMS2
python3 train.py --experiment_name exp3_tvxtt $PARAMS3

echo Experiment 5 - ttvxt
cp data/crossvalidation/5-ttvxt/* data/custom/
python3 train.py --experiment_name exp1_ttvxt $PARAMS1
python3 train.py --experiment_name exp2_ttvxt $PARAMS2
python3 train.py --experiment_name exp3_ttvxt $PARAMS3

# Leave dataset as with its default values
cp data/crossvalidation/1-tttvx/* data/custom/

# Organise experiments in folders
cd /home/angel/experiments
mkdir $NAME1
mv exp1* $NAME1
mkdir $NAME2
mv exp1* $NAME2
mkdir $NAME3
mv exp1* $NAME3

# Logs to dropbox
mkdir /home/angel/Dropbox/$NAME1
cp $NAME1/*/logs/* /home/angel/Dropbox/$NAME1
mkdir /home/angel/Dropbox/$NAME2
cp $NAME2/*/logs/* /home/angel/Dropbox/$NAME2
mkdir /home/angel/Dropbox/$NAME3
cp $NAME3/*/logs/* /home/angel/Dropbox/$NAME3

# Log finish date
echo Finished at:
date
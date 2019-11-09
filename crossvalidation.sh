NAME1="ObjScale_1-50"
PARAMS1="--pretrained_weights Yolo/weights/darknet53.conv.74 --augmentation True"


echo Experiment 1 - tttvx
cp data/crossvalidation/1-tttvx/* data/custom/
python3 train.py --experiment_name exp1_tttvx $PARAMS1

echo Experiment 2 - xtttv
cp data/crossvalidation/2-xtttv/* data/custom/
python3 train.py --experiment_name exp1_xtttv $PARAMS1

echo Experiment 3 - vxttt
cp data/crossvalidation/3-vxttt/* data/custom/
python3 train.py --experiment_name exp1_vxttt $PARAMS1

echo Experiment 4 - tvxtt
cp data/crossvalidation/4-tvxtt/* data/custom/
python3 train.py --experiment_name exp1_tvxtt $PARAMS1

echo Experiment 5 - ttvxt
cp data/crossvalidation/5-ttvxt/* data/custom/
python3 train.py --experiment_name exp1_ttvxt $PARAMS1

# Leave dataset as with its default values
cp data/crossvalidation/1-tttvx/* data/custom/

# Organise experiments in folders
cd /home/angel/experiments
mkdir $NAME1
mv exp1* $NAME1

# Logs to dropbox
mkdir /home/angel/Dropbox/$NAME1
cp $NAME1/*/logs/* /home/angel/Dropbox/$NAME1

# Log finish date
echo Finished at:
date
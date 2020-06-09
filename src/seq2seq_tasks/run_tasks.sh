mkdir -p logs
mkdir -p data/models
mkdir -p data/vocab
for seed in `seq 1 5`; do
   for coeff in 0.0 1.0 0.1; do
	for task in binary-flip rev copy en-de; do
            unbuffer python3.6 -u train.py --task $task --epochs 30 --loss-coef $coeff --seed $seed --batch-size 128  | tee -a logs/task=$task\_coeff=$coeff\_seed=$seed.txt;
            echo completed the config $task, coeff: $coeff, seed: $seed num;
        done;
    done;
done;

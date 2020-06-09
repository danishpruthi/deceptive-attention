mkdir -p logs
mkdir -p data/models
mkdir -p data/vocab
for seed in `seq 6 10`; do
    for task in binary-flip rev copy en-de; do
        unbuffer python3.6 -u train.py --debug --task $task --epochs 30 --loss-coef 0.0 --seed $seed --uniform | tee -a logs/task=$task\_uniform_coeff=0.0_seed=$seed;
        unbuffer python3.6 -u train.py --debug --task $task --epochs 30 --loss-coef 0.0 --seed $seed --no-attn | tee -a logs/task=$task\_no-attn_coeff=0.0_seed=$seed;
        echo completed the config $task, seed: $seed;
    done;
done;

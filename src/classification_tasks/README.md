## Classification Tasks

This part of the code will help you reproduce parts of Table 3 in the paper.


### Dependencies

For the dependencies, please check the `environment.yml` file in the parent directory. To create the same conda environment you can run `conda env create -f environment.yml`

### Datasets 

We have included the `SST+Wiki`, `Pronoun`, and `Occupation Classification` datasets in the paper. Unfortunately, we do not have permissions to share the reference letters dataset in the paper :pensive:

To run for the 3 tasks, using `Embedding+Attention` and `BiLSTM+Attention` model:

`CUDA_VISIBLE_DEVICES=0 bash run_tasks.sh`

Once the runs complete, you should be able to check the last few lines of the log files and find statistics like the following (for bigram-flip task, seed=1)

```
Final Test Accuracy ..........  100.00
Final Test Attention Mass ....  93.34
Convergence time in seconds ..  478.60
Sample efficiency in epochs ..  3
```


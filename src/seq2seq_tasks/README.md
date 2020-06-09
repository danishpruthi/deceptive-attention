## Sequence to sequence tasks

This part of the code will help you reproduce the Table 4 in the paper.


For the dependencies, please check the `environment.yml` file in the parent directory. To create the same conda environment you can run `conda env create -f environment.yml`


To run for all the 4 tasks, run:

`CUDA_VISIBLE_DEVICES=0 bash run_tasks.sh`

Once the runs complete, you should be able to check the last few lines of the log files and find statistics like the following (for bigram-flip task, seed=1)

```
Final Test Accuracy ..........  100.00
Final Test Attention Mass ....  93.34
Convergence time in seconds ..  478.60
Sample efficiency in epochs ..  3
```

A sample log file is available in the logs directory. You can effectively `grep`, `cut` the log files to attain the summarized results as in Table 4 in the paper.

For **no attention** and **uniform attention** baselines run the following:

`CUDA_VISIBLE_DEVICES=0 bash run_uniform_no_attn_baselines.sh`
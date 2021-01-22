## Classification Tasks

This part of the code will help you reproduce parts of Table 3 in the paper.


### Dependencies

For the dependencies, please check the `environment.yml` file in the parent directory. To create the same conda environment you can run `conda env create -f environment.yml`

### Datasets 

We have included the `sst-wiki`, `pronoun`, and `occupation-classicification` datasets in the paper. Unfortunately, we do not have permissions to share the reference letters dataset in the paper :pensive:

To run for the 3 tasks, using `Embedding+Attention` and `BiLSTM+Attention` model:

`CUDA_VISIBLE_DEVICES=0 bash run_tasks.sh`

Once the runs complete, you should be able to check the last few lines of the log files and find statistics like the following (for bigram-flip task, seed=1)

```
Final Test Accuracy ..........  100.00
Final Test Attention Mass ....  93.34
Convergence time in seconds ..  478.60
Sample efficiency in epochs ..  3
```

**Update**: For running BERT experiments, you will have to run `bash run_bert_tasks.sh`, where it will run on the three datasets (`pronoun-bert`, `sst-wiki-bert` and `occupation-classicification`). 

You would be able to see the dev results and test results in the output folder. Alternatively, you can also search in the log files. For example, running `cat occupation_classification_mean_0.1.log | grep "test results\|dev results\|acc =\|avg_mean_attention_mass"` will give you results like following for each epoch:

```
01/17/2021 13:22:51 - INFO - __main__ -   ***** dev results *****
01/17/2021 13:22:51 - INFO - __main__ -     acc = 0.9761810242159588
01/17/2021 13:22:51 - INFO - __main__ -     avg_mean_attention_mass = 7.405247478533373e-05
01/17/2021 13:23:35 - INFO - __main__ -   ***** test results *****
01/17/2021 13:23:35 - INFO - __main__ -     acc = 0.9724042088544769
01/17/2021 13:23:35 - INFO - __main__ -     avg_mean_attention_mass = 7.117244752855244e-05
````

`avg_mean_attention_mass` in the BERT code is the attention metric of interest (and `avg_max_attention_mass` corresponding to the max setting). Here we get a value of 7.405247478533373e-5, which when converted to a percentage becomes 0.007, and hence was rounded up to 0.01 in the paper. 



Lastly, note that the above BERT experiments use a significantly outdated version of what has now become the `transformers` library from Huggingface (earlier it was called `pretrained-pytorch-bert`). The local copy of that library is included as a part of this repository. 

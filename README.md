# Learning to Deceive with Attention-based Explanations
Code & associated data for the following paper.

> [Learning to Deceive with Attention-Based Explanations](https://arxiv.org/pdf/1909.07913.pdf)
> 
> *Danish Pruthi, Mansi Gupta, Bhuwan Dhingra, Graham Neubig, Zachary C. Lipton*
> 
> The 58th Annual Meeting of the Association for Computational Linguistics (ACL-20).



For dependencies, please check the `environment.yml` file in `src` directory. To create the same conda environment you can run `conda env create -f environment.yml` (You might have to edit the prefix in the last line in the file.) 

Please refer to README files for our experiments on [classification tasks](src/classification_tasks/README.md), and [sequence-to-sequence tasks](src/seq2seq_tasks/README.md).  

The examples alongside attention-based explanations used for the human-subject experiment:

- [Organic attention](https://docs.google.com/document/d/10sOLMX00OUAH7kojf1va7bP8Y5px9qhQMUhy3Vd-d9Y/edit?usp=sharing)
- [Manipulated attention from Wiegreffe & Pinter, 2019](https://docs.google.com/document/d/1KXxcGYCkDw6Rc0EJec0PyK_6hNuGqCTCRJf-OdpHrTM/edit?usp=sharing)
- [Attention from our manipulation scheme](https://docs.google.com/document/d/1LkgBFsoGcdBToscBBmNf4hNvvq2OJLM1I3kNf9ZF3cI/edit?usp=sharing)

# Bibtex

```
@article{pruthi2020learning,
  title={Learning to Deceive with Attention-Based Explanations},
  author={Pruthi, Danish and Gupta, Mansi and Dhingra, Bhuwan and Neubig, Graham and Lipton, Zachary C},
  booktitle = {The 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  address = {Seattle, USA},
  month = {July},
  year = {2020}
}
```

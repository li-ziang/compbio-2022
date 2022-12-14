## Homework 3 

### Reference

This repository is modified from the officially released code of [scETM](https://www.nature.com/articles/s41467-021-25534-2). One can refer to [original repository](https://github.com/hui2000ji/scETM) for more details.



### Reimplement Experiment Results 

For convenicence, all experimental configurations are explicitly written in `main.py`. For reimplementation our results, one can run follwing command:

```
main.py
```

To evaluation, one can run code in:

```
evaluate.py 
```





### Quantitative Evaluation

###### Classifier Accuracy

| Methods  |   Ours    | Lasso | Difference Analysis | Genes from Literature |
| :------: | :-------: | :---: | :-----------------: | :-------------------: |
| Accuracy | **86.7%** | 92.8% |        54.3%        |         76.4%         |



###### IoU Score

| Methods  |   Ours   | Lasso | Difference Analysis | Genes from Literature |
| :------: | :------: | :---: | :-----------------: | :-------------------: |
| Accuracy | **1.4%** | 2.3%  |        1.7%         |         10.1%         |


## Homework 2 

### Possible Directions

##### Preprocessing + Shared Weights Mapping

##### Mapping Multi-omics Data to Same Latent Space

"maybe add more here"



### Reference Paper

| Reference Paper                                              | Code                                          | Data                                                         | Type            |
| ------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------ | --------------- |
| [scJoint](https://www.nature.com/articles/s41587-021-01161-6) | [here](https://github.com/SydneyBioX/scJoint) | [here](https://www.nature.com/articles/s41587-021-01161-6#data-availability) | Semi-supervised |
| [MOFA+](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02015-1) | TBD                                           | TBD                                                          | TBD             |
| More?                                                        |                                               |                                                              |                 |

##### scJoint

![image-20221025161003056](/Users/bingliangzhang/Desktop/compbio-2022/hm2/README.assets/image-20221025161003056.png)

* First transfer **scATAC-seq** to **gene activitity scores** (same format as scRNA-seq)

* Shared MLP guarantees **same latent space** for different omics data.

  

##### MOFA++

TBD



### Progress

###### Data Related:

- [ ] Prepare and get familiar with dataset of **scJoint**. 
- [ ]  Maybe provide some flexible loading function (code).
- [ ] Explore the data feature, such as sparsity.

###### Algorithm Related: 

- [ ] Get familar with code of **scJoint** and find which parts are reuseable.
- [ ] Reproduce some basic result of scJoint (not sure if necessary)
- [ ] Propose our own solution/method (I guess it can be a simple one but still necessary for this homework)
- [ ] Some basic visualization

###### Evaluation&Analysis Related:

- [ ] Maybe get familiar with how **scJoint** evaluate and anylyse their model
- [ ] Provide some evaluation function (code) taking in the trained model

- [ ] Some comparison between existing method, pros and cons (not sure if necessary)

###### More Direction Related: 

- [ ] Update any possible dataset and reference paper that you think might be useful
- [ ] More problem settings with exact input and output

###### Report&Presentation Related (finally):

- [ ] Final pdf report
- [ ] Some slides for presentation next week

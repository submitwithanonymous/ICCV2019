## Introduction

This is the source code and additional visualization examples of our Radial-GCN, Radial Graph Convolutional Network for Visual Question Generation.

1) Different from the existing approaches that typically treat the VQG task as a reversed VQA task, we propose a novel answer-centric approach for the VQG task, which effectively models the associations between the answer and its relevant image regions.
2)  To our best knowledge, we are the first to apply GCN model for the VQG task and devise a new radial graph structure with graphic attention for superior question generation performance and interpretable model behavior. 
3)  We conduct comprehensive experiments on three benchmark datasets to verify the advantage of our proposed method on generating meaningful questions on the VQG task and boosting the existing VQA methods on the challenging zero-shot VQA task.

![framework](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/framwork_new_.png)

<br>
<br>

## Code Structure
More run details would be released later.
```

├── Radial-GCN/
|	  ├── run_vqg.py						    /* The main run files
|	  ├── layer_vqg.py					    /* Files for the model layer and structure (GCN, VQG)
|	  ├── dataset_vqg.py				    /* Files for construct vqg dataset
|	  ├── utils.py						      /* Files for tools
|	  ├── main.py							      /* Files for caption evaluation
|	  ├── supp_questions				    /* Files for generate questions for supplementary dataset for zero shot VQA
|	  ├── draw_*.py						      /* Files for drawing and visualisation
│   ├── ZS_VQA/
| 	    ├── data/						      /* Data file for zs_vqa
│   ├── data/         				    /* Data files for training vqg
|		    ├── tools/						    /* The modified file from 	 bottom-up attention
|		    ├── process_image_vqg.py  /* Files for preprocess image
|		    ├── preprocess_text.py	  /* Files for preprocess text
│   ├── readme.md

```
<br>

## Visual Examples
More details can be refer to our **main text** and **supplementary**.
<br>
### View VQG Process  
![VQG Process](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/visual_new3.png)

<br>
<br>   

### View Question Distribution
![Distribution](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/tsne_vis.png)

<br>
<br>  

### View Supp. for ZS-VQA
![Supp](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/supp_q.png)

<br>
<br>  

### View More Examples
![More Examples](https://github.com/submitwithanonymous/ICCV2019/raw/master/fig/visual3.png)

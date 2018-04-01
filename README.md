# Brain Tumor Segmention based on Dilated Convolution Refine Networks

This is an implementation of **Brain Tumor Segmention based on Dilated Convolution Refine Networks** in TensorFlow for  on the BRATS2015

## Model Description
The Dilated Convolution Refine Components is built on a fully convolutional variant of [ResNet-50](https://github.com/KaimingHe/deep-residual-networks) , the decoder stage  to acquire the output of the same size as the input, deconvolution layers are applied. The Dilated Convolution Refine Components is introduce to increase the sensitivity of the network to detailed information on networks multiple branches. For more details please refer to the following Figure 1:

 ![Alt text](https://github.com/wei-lab/DCR/raw/master/image/pipeline.png)
 

Fig. 1.	The framework of Brain Tumor Segmentation pipeline

The model is trained on a mini-batch of MRI images and corresponding ground truth masks with the softmax classifier  at the top, the Cross entropy  and L2 are training loss, during inference, the softmax layers is discard, The final segmentation mask is computed using argmax over the logits.
On the test set of  BRATS2015, the model achieves tumor segmentation results with a DEC score of<code>0.87</code>  and PPV score of <code>0.91`</code>.

For more details on the underlying model please refer to the following paper:

    The address of the paper will be updated in a few days

## Requirements
The script has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):
TensorFlow needs to be installed before running the scripts. TensorFlow v1.4.0 is supported; 

> numpy==1.14.1
> 
>  scipy==1.0.0 
>  
>  networkx==2.1
>  
>   SimpleITK==1.0
>   
>   matplotlib>=1.3.1


## usage
#### Training
Step1.  Download the data set and put it in the data directory.

Step2.  Run readfile.py in order to get the training data directory.

    python readfile.py
Step3.   Run   create_pascal_tf_record.py  to generate tfrecord data file:

    python create_pascal_tf_record.py
   
Step4.   Training model（Recommend Training on the GPU）:
	
```
python Train.py 
```
#### Inference
Step1.   Run   create_pascal_tf_record.py  to generate test tfrecord data file(modify the data path):

```
python create_pascal_tf_record.py
```

To perform inference over your own MRI, use the following command:

    python Inference.py


# Example of result
Visual example of our model semantic segmentation results：

![Alt text](https://github.com/wei-lab/DCR/raw/master/image/example.png)


# License

> School of Software Yunnan University




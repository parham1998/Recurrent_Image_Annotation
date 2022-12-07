# Recurrent_Image_Annotation
Implementation of some popular Recurrent Image Annotation papers on Corel-5k dataset with PyTorch library

## Dataset
<div align="justify"> There is a 'Corel-5k' folder that contains the (Corel-5k) dataset with 5000 images, which has 260 labels in the vocabulary. </div>

(for more information see [CNN_Image_Annotation_dataset](https://github.com/parham1998/CNN_Image_Annotaion#dataset))

## Long short-term memory (LSTM) 
![0](https://user-images.githubusercontent.com/85555218/138563103-f02523b5-f2b7-4a1b-99a7-b448d9d2d031.png)

**Performance of LSTM in one time step:**

![1](https://user-images.githubusercontent.com/85555218/138563233-703a5348-751d-4b04-ba5a-c8e743dfc65f.gif)
![2](https://user-images.githubusercontent.com/85555218/138563236-2c11bad0-b7d2-4ac3-87cf-4fd59112a870.gif)
![3](https://user-images.githubusercontent.com/85555218/138563237-885b3ac9-b0af-48b3-8f37-4ef80959f006.gif)
![4](https://user-images.githubusercontent.com/85555218/138563238-517344f7-b653-42ca-b831-30a28398c16d.gif)

## Convolutional neural network (CNN) 
<div align="justify"> As compared to other CNNs in my experiments, TResNet produced the best results for extracting features of images, so it has been chosen as the feature extractor. </div>

(more information can be found at [CNN_Image_Annotation_convolutional_models](https://github.com/parham1998/CNN_Image_Annotaion#convolutional-models))

## CNN+LSTM models
<div align="justify"> 
1) RIA: <br >
RIA is an encoder-decoder model that uses CNN as an encoder and LSTM as a decoder. In the training phase, it is trained using training images and human annotations. It is necessary to sort the label set as a label sequence before using the annotations as the input for LSTM. A <b>rare-first</b> order is used, which put the rarer label before the more frequent ones (based on label frequency in the dataset). During the test phase, the RIA model predicts the first output label after receiving the input image and being triggered by the <b>start</b> signal. Using the previous output as input for the next time step, it predicts the tag sequence recursively. The loop will continue until the <b>stop</b> signal is predicted. Its structure for the test phase is shown in the images below: </div>

![RIA](https://user-images.githubusercontent.com/85555218/201529357-3ddd0597-547b-43eb-8a24-78a305d62cb6.jpg)
<div align="justify"> The labels are mapped to embedding vectors by using lookup tables instead of one-hot vectors. The lookup table can be trained and learn what kind of representation to generate. However, experiments have shown that using pre-trained weights like the GloVe embedding weights provide better results. </div> <br >

<div align="justify"> 
2) SR-CNN-RNN: <br >
SR-CNN-RNN is another encoder-decoder model that has a similar architecture to RIA. The differences between them are that semantic concept learning is now done by the CNN model, which uses input images to generate a probabilistic estimate of semantic concepts. In order to generate label sequences, the RNN model takes the concept of probability estimates and models their correlations. Its structure for the test phase is shown in the images below: </div>

![SR-CNN-RNN](https://user-images.githubusercontent.com/85555218/201529366-83c4b665-2349-4dde-8130-598407e2d333.jpg)

<div align="justify"> 
3) CNN-RNN + Attention: <br >
Attention networks are widely used in deep learning. Models can use them to determine which parts of the encoding are relevant to the related task. Using the attention mechanism, pixels with more importance can be highlighted in image annotation. In most cases, labels are conceptual and cannot be annotated by the objects that appear in the image. Therefore, the attention mechanism is not able to improve results significantly. Its structure for the test phase is shown in the images below: </div>

![Attention](https://user-images.githubusercontent.com/85555218/203550694-5b899e1e-b8b1-4cab-ac9c-9fdd59b4fd8f.jpg)
<b> Inspired by [Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) </b> 

<b> Some examples of attention: </b>
![1](https://user-images.githubusercontent.com/85555218/206273788-b8cdd35e-7eff-4bf3-ba76-02c7438f228e.png)
![2](https://user-images.githubusercontent.com/85555218/206273810-b31802ed-b237-4f0d-b6bb-f6b1a8976f70.png)
![3](https://user-images.githubusercontent.com/85555218/206273820-6c0d9f6f-1109-4d09-805f-124582c4bad7.png)

<div align="justify">
4) CNN-RNN + Attention + MLA: <br >
It has been proposed to align the labels to the predictions of the network before computing the loss to reduce the problems caused by imposing a fixed order on the labels. A Hungarian algorithm can be used to solve the minimization problem since it is an assignment problem. So by preserving the attention architecture, we utilize minimal loss alignment (MLA) as the loss function instead of cross-entropy loss. (Furthermore, the frequency of a label is independent of the size of a given object in a dataset. Less frequent but larger objects can cause the LSTM prediction to stop earlier because of their domination in the image and their ranking in the prediction step.) </div>

## Beam Search
<div align="justify">
Choosing the word with the highest score and predicting the next word would be the greedy approach. However, this isn't optimal since the rest of the sequence depends on the first word. It isn't just the first word that determines whether the process is optimal or not; each word in the sequence has consequences for the words following it. (The best sequence might be like this: The third best word might have been selected at the first step, the second best word at the second step, and so on.) <br >
The beam search can be used instead of greedy searches to resolve this issue. However, experiments have shown that the RNN model cannot learn the complicated relationships between labels properly, and using beam search won't have any effect on the result. </div>

## Evaluation Metrics
<div align="justify"> Precision, Recall, F1-score, and N+ are the most popular metrics for evaluating different models in image annotation tasks.
I've used per-class (per-label) and per-image (overall) precision, recall, f1-score, and also N+ which are common in image annotation papers. </div>

(check out [CNN_Image_Annotation_evaluation_metrics](https://github.com/parham1998/CNN_Image_Annotaion#evaluation-metrics) for more information)

## Train and Evaluation
To train and evaluate models in Spyder IDE use the codes below:

<div align="justify"> 1) RIA: </div>

```python
run main.py --method RIA --max-seq-len 5 --order-free None --is_glove --sort
```

```python
run main.py --method RIA --max-seq-len 5 --order-free None --is_glove --evaluate
```

<div align="justify"> 2) SR-CNN-RNN: </div>

```python
run main.py --method SR-CNN-RNN --max-seq-len 5 --order-free None --is_glove --sort
```

```python
run main.py --method SR-CNN-RNN --max-seq-len 5 --order-free None --is_glove --evaluate
```

<div align="justify"> 3) CNN-RNN + Attention: </div>

```python
run main.py --method Attention --max-seq-len 5 --order-free None --is_glove --sort
```

```python
run main.py --method Attention --max-seq-len 5 --order-free None --is_glove --evaluate
```

<div align="justify"> 4) CNN-RNN + Attention + MLA: </div>

```python
run main.py --method Attention --max-seq-len 5 --order-free MLA --is_glove
```

```python
run main.py --method Attention --max-seq-len 5 --order-free MLA --is_glove --evaluate
```

## Results
<div align="justify"> 1) RIA: </div>

| batch-size | num of training images | image-size | epoch time | GloVe weights | features embedding dim | label embedding dim 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| 32 | 4500 | 448 * 448 | 136s | True | 2048 | 300 |
  
| data | precision | recall | f1-score |
| :------------: | :------------: | :------------: | :------------: |
| *testset* per-image metrics | 0.647  | 0.606 | 0.626 | 
| *testset* per-class metrics | 0.409 | 0.421 | **0.415** |

| data | N+ |
| :------------: | :------------: |
| *testset* | 156 |

<div align="justify"> 2) SR-CNN-RNN: 
The CNN and LSTM models were pre-trained with ground truth labels separately, as mentioned in the paper.
</div>

| batch-size | num of training images | image-size | epoch time | GloVe weights | predicted labels embedding dim | label embedding dim 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| 32 | 4500 | 448 * 448 | 135s | True | 2048 | 300 |
  
| data | precision | recall | f1-score |
| :------------: | :------------: | :------------: | :------------: |
| *testset* per-image metrics | 0.680  | 0.616 | 0.646 | 
| *testset* per-class metrics | 0.405 | 0.391 | **0.398** |

| data | N+ |
| :------------: | :------------: |
| *testset* | 145 |

<div align="justify"> 3) CNN-RNN + Attention: </div>

| batch-size | num of training images | image-size | epoch time | GloVe weights | features embedding dim | attention dim | label embedding dim 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| 32 | 4500 | 448 * 448 | 150s | True | 2048 | 1024 | 300 |
  
| data | precision | recall | f1-score |
| :------------: | :------------: | :------------: | :------------: |
| *testset* per-image metrics | 0.663  | 0.616 | 0.638 | 
| *testset* per-class metrics | 0.438 | 0.429 | **0.434** |

| data | N+ |
| :------------: | :------------: |
| *testset* | 160 |

<div align="justify"> 4) CNN-RNN + Attention + MLA: </div>

| batch-size | num of training images | image-size | epoch time | GloVe weights | features embedding dim | attention dim | label embedding dim 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| 32 | 4500 | 448 * 448 | 158s | True | 2048 | 1024 | 300 |
  
| data | precision | recall | f1-score |
| :------------: | :------------: | :------------: | :------------: |
| *testset* per-image metrics | 0.656  | 0.608 | 0.632 | 
| *testset* per-class metrics | 0.449 | 0.413 | **0.431** |

| data | N+ |
| :------------: | :------------: |
| *testset* | 155 |

## References
J. Jin, and H. Nakayama. <br />
*"Recurrent Image Annotator for Arbitrary Length Image Tagging"* (ICPR-2016)

F. Liu, T. Xiang, T. M Hospedales, W. Yang, and C. Sun. <br />
*"Semantic Regularisation for Recurrent Image Annotation"* (CVPR-2017)

V. O. Yazici, A. Gonzalez-Garcia, A. Ramisa, B. Twardowski, and J. van de Weijer <br />
*"Orderless Recurrent Models for Multi-label Classification"* (CVPR-2020)

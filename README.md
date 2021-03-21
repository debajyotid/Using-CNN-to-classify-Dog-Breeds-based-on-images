# Using-CNN-to-classify-Dog-Breeds-based-on-images
---
Great Lakes PGP-AIML November 2019 Batch: Group 1: Capstone
---

## Introduction:
---

The main objective of this capstone project is to identify & classify the breed of a dog, based on the subtle changes in its visual appearance. 
The idea is to explore advanced computer vision techniques to intelligently & correctly predict the breed of a dog. 
Although most dogs confirm to certain visual traits that are unique to dogs, there are sometimes subtle & at time prominent features that help distinguish between different breeds of dogs

In the current task we are trying to build a computer-vision based capability that helps computers to be also able to intuitively discern between dog breeds, without being clearly told what distinguishable features to look for.

The data-set used for this project was hosted by Kaggle, as part of their Kaggle Playground Prediction Competition in 2017. 
The data-set is a strictly canine subset of ImageNet & comprised of 120 breeds of dogs and a limited number training images per class. 
Each image was assigned a unique id and the same was tagged against the breed the dog in the image belonged to in a separate csv file. 

File descriptions:

train.zip - the training set, comprised of images of different dog, belonging to one of 120 breeds. Each image had a filename that is its unique id. 

test.zip – the testing set, where the model is expected to predict the probability of the image belonging to one of the 120 breeds

sample_submission.csv - a sample submission file in the correct format

labels.csv - mapped the breeds, for the images in the train set, with the corresponding image id.

Kaggle authorities extend their gratitude to the creators of the Stanford Dogs Dataset, which made this competition possible: Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li.

We have tried 3 types of image resizing techniques before feeding to our models:
1.128X128X1
2.128X128X3
3.224X224X3

We have predominantly used 80:20 split for training & validation sets, but in the last iteration we have also tried to build the model on 90:10 split.
We have throughout used only 10000 of the total 10222 records for training & validation.
---

## Model:
---

For our modelling activity, we experimented with the below architectures:
1. Custom CNN Architecture: trained on 10222 images of size 128X128X1, split into an 80:20 split for training & validation, resulting in 8177 & 2045 images for each respectively
2. Same Custom CNN architecture as above: this time trained using Image Augmentation on top of the 10222 images of size 128X128X1, split into an 80:20 split for training & validation respectively with 32 images in each batch of augmented images.
3. Custom VGG16 model with pre-trained weights: We trained the VGG16 model on 10222 images of size 128X128X3, split into an 80:20 split for training & validation, resulting in 8177 & 2045 images for each respectively.
4. Pre-trained VGG16 model with weights & architecture unchanged: trained on 10222 images of size 128X128X3, split into an 80:20 split for training & validation, resulting in 8177 & 2045 images for each respectively.
5. Same pre-trained VGG16 as above: this time trained using Image Augmentation on top of the 10222 images of size 128X128X3, split into an 80:20 split for training & validation respectively with 32 images in each batch of augmented images.
6. Resnet50 pre-trained on ImageNet: We trained the Resnet50 model on 10000 images of size 224X224X3 each, split into an 80:20 split for training & validation, resulting in 8000 & 2000 images for each respectively.
7. Pre-trained Resnet50 as above: this time trained using Image Augmentation on top of the 10000 images of size 224X224X3, split into an 80:20 split for training & validation respectively with 32 images in each batch of augmented images.
8. Pre-trained Resnet50 as above: with the layers kept as trainable and trained on 10000 images of size 224X224X3, split into an 80:20 split for training & validation, resulting in 8000 & 2000 images for each respectively
9. MobileNet V2 architecture pre-trained on ImageNet: We trained the model on 10000 images of size 224X224X3, split into an 80:20 split for training & validation, resulting in 8000 & 2000 images for each respectively.
10. Pre-trained MobileNet architecture as above: this time trained using Image Augmentation on top of the 10000 images of size 224X224X3, split into an 80:20 split for training & validation respectively with 32 images in each batch of augmented images.
11. MobileNet architecture pre-trained on ImageNet: trained on 10000 images of size 224X224X3, split into a 90:10 split for training & validation, resulting in 9000 & 1000 images for each respectively
12. Pre-trained MobileNet architecture as above: this time trained using Image Augmentation on top of the 10000 images of size 224X224X3, split into a 90:10 split for training & validation respectively with 32 images in each batch of augmented images.
---

## Optimisation and Regularisation
---

We have kept the parameters consistent across all of the above models, going for Adam as an Optimiser. We have also implemented call-backs to:
1. Save model weights based on model accuracy, if there is no increase beyond 0.1%, over 5 iterations
2. Reduce Learning Rate by a factor of 0.01 if there is no change (we have chosen the parameter as ‘auto’) in validation loss, i.e., no decrease in the same, by at least 0.01% over 5 iterations
---

## Model Training
---

The pre-processed images were fed to the models in batches. 
Either with, or without Image Augmentation, the model was always trained on batches of 32. 
In case of Image Augmentation, the augmented images were also generated in batches of 32 images each for training & validation.
When using Image Augmentation, we have referred to a standard set of augmentation techniques as below:

i. width_shift_range=0.2,
ii. height_shift_range=0.2,
iii. rotation_range=30,
iv. shear_range=0.2,
v. zoom_range=0.2,
vi. horizontal_flip=True,
vii. vertical_flip=True,
viii. fill_mode='nearest'

All models were trained for 20 iterations. We also used call-backs like ModelCheckpoint & ReduceLROnPlateau

---

## Evaluation
---

We have referred to Top-5 Accuracy & ROC-AUC Score to evaluate the model performance across 120 dog breed classes. 
We chose this approach as this being a large multi-class problem, where different classes didn’t find equal representation in the training dataset, we felt looking at traditional evaluation metrics like Accuracy, Precision, Recall, F1-Score wouldn’t offer a truly complete picture. 

---

## Results
---
Model	Model Name	                                          Image Augmentation (Y/N)	Training Accuracy	Validation Accuracy	Val Top 5 Accuracy	ROC-AUC Score
1	    Custom CNN Architecture	                              N	                        1.23%	            1.08%	              5.23%	              0.50
2	    Custom CNN Architecture	                              Y	                        1.24%	            0.99%	              4.86%	              0.49
3	    VGG-16 with pre-trained weights	                      N	                        99.96%	          29.34%	            60.54%	            0.92
4	    VGG-16 with pre-trained weights	                      N	                        54.76%	          25.23%	            53.89%	            0.92
5	    VGG-16 with pre-trained weights	                      Y	                        21.35%	          19.39%	            47.72%	            0.89
6	    Resnet-50 using pre-trained weights	                  N	                        1.15%	            0.75%	              5.05%	              0.5
7	    Resnet-50 using pre-trained weights	                  Y	                        1.09%	            0.76%	              5.04%	              0.5
8	    Resnet-50 with all layers trainable	                  N	                        5.92%	            4.80%	              18.75%	            0.77
9	    Mobilenet V2 with pre-trained weights	                N	                        99.96%	          71.40%	            93.65%	            0.99
10	  Mobilenet V2 with pre-trained weights	                Y	                        74.9%	            65.78%	            91.53%	            0.99
11	  Mobilenet V2 with pre-trained weights on 90:10 split	N	                        98.9%	            67.10%	            91.2%	              0.99
12	  Mobilenet V2 with pre-trained weights on 90:10 split	Y	                        71.62%	          65.12%	            91.94%	            0.99

---



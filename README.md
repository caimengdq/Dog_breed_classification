# Dog_breed_classification

The dataset is sourced originally from the Oxford-IIIT Pet Dataset, which comprises 37 classes, each class containing roughly 200 images of a certain breed of pet (it includes both dogs and cats).

We want to build a quick, efficient model to predict a subset of these breeds. We use a subset(6 classes), because a model designed to predict all 37 classes of the dataset is expensive in terms of resources and time, and as such, a basic version that can be trained on just about any modern computer is significantly more appealing. 

This model is using CNN to classify dog breeds.

Firstly we used a basic convolutional neutral network.
Then we used EfficientNet to finetune the model in order to improve accuracy.

## Multi-object Detection With Deep Learning 

This rep is to solve multi object detection with deep neural networks.

I will start by implementing the first version of the YOLO network. The idea is to subsequently increase the complexity of architecture by implementing sub systems such as spatial transformer network, attention transformers, feature pyramid network, so on and so forth to see how each hyperparameter affects the accuracies. 

### The Dataset

The dataset is a subset of the VOC dataset which contains 5000 images annotated with bounding boxes of objects present in the images.

Let's visualze 25 images from the dataset. For this, I created a class for visualizing the dataset. It is fairly flexibe and modular. It randomly samples 25 images from the dataset directory and plots it on a grid with matplotlib. So far, it does it's job reasonably. 

Let's run it!

<img src="https://raw.githubusercontent.com/deveshdatwani/yolo/main/assets/datasetVisualizer.png" height=400, width=800 align="center">

The images appear to be of different sizes. Therefore there will be some pre-processing needed before feeding each to the network. Or perhaps not, if the network is fully-convolution. I will update these lines once I read the paper today. 






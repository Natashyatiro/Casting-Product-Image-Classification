# Casting Product Image Classification to Detect Defective Products: Project Overview
*	Created a tool that can classify defect and non-defective casting product based on product images.
*	Built and compared Logistic Regression, Tree, Gradient Boosting, Random Forest, Neural Network model using different embedders (Inception V3, VGG-16, VGG-19) in Orange.
*	The best model for this case is logistic regression or neural network with Inception V3 embedders that result in 99.7% of accuracy.

## Data Information
The data consist of 7,348 casting manufacturing product images. These photos are all top view of submersible pump impeller, which are the size of (300*300) pixels grey-scaled images. There are mainly two categories: (1) Defective (2) Non-detective. Data already split into two folders for training and testing.
![image](https://user-images.githubusercontent.com/84263856/201731804-a8e349f9-ed2f-49bb-8a25-3a2b2c11d8e6.png)

## Image Embedder In Orange
Image Embedding reads images and uploads them to a remote server or evaluate them locally. Deep learning models are used to calculate a feature vector for each image. It returns an enhanced data table with additional columns (image descriptors).
In Orange, Image Embedding offers several embedders, each trained for a specific task. Images are sent to a server or they are evaluated locally on the user’s computer, where vectors representations are computed. SqueezeNet embedder offers a fast evaluation on users computer which does not require an internet connection. Below are the image embedding models that Orange provides:
* SqueezeNet: Small and fast model for image recognition trained on ImageNet.
* Inception v3: Google’s Inception v3 model trained on ImageNet.
* VGG-16: 16-layer image recognition model trained on ImageNet.
* VGG-19: 19-layer image recognition model trained on ImageNet.
* Painters: A model trained to predict painters from artwork images.
* DeepLoc: A model trained to analyze yeast cell images.

## Machine Learning Models
### Logistic Regression
Logistic Regression is the appropriate regression analysis to conduct when the dependent
variable is dichotomous (binary). Logistic regression is used to describe data and to
explain the relationship between one dependent binary variable and one or more nominal,
ordinal, interval or ratio-level independent variables.

### Tree
Tree is a simple algorithm that splits the data into nodes by class purity (information
gain for categorical and MSE for numeric target variable). It is a precursor to Random
Forest. Tree in Orange is designed in-house and can handle both categorical and numeric
datasets.

### Gradient Boosting
Gradient Boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction
models, typically decision trees.

### Random Forest
Random Forest builds a set of decision trees. Each tree is developed from a bootstrap
sample from the training data. When developing individual trees, an arbitrary subset
of attributes is drawn (hence the term “Random”), from which the best attribute for
the split is selected. The final model is based on the majority vote from individually
developed trees in the forest.

### Neural Networks
Neural Networks are computing systems inspired by the biological neural networks that
constitute animal brains. Neural Networks are based on a collection of connected units
or nodes called artificial neurons, which loosely model the neurons in a biological brain.
Each connection, like the synapses in a biological brain, can transmit a signal to other
neurons. An artificial neuron receives signals then processes them and can signal neurons
connected to it. The ”signal” at a connection is a real number, and the output of each
neuron is computed by some non-linear function of the sum of its inputs. 

## Model Building and Results in Orange
After trying all the Six embedding models provided by Orange, we chose InceptionV3
because of its highest accuracy. InceptionV3 is Google’s deep neural network for image
recognition. It is trained on the ImageNet data set. The model we are using is available
here. For the embedding, we use the activations of the penultimate layer of the model,
which represents images with vectors.

After embedding by InceptionV3, we trained five machine learning models (i.e. Logistic
Regression, Tree, Gradient Boosting, Random Forest, Neural Network) respectively in
Orange and had them tested using the testset we prepared.
![image](https://user-images.githubusercontent.com/84263856/201733419-08be9cd0-bcd3-4636-b079-35d9ae3347f8.png)

![image](https://user-images.githubusercontent.com/84263856/201733428-5805b430-d655-4dc9-a3b2-ba8246015ffe.png)

![image](https://user-images.githubusercontent.com/84263856/201733435-60c369e0-35ae-4c07-a703-6179c053e40b.png)

Based on the results we get from Orange, we decide to apply Neural Network as our
machine learning model which has a higher classification accuracy, a better receiver operating curve and a relatively more stable performance in future application.

## Business Implications
Using artificial intelligence for visual inspection of casting product can solve business
issues at once, such as bottleneck issue and high defect rate that are caused by manual inspection which is unreliable and expensive. The benefit of implementing AI can
be translated into two major aspects which are operational implications and financial
implications to the company.

### Operational Implications
1. Reduce Inspection Time
2. Increase Inspection Accuracy
3. High scalability

### Financial Implications
1. Increase Production Rate and Decrease Production Cost
2. Reduce Cost Caused by Inspection Errors
3. Reduce Labor Cost

## Conclusion
In this project, we have developed a machine learning model using Neural Network with
Inception v3 Embedders that can successfully applied in inspecting casting product. The
accuracy of the model is really high which is 99.7%. The operational implications by
utilizing AI visual inspection are reducing inspection, increasing inspection accuracy, and
higher scalability. These operational implications can be translated into financial benefits
which are increase in production rate, and decrease in production cost, cost caused by
inspection errors, and labor cost. This leads to higher revenue, lower overall costs, and
higher profit. The company can easily implement this by using our implementation plan
that utilizes simple optical system or utilize AI for broader purposes to conduct root
cause analysts, predictive maintenance, and production optimization.

## Code and Resources Used
You can access the full report [here](https://github.com/Natashyatiro/casting-product-image-classification/blob/main/Full%20Report.pdf).

**Dataset Source:** https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product 

**Orange Version:** 3.32


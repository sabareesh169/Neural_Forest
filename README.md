# Neural_Forest

##Purpose
Traditional Neural networks require that the input features must not contain NaN values and hence require imputation for them to not throw out any errors. The most coomon methods of imputation are generally mean or median (if the dataset contains outliers). There are other imputation methods which are more efficient like imputation with kNN or random forest. But these have there own drawbacks like KNN being computationally expensive or you need to build a random forest for the latter. So, there is a need to come up with an algorithm for datasets which have missing values for multiple features.
 
##Inspiration
One of the most useful algorithms formissing value datasets is Random Forests but they have the disadvantages of inability to use for online machine learning or relatively poor performance for regression problems. Neural network is one algorithm which can adress these advantages. So, a mix of both these algorithms would be a good idea to take advantages of both these algorithms.

The reason for Random Forest being efficient is largely due to two attributes of the Decision tree. 
a) High variance or overfitting
b) Model highly depnedent on the training set. (Addition or removing even one datapoint can give an entirely different model)

And fortunately enough for us neural networks can also be highly overfitting and they can give rise to different models if we select different features to use for building each model. So, we can try to use neural networks to become the building block of a random forest algorithm. 

Most of the times, there are always some features which are most important for building a model. In random forests, we select subset of features at every split which means that we get to use the most important feature at some point of the decision tree. But this is not a possiblity when we use neural networks to add features later. So, all the neural nets in the forest may not be good enough. So, we need to take that into account when combining the output of each neural net. We will combine them taking the accuracy of the models measure over the oob samples for each of the bagged neural network.

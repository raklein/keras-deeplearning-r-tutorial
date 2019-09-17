####### Tuto Keras #######

# Tutorial created by Karlijn Willems: https://www.datacamp.com/community/tutorials/keras-r-deep-learning
# ALL CREDIT TO HER

# Also found this video helpful for understanding data structures: https://vimeo.com/130411487

# Script compiled by Olivier Dujols. Contact Rick Klein raklein22@gmail.com with any issues.

###install and load required packages:
# NOTE: You may have to close the R project and install these packages in a fresh R window.

#devtools::install_github("rstudio/keras")
library(keras)

#devtools::install_github("rstudio/tensorflow")
library(tensorflow)

# Run this line the first time to install tensorflow
#install_tensorflow()

#install.packages("corrplot")
library(corrplot)

### Other built-in datasets you can use
# Read in MNIST data
#mnist <- dataset_mnist()
# Read in CIFAR10 data
#cifar10 <- dataset_cifar10()
# Read in IMDB data
#imdb <- dataset_imdb()

### Or make your own data:
#data <- matrix(rexp(1000*784), nrow = 1000, ncol = 784)
# Make dummy target values for your dummy data
#labels <- matrix(round(runif(1000*10, min = 0, max = 9)), nrow = 1000, ncol = 10)

### But by default we'll use the iris dataset:
# Read in `iris` data
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE) 
# Return the first part of `iris`
head(iris)
# Inspect the structure
str(iris)
# Obtain the dimensions
dim(iris)

# rename IRIS' variables :
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

plot(iris$Petal.Length, 
     iris$Petal.Width, 
     pch=21, bg=c("red","green3","blue")[unclass(iris$Species)], 
     xlab="Petal Length", 
     ylab="Petal Width")

# Overall correlation between `Petal.Length` and `Petal.Width` 
cor(iris$Petal.Length, iris$Petal.Width)
# Store the overall correlation in `M`
M <- cor(iris[,1:4])
# Plot the correlation plot with `M`
corrplot(M, method="circle")

#### Data Preprocessing : 
# Pull up a summary of `iris`
summary(iris)
# Inspect the structure of `iris`
str(iris)

## Normalizing Your Data With A User Defined Function (UDF)
# Build your own `normalize()` function
#normalize <- function(x) {
 # num <- x - min(x)
 # denom <- max(x) - min(x)
 # return (num/denom)
#}
# Normalize the `iris` data
#iris_norm <- as.data.frame(lapply(iris[1:4], normalize))
# visualize the normalization
#hist(iris_norm$Sepal.Length)
#hist(iris$Sepal.Length)


# change for numerical vector
#A numerical data frame is alright, but you’ll need to convert the data to an array or a matrix 
#if you want to make use of the keras package.
iris[,5] <- as.numeric(iris[,5]) -1
# Turn `iris` into a matrix
iris <- as.matrix(iris)
# Set iris `dimnames` to `NULL`
dimnames(iris) <- NULL

# Normalize the `iris` data
iris_N <- normalize(iris[,1:4])

# Return the summary of `iris`
summary(iris_N)

### Training And Test Sets (with iris non normalized)
set.seed(23)
# Determine sample size
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))
# Split the `iris` data
iris.training <- iris[ind==1, 1:4]
iris.test <- iris[ind==2, 1:4]
# Split the class attribute
iris.trainingtarget <- iris[ind==1, 5]
iris.testtarget <- iris[ind==2, 5]

#### One Hot Encoding (OHE) :  
#It’s easier to work with numerical data, 
#and you have preprocessed the data and one hot encoded the values of the target variable: 
#a flower is either of type versicolor, setosa or virginica 
#and this is reflected with binary 1 and 0 values. (e.g.,: 100,010,001)
# One hot encode training target values
iris.trainLabels <- to_categorical(iris.trainingtarget)
# One hot encode test target values
iris.testLabels <- to_categorical(iris.testtarget)
# Print out the iris.testLabels to double check the result
print(iris.testLabels)

#### Constructing the Model ####
# multi-layer perceptron & relu activation function
# Initialize a sequential model
model <- keras_model_sequential() 

# Add layers to the model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')

#The first layer, which contains 8 hidden notes, on the other hand, has an input_shape of 4. This is because your training data iris.training has 4 columns.

## Inspect the model
# Print a summary of a model
summary(model)
# Get model configuration
get_config(model)
# Get layer configuration
get_layer(model, index = 1)
# List the model's layers
model$layers
# List the input tensors
model$inputs
# List the output tensors
model$outputs

## Compile the model
model %>% compile(
  loss = 'categorical_crossentropy', #use binary_crossentropy for a binary class classification pb.
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model 
model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 200, 
  batch_size = 5, 
  validation_split = 0.2
)
# Store the fitting history in `history` 
history <- model %>% fit(
  iris.training, 
  iris.trainLabels, 
  epochs = 200,
  batch_size = 5, 
  validation_split = 0.2
)

# Plot the history
plot(history)

# Plot the model loss of the training data
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
# Plot the model loss of the test data
lines(history$metrics$val_loss, col="green")
# Add legend
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))
# Plot the accuracy of the training data 
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
# Plot the accuracy of the validation data
lines(history$metrics$val_acc, col="green")
# Add Legend
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

#If your training data accuracy keeps improving while your validation data accuracy gets worse, you are probably overfitting: your model starts to just memorize the data instead of learning from it.
#If the trend for accuracy on both datasets is still rising for the last few epochs, you can clearly see that the model has not yet over-learned the training dataset.

### Predict Labels of New Data
# Predict the classes for the test data
classes <- model %>% predict_classes(iris.test, batch_size = 128)
# Confusion matrix
table(iris.testtarget, classes)

### Evaluating Your Model
# Evaluate on test data and labels
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)
# Print the score
print(score)

### Fine-tuning Your Model

  ### 1/ Adding layers

# Initialize the sequential model
model <- keras_model_sequential() 

# Add layers to model
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 5, activation = 'relu') %>% 
  layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model to the data
model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)

# Save the training history in history
history <- model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5,
  validation_split = 0.2
)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

    ### 2/ Hidden Units

# Initialize a sequential model
model <- keras_model_sequential() 

# Add layers to the model
model %>% 
  layer_dense(units = 28, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

# Fit the model to the data
model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the score
print(score)

# Save the training history in the history variable
history <- model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

      ### 3a/ Optimization Parameters

# Initialize a sequential model
model <- keras_model_sequential() 

# Build up your model by adding layers to it
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 3, activation = 'softmax')

# Define an optimizer
sgd <- optimizer_sgd(lr = 0.01)

# Use the optimizer to compile the model
model %>% compile(optimizer=sgd, 
                  loss='categorical_crossentropy', 
                  metrics='accuracy')

# Fit the model to the training data
model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(iris.test, iris.testLabels, batch_size = 128)

# Print the loss and accuracy metrics
print(score)

### 3b/ change the learning rate 

# Define an optimizer
sgd <- optimizer_sgd(lr = 0.01)

# Compile the model
model %>% compile(optimizer=sgd, 
                  loss='categorical_crossentropy', 
                  metrics='accuracy')

# Fit the model to the training data
history <- model %>% fit(
  iris.training, iris.trainLabels, 
  epochs = 200, batch_size = 5, 
  validation_split = 0.2
)

# Plot the model loss
plot(history$metrics$loss, main="Model Loss", xlab = "epoch", ylab="loss", col="blue", type="l")
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

# Plot the model accuracy
plot(history$metrics$acc, main="Model Accuracy", xlab = "epoch", ylab="accuracy", col="blue", type="l")
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

#### Saving, Loading or Exporting Your Model #####

save_model_hdf5(model, "my_model.h5") #save
model <- load_model_hdf5("my_model.h5") #load

save_model_weights_hdf5(model, "my_model_weights.h5") # save model w
load_model_weights_hdf5(model, "my_model_weights.h5") # load model w

#  export your model configuration to JSON or YAML. 
json_string <- model_to_json(model)
model <- model_from_json(json_string)
#load the configurations back into your workspace,
yaml_string <- model_to_yaml(model)
model <- model_from_yaml(yaml_string)

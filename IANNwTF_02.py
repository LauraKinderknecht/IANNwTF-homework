##2. MNIST classification ##

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

#2.1 Loading the MNIST dataset
( train_dataset , test_dataset ) , ds_info = tfds.load( "mnist" , split =[ "train" ,"test"] , as_supervised = True , with_info = True )
"""
print(ds_info)
How many training/test images are there? 60000/ 10000
What's the image shape? (28,28,1)
What range are pixel values in? data type = uint8, range =

tfds.show_examples(train_dataset, ds_info)
"""

#2.2 Setting up the data pipeline
def preprocessing_data(mnist):
    #flatten images to 28x28
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    #change datatype of images
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32),target))
    #normalize image values
    mnist = mnist.map(lambda img, target:((img/128.)-1.,target))
    #one-hot encode labels
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth = 10)))
    #cache progress
    mnist = mnist.cache()
    #shuffle, batch, prefetch
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(32)
    mnist = mnist.prefetch(20)
    return mnist

#apply preprocessing
train_ds = train_dataset.apply(preprocessing_data)
test_ds = test_dataset.apply(preprocessing_data)


##2.3 Building a deep neural network with TensorFlow
#create empty network
model = tf.keras.models.Sequential([

    #add input layer that flattens input shape 28x28 to 783 units in a flat line
    #tf.keras.layers.Flatten(input_shape=(28,28)),

    #add 2 hidden layers with 256 units each and relu activation function
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),

    #add output layer with 10 units (one for each possible answer (0-10))
    #softmax activation function gives probability for every possible output
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)])

#see model summary
#model.summary()


##2.4 Training the Network
#Hyperparameters
num_epochs = 10
lr = 0.1

#initialize loss function
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
#initialize optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=lr) 

#initializations for visualization
train_losses = []
test_losses = []
test_accuracies = []

#training step function
def train_step(model, input, target, loss_func, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_func(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

#test model function
def test(model, test_data, loss_function):

    test_accuracy_aggr = []
    test_loss_aggr = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggr.append(sample_test_loss.numpy())
        test_accuracy_aggr.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggr)
    test_accuracy = tf.reduce_mean(test_accuracy_aggr)

    return test_loss, test_accuracy


#training loop function
def training_loop(num_epochs, model, train_ds, test_ds, loss, optimizer, train_losses, test_losses, test_accuracies):
    for epoch in range(num_epochs):
        print(f"Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}")

        epoch_loss_aggr = []
        for input, target in train_ds:
            train_loss = train_step(model, input, target, loss, optimizer)
            epoch_loss_aggr.append(train_loss)
            
        train_losses.append(tf.reduce_mean(epoch_loss_aggr))

        test_loss, test_accuracy = test(model, test_ds, loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    
    return train_losses, test_losses,  test_accuracies

#test
test_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check performance
train_loss, _ = test(model, train_ds, cross_entropy_loss)
train_losses.append(train_loss)

#train the model
train_losses, test_losses, test_accuracies  = training_loop(num_epochs, model, train_ds, test_ds, cross_entropy_loss, optimizer, train_losses, test_losses, test_accuracies)


##2.5 Visualization
def visualization(train_losses , test_losses ,test_accuracies ):
    plt.figure ()
    line1 , = plt.plot( train_losses , "b-")
    line2 , = plt.plot( test_losses , "r-")
    line3 , = plt.plot ( test_accuracies , "r:")
    plt.xlabel("Training steps")
    plt.ylabel("Loss / Accuracy")
    plt.legend(( line1 , line2 , line3 ), ("training loss", "test loss", "test accuracy"))
    plt.show()

visualization(train_losses, test_losses, test_accuracies)
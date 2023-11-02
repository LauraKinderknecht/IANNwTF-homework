import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

#extract into (input, target) tuples
data = digits.data
target = digits.target
data_tuples = list(zip(data, target))

#plot first 10 data images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

for i, (image, t) in enumerate(data_tuples[:10]):
    ax = axes[i // 5, i % 5]
    ax.imshow(image.reshape(8, 8), cmap='gray')
    ax.axis('off')
#plt.show()

#images have the shape (64)
print("Image shapes: ",np.shape(data))

#change type into float32 and reshape with maximum pixel value
data = data.astype(np.float32)
data = data / 16.0
print("Data type: ", type(data[0][0]))

targets = []
#onehot encoding target
targets = []
for j in target:
    one_hot_target = np.zeros(10) 
    one_hot_target[j] = 1.0  
    targets.append(one_hot_target)
target = targets
print("First target with one-hot encoding: ", targets[0])

data_tuples = list(zip(data, target))



##### first generator function does not work! 
"""def generator(data_tuples, minibatchsize):
    random.shuffle(data_tuples)
    print(type(data_tuples))
    data_batch = []
    target_batch = []

    for i in range(0, minibatchsize+1):
        data_batch = np.append(data_batch, [tuple[0][:64] for tuple in data_tuples[i]])
        target_batch = np.append(target_batch,[tuple[1][:10] for tuple in data_tuples[i]])

    return data_batch, target_batch


def generator2(data_tuples, minibatch_size):
    random.shuffle(data_tuples)
    num_samples = len(data_tuples)
    num_minibatches = num_samples // minibatch_size # number of minibatches dependend on minibatch_size

    for i in range(num_minibatches):
        start_idx = i * minibatch_size
        end_idx = (i + 1) * minibatch_size
        minibatch_data = [data for data, target in data_tuples[start_idx:end_idx]]
        minibatch_target = [target for data, target in data_tuples[start_idx:end_idx]]

    return np.array(minibatch_data), np.array(minibatch_target)

print(generator2(data_tuples, 5 ))"""


import time

import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score



"""

DIMENSIONALITY ANALISYS

N: dataset dimension
d: input dimension
L: number of neurons
m: output dimension (numbers of classes)

X = N*d
Y = N*m
W = d*L
b = 1*L

"""

# Generate binary classification dataset
def classification_data(n_samples=10000, n_features=8, test_size=0.2, random_state=42):
    # Generate dataset
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_classes=2, 
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Data standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Conversion in PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# Bias and weights random inizialization
def random_weights(input_dimension, n_neuron):
    W = torch.randn(input_dimension, n_neuron)  
    b = torch.randn(1, n_neuron)
    return W, b

# Sigmoid function
def sigmoid(X, W, b):
    z = torch.matmul(X, W) + b
    return torch.sigmoid(z)
    e = torch.exp(-torch.matmul(X, W) + b)
    H = torch.div(1, torch.add(1, e))
    return H

def relu(X, W, b):
    return torch.relu(torch.matmul(X, W) + b)

# Training 
def training(n_neuron, X_train, Y_train, W, b, loss='sigmoid', learning_rate=0.01, epochs=500, silent=False):
    # Calculate H one time because W and b are fixed
    if loss == 'relu':
        H = relu(X_train, W, b)
    else:
        H = sigmoid(X_train, W, b)

    # Inizialize beta as a trainable parameters, the dimensions are (n_neuron, num_classes)
    num_classes = torch.unique(Y_train).shape[0]

    beta = torch.randn(n_neuron, num_classes, requires_grad=True)

    # Define Cross Entropy Loss and Adam
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([beta], lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()  # Reset the gradients

        # Compute the forward pass
        logits = torch.matmul(H, beta)  # The forward pass in ELM is obtained through H * beta

        # Compute the loss
        loss = criterion(logits, Y_train)

        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)  # Extract the class with max logit
        accuracy = (predictions == Y_train).float().mean()  

        # Backpropagation and optimazer step
        loss.backward()
        optimizer.step()

        if not silent and (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%')

    return beta  # Return trained parameters

# Evaluate model
def test_model(X_test, Y_test, W, b, beta, silent=False):
    # Compute the forward pass
    H_test = sigmoid(X_test, W, b)
    logits = torch.matmul(H_test, beta)

    # Extract the class with max logit
    predictions = torch.argmax(logits, dim=1)
    
    # Compute accuracy
    accuracy = (predictions == Y_test).float().mean()

    #compute f1
    f1 = f1_score(Y_test, predictions, average='weighted')

    if not silent:
        print(f'Accuracy sui dati di test: {accuracy.item() * 100:.2f}%')
        print(f'F1 score: {f1}')
    return accuracy, f1

# ----------------------- PLOT -----------------------

def initialize_heatmap_data(n_neuron_len, lmda_len):
    return np.zeros((len(n_neuron_len), len(lmda_len)))

def update_heatmap_data(heatmap_data, mean_loss, n_neuron_idx, lmda_idx):
    heatmap_data[n_neuron_idx, lmda_idx] = mean_loss

def plot_heatmap(heatmap_data, n_neuron_len, lmda_len, name, type ):
    n_neuron_len = n_neuron_len.tolist()
    lmda_len = lmda_len.tolist()

    lmda_len_str = [f"{x:.1e}" for x in lmda_len]
    if type == 'c':
        max_val = np.max(heatmap_data)
        max_pos = np.unravel_index(np.argmax(heatmap_data), heatmap_data.shape)
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, annot=False, fmt=".2f", xticklabels=lmda_len_str, yticklabels=n_neuron_len,
                         cmap="viridis")

        ax.add_patch(plt.Rectangle((max_pos[1], max_pos[0]), 1, 1, fill=False, edgecolor='blue', lw=3))
        ax.text(max_pos[1] + 0.5, max_pos[0] + 0.5, f"Max: {max_val:.2f}", color='blue', ha="center", va="center",
                fontsize=12, fontweight='bold')
        plt.xlabel("Learning Rate")
        plt.ylabel("Number of Neurons")
        plt.title(str("Accuracy Heatmap " + name))
    else:
        min_val = np.min(heatmap_data)
        min_pos = np.unravel_index(np.argmin(heatmap_data), heatmap_data.shape)
        plt.figure(figsize=(10, 8))

        ax = sns.heatmap(heatmap_data, annot=False, fmt=".2f", xticklabels=lmda_len_str, yticklabels=n_neuron_len, cmap="viridis")


        ax.add_patch(plt.Rectangle((min_pos[1], min_pos[0]), 1, 1, fill=False, edgecolor='red', lw=3))
        ax.text(min_pos[1] + 0.5, min_pos[0] + 0.5, f"Min: {min_val:.2f}", color='red', ha="center", va="center", fontsize=12, fontweight='bold')
        plt.xlabel("Lambda")
        plt.ylabel("Number of Neurons")
        plt.title(str("Loss Heatmap " + name))
    
    plt.show()

# ----------------------- FINISH PLOT -----------------------
if __name__ == 'main':
# Dataset parameters
    input_dimension = 10
    num_classes = 5

    # Model parameters configuration
    n_neuron_len = torch.tensor([10,70,300,1000,2500])
    # n_neuron_len = torch.tensor([10,20,30,50,70,80,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500])
    learning_rate_len = torch.tensor([1.0e-1, 1.0e-2,1.0e-3, 2.0e-4])
    # learning_rate_len = torch.tensor([1.0e-1,2.0e-1,1.0e-2,2.0e-2,1.0e-3,2.0e-3,1.0e-4,2.0e-4])

    # Generate dataset
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = classification_data(n_features=input_dimension)

    # Matrix heatmap inizialization
    heatmap_data = initialize_heatmap_data(n_neuron_len, learning_rate_len)

    i = 0
    # For each configuration
    for loss_f in ['sigmoid', 'relu']:
        for n_neuron in n_neuron_len:
            for learning_rate in learning_rate_len:

                print("\n\n---------- Test number " + str(i + 1) + " ----------")
                print("loss_f:", loss_f)
                print("n_neurons: ", n_neuron.item())
                print("learning_rate: ", learning_rate.item())
                i = i + 1

                # Inizialization of weights W and b
                W, b = random_weights(input_dimension, n_neuron)

                # Training model
                accuracies=[]
                iterations = 5
                for _ in range(iterations):
                    start = time.time()
                    beta = training(n_neuron, X_train_tensor, y_train_tensor, W, b, loss_f, num_classes, learning_rate)
                    print("Time: ", time.time() - start)
                    # Test model
                    accuracy = test_model(X_test_tensor, y_test_tensor, W, b, beta)
                    accuracies.append(accuracy)

                # ----------- HEATMAP -----------
                n_neuron_idx = n_neuron_len.tolist().index(n_neuron)
                lmda_idx = learning_rate_len.tolist().index(learning_rate)
                update_heatmap_data(heatmap_data, np.mean(accuracies), n_neuron_idx, lmda_idx)
                # ----------- FINISH HEATMAP -----------

        print("Test ended!")
        plot_heatmap(heatmap_data, n_neuron_len, learning_rate_len, loss_f)
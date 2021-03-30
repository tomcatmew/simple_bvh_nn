import torch
import torch.nn as nn
import numpy as np
from data_bvh import BvhDataset

torch.set_printoptions(precision=6,sci_mode =False)
# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 96
hidden_size = 500
output_size = 96
num_epochs = 5
batch_size = 20
learning_rate = 0.001

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data',
#                                           train=False,
#                                           transform=transforms.ToTensor())

train_dataset = BvhDataset("LocomotionFlat01_000.bvh")
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)

# count = 0
# for i, (c_frame, c_next) in enumerate(train_loader):
#     print(i)
#     print(c_frame.shape)
#     print(c_next.shape)
#     count += 1
# print(count)



# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.relu2 = nn.ReLU()
        #self.fc3 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.relu2(out)
        #out = self.fc3(out)
        return out


model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
model.train()

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (current_frame, next_frame) in enumerate(train_loader):
        # Move tensors to the configured device
        current_frame = current_frame.to(device)
        next_frame = next_frame.to(device)

        # Forward pass
        outputs = model(current_frame)
        loss = criterion(outputs, next_frame)

        # Backprpagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # every 100 steps give a record
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


def my_mean_square_difference(t_size,tensor1,tensor2):
    result = 0
    for i in range(t_size):
        result += (tensor1[i] - tensor2[i])**2
    return result

# Test the model
# In the test phase, don't need to compute gradients (for memory efficiency)
model.eval()
with torch.no_grad():
    print("test prediction : ")
    test_frame_pair = np.loadtxt("test.bvh",dtype=np.float32)
    #print(test_frame_pair[0])
    input = torch.tensor(test_frame_pair[0])
    output = model(input)
    result = my_mean_square_difference(input_size,output,test_frame_pair[1])
    n_digits = 6
    output = torch.round(output * 10**n_digits) / (10**n_digits)
    #print(type(output[1]))

    print("result_pos : {}".format(output))
    print("result : {}".format(result))

torch.save(model.state_dict(), 'my_test.ckpt')
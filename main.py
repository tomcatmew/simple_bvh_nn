import torch
import torch.nn as nn
import numpy as np
from data_bvh import BvhDataset

torch.set_printoptions(precision=6, sci_mode=False)
# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters 
input_size = 96
hidden_size = 512
output_size = 96
num_epochs = 10
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

train_dataset0 = BvhDataset("LocomotionFlat01_000.bvh")
train_dataset1 = BvhDataset("LocomotionFlat02_000.bvh")
train_dataset2 = BvhDataset("LocomotionFlat02_001.bvh")
train_dataset3 = BvhDataset("LocomotionFlat03_000.bvh")
train_dataset4 = BvhDataset("LocomotionFlat04_000.bvh")
train_dataset5 = BvhDataset("LocomotionFlat05_000.bvh")
train_dataset6 = BvhDataset("LocomotionFlat06_000.bvh")
# Data loader
train_loader0 = torch.utils.data.DataLoader(dataset=train_dataset0, batch_size=batch_size, shuffle=True)
train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=batch_size, shuffle=True)
train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=batch_size, shuffle=True)
train_loader3 = torch.utils.data.DataLoader(dataset=train_dataset3, batch_size=batch_size, shuffle=True)
train_loader4 = torch.utils.data.DataLoader(dataset=train_dataset4, batch_size=batch_size, shuffle=True)
train_loader5 = torch.utils.data.DataLoader(dataset=train_dataset5, batch_size=batch_size, shuffle=True)
train_loader6 = torch.utils.data.DataLoader(dataset=train_dataset6, batch_size=batch_size, shuffle=True)


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
class MyNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


model = MyNeuralNet(input_size, hidden_size, output_size).to(device)
# fc1_weight = model.state_dict()['fc1.weight']
# print(fc1_weight)
# fc1_np = fc1_weight.numpy()
# print(fc1_np.shape)

# my_file = open('fc1.txt', 'ab')
# np.savetxt('fc2.txt', fc1_weight.numpy())
# my_file.close()
# torch.save(fc1_weight,'t.csv')
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = 0

model.train()

total_step = len(train_loader0)
for epoch in range(num_epochs):
    for i, (current_frame, next_frame) in enumerate(train_loader0):

        # for i, (current_frame, next_frame) in enumerate(train_loader0):
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

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

scripted_module = torch.jit.script(model)
torch.jit.save(scripted_module, 'bvhmodulenew.pt')


# torch.jit.load('bvhmodule2.pt')


def my_mean_square_difference(t_size, tensor1, tensor2):
    result = 0
    for i in range(t_size):
        result += (tensor1[i] - tensor2[i]) ** 2
    return result


# Test the model
# In the test phase, don't need to compute gradients (for memory efficiency)

model.eval()
with torch.no_grad():
    print("test prediction : ")
    test_frame_pair = np.loadtxt("test.bvh", dtype=np.float32)
    # print(test_frame_pair[0])
    input_bvh = torch.tensor(test_frame_pair[0])
    # print(input_bvh.shape)
    output = model(input_bvh)
    print(output)
    i = 0
    while i < 30:
        input_bvh = output
        output = model(input_bvh)
        print(output)
        i = i + 1

    # result = my_mean_square_difference(input_size, output, test_frame_pair[0])
    #     n_digits = 6
    #     output = torch.round(output * 10**n_digits) / (10**n_digits)
    #     print(type(output[1]))
    #
    #     print("result_pos : {}".format(output))
    # rint("result : {}".format(result))

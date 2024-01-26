from utility import *

class DecisionalNN(torch.nn.Module):
    def __init__(self, _input_size, _output_size, _device='cuda'):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(_input_size, 64, 3)
        self.pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(64, 128, 3)
        self.pool2 = torch.nn.MaxPool2d(3)

        self.dense1 = torch.nn.Linear(128, 64)
        self.dense2 = torch.nn.Linear(64, _output_size)

        self.to(_device)

    def forward(self, _x):
        #print (f'init x: {_x.shape}')
        x = self.conv1(_x)
        #print (f'conv1 x: {x.shape}')
        x = self.pool1(x)
        #print (f'pool1 x: {x.shape}')
        x = self.conv2(x)
        #print (f'conv2 x: {x.shape}')
        x = self.pool2(x)
        #print (f'pool2 x: {x.shape}')

        x = torch.flatten(x)
        #print (f'flat x: {x.shape}')
        x = self.dense1(x)
        #print (f'dense1 x: {x.shape}')
        x = torch.relu(x)
        x = self.dense2(x)
        #print (f'dense2 x: {x.shape}')
        x = torch.relu(x)

        # * normalize
        # x -= x.min() 
        # x /= x.max()
        n = torch.norm(x)
        if n == 0.0:
            n = 1e-10
        x /= n

        return x
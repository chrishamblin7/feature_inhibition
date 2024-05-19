#simple model from 'Toy Models of Superposition'

import torch
import torch.nn as nn
import torch.optim as optim

device = 'cpu'

# Hyperparameters
num_trys = 1000 #number of times to run the model
n = 6  # Number of input features
m = 8 # Number of features in hidden layers
S = .05 # Probability of non-zero input
p = .5 #probability of 0 or 1 for input thats 'on'
batch_size = 600
num_epochs = 20000
num_test_epochs = 5000
learning_rate = 0.001
i_curve = .9

out_name = 'xor_%s_%s'%(str(n),str(m))

#MODEL

class ScaledXORNet(nn.Module):
    def __init__(self, n, m, xor_model=None):
        super(ScaledXORNet, self).__init__()
        self.n = n
        # Expand the input layer to handle 2*n inputs and create 2*n hidden units
        self.hidden = nn.Linear(2*n, m)
        # The output layer now produces n outputs from the 2*n hidden units
        self.output = nn.Linear(m, n)
        self.relu = nn.ReLU()
        
        # Initialize weights and biases using the provided XOR model
        if xor_model is not None:
            self.init_weights(xor_model)
    
    def init_weights(self, xor_model):
        # Extract weights and biases from the provided XOR model
        xor_hidden_weights = xor_model.hidden.weight.data
        xor_hidden_bias = xor_model.hidden.bias.data
        xor_output_weights = xor_model.output.weight.data
        xor_output_bias = xor_model.output.bias.data
        
        # Initialize hidden layer weights and biases
        hidden_weights = torch.zeros((2*n, 2*n))
        hidden_bias = torch.zeros(2*n)
        for i in range(n):
            # Use the XOR model's weights and biases for each pair in the hidden layer
            hidden_weights[2*i:2*(i+1), 2*i:2*(i+1)] = xor_hidden_weights
            hidden_bias[2*i:2*(i+1)] = xor_hidden_bias
        
        self.hidden.weight = nn.Parameter(hidden_weights)
        self.hidden.bias = nn.Parameter(hidden_bias)
        
        # Initialize output layer weights and biases
        output_weights = torch.zeros((n, 2*n))
        output_bias = torch.zeros(n)
        for i in range(n):
            # Use the XOR model's output weights and biases, adapted for the scaled model
            output_weights[i, 2*i:2*(i+1)] = xor_output_weights
            # Adjust for the number of outputs if needed
            if xor_output_bias.numel() == n:
                output_bias[i] = xor_output_bias[i]
            else:
                output_bias[i] = xor_output_bias[0]
        
        self.output.weight = nn.Parameter(output_weights)
        self.output.bias = nn.Parameter(output_bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.relu(x)
        return x
    
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.relu(x)
        return x
    



# Function to Generate Synthetic Data
def generate_xor_data(batch_size, n, S, p=.5, device='cpu'):
    x = torch.bernoulli(torch.full((batch_size, 2*n), p, dtype=torch.float)).to(device)
    mask = torch.rand(batch_size, n) < S
    interleaved_mask = mask.unsqueeze(2).repeat(1, 1, 2).flatten(start_dim=1)
    x = x * interleaved_mask.to(device)
    x_pairs = x.reshape(batch_size, n, 2)
    # Compute XOR by adding the elements in the last dimension and taking modulo 2
    y = torch.sum(x_pairs, dim=2) % 2
    return x, y


# Custom Loss Function
def feature_importance_loss(y_pred, y_true, n, curve = .8):
    feature_importances = torch.tensor([curve ** i for i in range(n)]).to(y_pred.device)
    diff = y_pred - y_true
    weighted_diff = diff ** 2 * feature_importances
    loss = weighted_diff.sum(dim=1).mean()
    return loss



top_model = None
best_loss = 0

for try_i in range(num_trys):
    print('model '+str(try_i))
    # Model, and Optimizer
    model = ScaledXORNet(n, m).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    #losses = []
    # Training Loop
    for epoch in range(num_epochs):
        x, y = generate_xor_data(batch_size, n, S, p, device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = feature_importance_loss(outputs, y, n, curve = i_curve)
        loss.backward()
        optimizer.step()

#         if (epoch) % 10 == 0:
#             losses.append(loss.item())
        if (epoch) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('testing')
    total_loss = 0
    for epoch in range(num_test_epochs):
        x, y = generate_xor_data(batch_size, n, S, p)
        outputs = model(x)
        loss = feature_importance_loss(outputs, y, n, curve = i_curve)
        total_loss +=loss
        
    if (top_model is None) or (total_loss< best_loss):
        print('TOP MODEL')
        top_model = model
        best_loss = total_loss
        torch.save(top_model.state_dict(),'models/'+out_name+'.pt')
        meta_data = {
                    'n':n,
                    'm':m,
                    'S': S,
                    'batch_size':batch_size,
                    'num_epochs':num_epochs,
                    'num_test_epochs':num_test_epochs,        
                    'learning_rate':learning_rate,
                    'i_curve':i_curve
                    }
        torch.save(meta_data,'models/'+out_name+'.json')
        


torch.save(top_model.state_dict(),'models/'+out_name+'.pt')
meta_data = {
            'n':n,
            'm':m,
            'S': S,
            'batch_size':batch_size,
            'num_epochs':num_epochs,
            'num_test_epochs':num_test_epochs,        
            'learning_rate':learning_rate,
            'i_curve':i_curve
            }
torch.save(meta_data,'models/'+out_name+'.json')

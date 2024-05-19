#simple model from 'Toy Models of Superposition'

import torch
import torch.nn as nn
import torch.optim as optim

device = 'cpu'
out_name = 'abs_6_8'
# Hyperparameters
num_trys = 1000 #number of times to run the model
n = 6  # Number of input features
m = 8 # Number of features in hidden layers
S = .01 # Probability of non-zero input
batch_size = 400
num_epochs = 30000
num_test_epochs = 5000
learning_rate = 0.001
i_curve = .9



# Model Definition
class AbsoluteValueModel(nn.Module):
    def __init__(self, n, m):
        super(AbsoluteValueModel, self).__init__()
        self.linear1 = nn.Linear(n, m, bias=False)
        self.linear2 = nn.Linear(m, n, bias=True)
        self.rl1 = nn.ReLU(inplace = False)
        self.rl2 = nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.rl1(x)
        x = self.linear2(x)
        x = self.rl2(x)
        return x

# Function to Generate Synthetic Data
def generate_data(batch_size, n, S, device='cpu'):
    x = torch.zeros(batch_size, n).uniform_(-1, 1).to(device)
    mask = torch.rand(batch_size, n) < S
    x = x * mask.to(device)
    y = x.abs()
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
    model = AbsoluteValueModel(n, m).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    #losses = []
    # Training Loop
    for epoch in range(num_epochs):
        x, y = generate_data(batch_size, n, S, device)
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
        x, y = generate_data(batch_size, n, S)
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

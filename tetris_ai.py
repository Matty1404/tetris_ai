import random
import torch
import torch.nn as nn
import torch.optim as optim


class TetrisAI():
  def choose_move(self, inputs = None):
    pass



class TetrisNeuralNetwork(nn.Module):
  def __init__(self, input_size, h1, h2, h3, h4, h5, h6, h7, h8, output_size, genome):
    super(TetrisNeuralNetwork, self).__init__()
    
    # Define the layers of the neural network
    self.input_layer = nn.Linear(input_size, h1)
    self.h1 = nn.Linear(h1, h2)
    self.h2 = nn.Linear(h2, h3)
    self.h3 = nn.Linear(h3, h4)
    self.h4 = nn.Linear(h4, h5)
    self.h5 = nn.Linear(h5, h6)
    self.h6 = nn.Linear(h6, h7)
    self.h7 = nn.Linear(h7, h8)
    self.output_layer = nn.Linear(h8, output_size)
    
    self.hidden_layers = [self.h1, self.h2, self.h3, self.h4, self.h5, self.h6, self.h7]
    
    # Initialize the weights with values from the genome
    self.initialize_weights(genome)
    
    self.faltten = nn.Flatten()
    
    
    
  
  def forward(self, x):
    
    x = torch.relu(self.input_layer(x))
    for layer in self.hidden_layers:
        x = torch.relu(layer(x))
    x = torch.softmax(self.output_layer(x), dim=0)
    return x

  
  
  def initialize_weights(self, genome):
    # Check if the genome has the correct number of weights
    total_params = sum(p.numel() for p in self.parameters())
    if len(genome) != total_params:
        raise ValueError("Genome size does not match the network's total parameters.")
    
    # Counter for tracking the position in the genome
    genome_index = 0
    
    # Set the weights for input layer
    layer_weight = torch.FloatTensor(genome[genome_index:genome_index+self.input_layer.weight.numel()])
    self.input_layer.weight.data = layer_weight.view(self.input_layer.weight.size())
    genome_index += layer_weight.numel()
    layer_bias = torch.FloatTensor(genome[genome_index:genome_index+self.input_layer.bias.numel()])
    self.input_layer.bias.data = layer_bias.view(self.input_layer.bias.size())
    genome_index += layer_bias.numel()
    
    # Set the weights for h1
    layer_weight = torch.FloatTensor(genome[genome_index:genome_index+self.h1.weight.numel()])
    self.h1.weight.data = layer_weight.view(self.h1.weight.size())
    genome_index += layer_weight.numel()
    layer_bias = torch.FloatTensor(genome[genome_index:genome_index+self.h1.bias.numel()])
    self.h1.bias.data = layer_bias.view(self.h1.bias.size())
    genome_index += layer_bias.numel()
    
    # Set the weights for h2
    layer_weight = torch.FloatTensor(genome[genome_index:genome_index+self.h2.weight.numel()])
    self.h2.weight.data = layer_weight.view(self.h2.weight.size())
    genome_index += layer_weight.numel()
    layer_bias = torch.FloatTensor(genome[genome_index:genome_index+self.h2.bias.numel()])
    self.h2.bias.data = layer_bias.view(self.h2.bias.size())
    genome_index += layer_bias.numel()
    
    # Set the weights for h3
    layer_weight = torch.FloatTensor(genome[genome_index:genome_index+self.h3.weight.numel()])
    self.h3.weight.data = layer_weight.view(self.h3.weight.size())
    genome_index += layer_weight.numel()
    layer_bias = torch.FloatTensor(genome[genome_index:genome_index+self.h3.bias.numel()])
    self.h3.bias.data = layer_bias.view(self.h3.bias.size())
    genome_index += layer_bias.numel()
    
    # Set the weights for h4
    layer_weight = torch.FloatTensor(genome[genome_index:genome_index+self.h4.weight.numel()])
    self.h4.weight.data = layer_weight.view(self.h4.weight.size())
    genome_index += layer_weight.numel()
    layer_bias = torch.FloatTensor(genome[genome_index:genome_index+self.h4.bias.numel()])
    self.h4.bias.data = layer_bias.view(self.h4.bias.size())
    genome_index += layer_bias.numel()
    
    # Set the weights for h5
    layer_weight = torch.FloatTensor(genome[genome_index:genome_index+self.h5.weight.numel()])
    self.h5.weight.data = layer_weight.view(self.h5.weight.size())
    genome_index += layer_weight.numel()
    layer_bias = torch.FloatTensor(genome[genome_index:genome_index+self.h5.bias.numel()])
    self.h5.bias.data = layer_bias.view(self.h5.bias.size())
    genome_index += layer_bias.numel()
    
    # Set the weights for h6
    layer_weight = torch.FloatTensor(genome[genome_index:genome_index+self.h6.weight.numel()])
    self.h6.weight.data = layer_weight.view(self.h6.weight.size())
    genome_index += layer_weight.numel()
    layer_bias = torch.FloatTensor(genome[genome_index:genome_index+self.h6.bias.numel()])
    self.h6.bias.data = layer_bias.view(self.h6.bias.size())
    genome_index += layer_bias.numel()
    
    # Set the weights for h7
    layer_weight = torch.FloatTensor(genome[genome_index:genome_index+self.h7.weight.numel()])
    self.h7.weight.data = layer_weight.view(self.h7.weight.size())
    genome_index += layer_weight.numel()
    layer_bias = torch.FloatTensor(genome[genome_index:genome_index+self.h7.bias.numel()])
    self.h7.bias.data = layer_bias.view(self.h7.bias.size())
    genome_index += layer_bias.numel()
    
    # Set the weights for output_layer
    layer_weight = torch.FloatTensor(genome[genome_index:genome_index+self.output_layer.weight.numel()])
    self.output_layer.weight.data = layer_weight.view(self.output_layer.weight.size())
    genome_index += layer_weight.numel()
    layer_bias = torch.FloatTensor(genome[genome_index:genome_index+self.output_layer.bias.numel()])
    self.output_layer.bias.data = layer_bias.view(self.output_layer.bias.size())
    genome_index += layer_bias.numel()



class Smart_AI(TetrisAI):
  
  
  def __init__(self, genome = None, genome_2 = None):
    self.genome = genome
    if genome is None:
      self.genome = [random.uniform(-1,1) for _ in range(699492)]
    elif genome_2 is not None:
      # cross over and mutate
      for i in range(len(genome)):
        if random.randint(0,1) == 1:
          self.genome[i] = genome_2[i]
        self.genome[i] += random.uniform(-0.15, 0.15)
    
    self.nn = TetrisNeuralNetwork(252, 256, 256, 512, 512, 256, 128, 64, 32, 4, genome = self.genome)
  
  
  
  def choose_move(self, inputs = None):
    input_list = inputs
    if len(input_list) != 252:
      raise ValueError("Not big enough inputs]")
    if inputs is None:
      input_list = [random.randint(0,1) for _ in range(252)]
    tlist = torch.Tensor(input_list)
    # print(self.nn.forward(tlist))
    selection = self.nn.forward(tlist).tolist()
    return selection.index(max(selection))



class RandomAI(TetrisAI): 
  def choose_move(self, inputs = None):
    return random.randint(0,3)
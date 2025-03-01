import torch
import torch.nn as nn

# Define the attention module
class Attention(nn.Module):
    def __init__(self, hidden_size):
        """
        Function to initialize an additive Attention layer

        Parameters:
            hidden_size: Dimension of the hidden size
        """
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # Feed forward layers to create the weight vectors
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        # Initializing weights
        self.initialize_weights()
    def forward(self, lstm_out):
        """
        Function to run forward pass with attention layer

        Parameters:
            lstm_out: Output from the lstm
        Returns:
            context_vector: Shape = (batch_size, hidden_size)
            attention_weights: Shape = (batch_size, seq_length, 1)
        """
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        # Calculate attention scores
        scores = self.Va(torch.tanh(self.Wa(lstm_out) + self.Ua(lstm_out)))
        attention_weights = torch.softmax(scores, dim=1)  # Shape: (batch_size, seq_length, 1)

        # Weighted sum of the LSTM outputs
        context_vector = torch.bmm(attention_weights.permute(0, 2, 1), lstm_out)  # Shape: (batch_size, 1, hidden_size)
        context_vector = context_vector.squeeze(1)  # Shape: (batch_size, hidden_size)

        return context_vector, attention_weights

    def initialize_weights(self):
        """
        Function to initialize the weights of the Attention Layers
        Customize as per optimizer used
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param.data)  # Kaiming uniform initialization
            elif 'bias' in name:
                nn.init.zeros_(param.data)  # Initialize biases to zero


# Define the LSTM Model 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        Function to initialize the LSTM model
        Parameters
            input_size: The dimension of the input data
            hiiden_size: The dimension of the hidden layer
            num_layers: The number of hidden layers
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer for 1D output
        # Initializing weight
        self.initialize_weights()

    def forward(self, x):
        """
        Function to run forward pass
        Parameters
            x: Input
        Return 
            output: Tensor corresponding to the output
        """
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)  # Get the LSTM output
        context_vector, attention_weights = self.attention(lstm_out)  # Apply attention mechanism
        output = self.fc(context_vector)  # Shape: (batch_size, 1)
        return output

    def initialize_weights(self):
        """
        Function to initialize weights of the LSTM layers
        Customise as per the optimizer used
        """
        # Initialize weights for LSTM layers
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param.data)  # Kaiming uniform initialization
            elif 'bias' in name:
                nn.init.zeros_(param.data)  # Initialize biases to zero


import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import CNN
from models.fcnn import FCNN
from utils.data_utils import load_data
from train import train_fcnn, train_cnn
from test import test_models

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001
    input_size = 784
    hidden_size = 500
    num_classes = 10

    # Load data
    train_loader, test_loader = load_data(batch_size)

    # Initialize models
    fcnn_model = FCNN(input_size, hidden_size, num_classes).to(device)
    cnn_model = CNN(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    fc_optimizer = optim.Adam(fcnn_model.parameters(), lr=learning_rate)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

    # Train models
    print("Training FCNN...")
    train_fcnn(fcnn_model, train_loader, criterion, fc_optimizer, num_epochs, device)
    print("Training CNN...")
    train_cnn(cnn_model, train_loader, criterion, cnn_optimizer, num_epochs, device)

    # Test models and get accuracy
    print("Testing models...")
    correct_fcnn, total_fcnn, correct_cnn, total_cnn = test_models(fcnn_model, cnn_model, test_loader, device)
    fcnn_accuracy = 100 * correct_fcnn / total_fcnn
    cnn_accuracy = 100 * correct_cnn / total_cnn
    print('FCNN Accuracy: {} %'.format(100 * correct_fcnn / total_fcnn))
    print('CNN Accuracy: {} %'.format(100 * correct_cnn / total_cnn))

    # Save the best performing model
    print("Saving the best performing model...")
    best_model = fcnn_model if fcnn_accuracy > cnn_accuracy else cnn_model
    torch.save(best_model.state_dict(), 'model/best_model.pth')
    print("Best performing model saved successfully.")

if __name__ == "__main__":
    main()
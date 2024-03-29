import torch

def test_models(fcnn_model, cnn_model, test_loader, device):
    fcnn_model.eval()
    cnn_model.eval()
    with torch.no_grad():
        correct_fcnn = 0
        total_fcnn = 0
        correct_cnn = 0
        total_cnn = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs_fcnn = fcnn_model(images)
            _, predicted_fcnn = torch.max(outputs_fcnn.data, 1)
            total_fcnn += labels.size(0)
            correct_fcnn += (predicted_fcnn == labels).sum().item()

            images = images.reshape(-1, 1, 28, 28).to(device)
            outputs_cnn = cnn_model(images)
            _, predicted_cnn = torch.max(outputs_cnn.data, 1)
            total_cnn += labels.size(0)
            correct_cnn += (predicted_cnn == labels).sum().item()

        return correct_fcnn, total_fcnn, correct_cnn, total_cnn
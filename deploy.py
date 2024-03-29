import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.cnn import CNN
from models.fcnn import FCNN
from train import train_fcnn, train_cnn
from test import test_models
from utils.data_utils import load_data
from main import main

def preprocess_image(image_path):
    # Preprocess the image as per your requirements
    # (This is just an example, you should modify it based on your actual preprocessing steps)
    image = cv2.imread(image_path)
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]
        
        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))
        
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        
        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)
    print("\n\n\n----------------Contoured Image--------------------")
    plt.imshow(image, cmap="gray")
    plt.show()
        
    return preprocessed_digits

def deploy(model_path, image_path, device):

    # Try loading the saved model as an FCNN
    if os.path.exists(model_path):
        try:
            model = FCNN(input_size=784, hidden_size=500, num_classes=10).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded the saved FCNN model from {model_path}")
        except:
            # If loading as FCNN fails, try loading as CNN
            try:
                model = CNN(num_classes=10).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded the saved CNN model from {model_path}")
            except:
                # If loading as CNN also fails, call main() to train and save the best model
                print("Model file could not be loaded. Training a new model...")
                main()
                # Load the saved model again after training
                model = FCNN(input_size=784, hidden_size=500, num_classes=10).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # If model file doesn't exist, call main() to train and save the best model
        print("Model file not found. Training a new model...")
        main()
        try:
            model = FCNN(input_size=784, hidden_size=500, num_classes=10).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded the saved FCNN model from {model_path}")
        except:
            # If loading as FCNN fails, try loading as CNN
            try:
                model = CNN(num_classes=10).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded the saved CNN model from {model_path}")
            except:
                # If loading as CNN also fails, call main() to train and save the best model
                print("Model file could not be loaded. Training a new model...")
                main()
                # Load the saved model again after training
                model = FCNN(input_size=784, hidden_size=500, num_classes=10).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()

    # Preprocess the image
    preprocessed_digits = preprocess_image(image_path)

    # Make predictions
    all_predictions = []
    for idx, digit in enumerate(preprocessed_digits):
        digit_tensor = torch.from_numpy(digit).unsqueeze(0).unsqueeze(0).float().to(device)
        prediction = model(digit_tensor)
        all_predictions.append((idx, prediction.argmax().item()))

        print("\n\n---------------------------------------\n\n")
        print("=========PREDICTION============ \n\n")
        plt.imshow(digit.reshape(28, 28), cmap="gray")
        plt.show()
        print("\n\nFinal Output: {}".format(prediction.argmax().item()))
        print("\n\n---------------------------------------\n\n")

    print("\n\n---------------Final Prediction-----------------\n\n")
    for idx, prediction in all_predictions:
        print("Digit {} : {}".format(idx, prediction))

    print("\n\n---------------------------------------\n\n")
    print("Classification completed.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model/best_model.pth"
    image_path = "test_image/test_image.jpeg"
    deploy(model_path, image_path, device)
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import matplotlib.pyplot as plt

# # Load your model (DenseNet121)
# model = models.densenet121(pretrained=True)

# # Modify the classifier to match the number of classes in your dataset (e.g., 11 classes)
# num_classes = 11
# model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# # Load the saved weights for the model
# model.load_state_dict(torch.load(r"C:\Final Year Project\Model Save\Densenet121_Fetal.pth", map_location=torch.device('cpu')))
# model.eval()  # Set to evaluation mode

# # Define the image transformation (based on model's expected input size)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resizing the image to 224x224
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same as ImageNet, adjust if needed
# ])

# # Function to classify the image
# def classify_image(image_path):
#     img = Image.open(image_path)
#     img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

#     with torch.no_grad():  # Inference without updating the model
#         outputs = model(img_tensor)
#         _, predicted = torch.max(outputs, 1)
#         class_id = predicted.item()

#     return class_id

# # Example usage
# image_path = r"C:\Final Year Project\Classification of fetal brain abnormalities.v1i.multiclass\test\anold-chiari-malformation-16e_aug_0_png_jpg.rf.a3346cf82c525c565127a92c33301a29.jpg"
# class_id = classify_image(image_path)

# # Define your class names list
# class_names = [
#     'anold chiari malformation', 'arachnoid cyst', 'cerebellah hypoplasia', 
#     'cisterna magna', 'colphocephaly', 'encephalocele', 'intracranial hemorrdge', 
#     'mild ventriculomegaly', 'moderate ventriculomegaly', 'polencephaly', 'severe ventriculomegaly'
# ]

# # Get the predicted class name
# predicted_class = class_names[class_id]

# # Print the predicted class
# print(f"Predicted Class: {predicted_class}")

# # Optionally, display the image
# img = Image.open(image_path)
# plt.imshow(img)
# plt.title(f"Predicted Class: {predicted_class}")
# plt.axis('off')
# plt.show()
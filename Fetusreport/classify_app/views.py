from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import torch
from torchvision import models, transforms
from django.http import HttpResponse
from PIL import Image

# Load the model globally (once when the server starts)
model = models.densenet121(weights='IMAGENET1K_V1')  # Load pre-trained model
num_classes = 11
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load(r"C:\Final Year Project\Model Save\Densenet121_Fetal.pth", map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = [
    'anold chiari malformation', 'arachnoid cyst', 'cerebellah hypoplasia',
    'cisterna magna', 'colphocephaly', 'encephalocele', 'intracranial hemorrdge',
    'mild ventriculomegaly', 'moderate ventriculomegaly', 'polencephaly', 'severe ventriculomegaly'
]

def classify_image(image_path):
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# View to handle file upload and classification
def classify(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        
        # Save the uploaded image to a temporary file
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        uploaded_image_path = fs.url(filename)
        
        # Run classification
        image_path = fs.url(filename)
        class_id = classify_image(image_path)
        predicted_class = class_names[class_id]
        
        # Check for "Invalid image" condition (adjust as necessary)
        if predicted_class not in class_names:
            predicted_class = "Invalid image, please upload a valid image"
        
        # Return the result to the template
        return render(request, './classify_app/upload.html', {'class': predicted_class, 'image_url': uploaded_image_path})

    return render(request, './classify_app/upload.html')


def meow(request):
    image_path = r"C:\Final Year Project\Classification of fetal brain abnormalities.v1i.multiclass\test\anold-chiari-malformation-16e_aug_0_png_jpg.rf.a3346cf82c525c565127a92c33301a29.jpg"
    class_id = classify_image(image_path)

    # Define your class names list
    class_names = [
        'anold chiari malformation', 'arachnoid cyst', 'cerebellah hypoplasia', 
        'cisterna magna', 'colphocephaly', 'encephalocele', 'intracranial hemorrdge', 
        'mild ventriculomegaly', 'moderate ventriculomegaly', 'polencephaly', 'severe ventriculomegaly'
    ]

    # Get the predicted class name
    predicted_class = class_names[class_id]

    # Print the predicted class
    print(f"Predicted Class: {predicted_class}")

    variable_dict = {
        'my_variable' : str(predicted_class)
    }

    print(type(predicted_class))
    return render(request,'./classify_app/test.html' ,variable_dict)

# View to handle image upload
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        # Save the file to the media directory using Django's FileSystemStorage
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)  # Get the URL of the uploaded file
        file_path = fs.path(filename)  # Get the local file path
        class_id = classify_image(file_path)

        # Define your class names list
        class_names = [
            'anold chiari malformation', 'arachnoid cyst', 'cerebellah hypoplasia', 
            'cisterna magna', 'colphocephaly', 'encephalocele', 'intracranial hemorrdge', 
            'mild ventriculomegaly', 'moderate ventriculomegaly', 'polencephaly', 'severe ventriculomegaly'
        ]

        # Get the predicted class name
        predicted_class = class_names[class_id]

        # Print the predicted class
        print(f"Predicted Class: {predicted_class}")
        return render(request, './classify_app/upload.html', {
            'file_url': file_url,
            'file_path': file_path,
            'class' : predicted_class,
        })
    return render(request, './classify_app/upload.html')

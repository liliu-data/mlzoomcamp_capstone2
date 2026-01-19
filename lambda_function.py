import json
import base64
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Global Initialization: Loading the model outside the handler 
# saves time on subsequent requests (Warm Start).
device = torch.device("cpu")
MODEL_PATH = "model.pt"

# Load the TorchScript model
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()

# 2. Define Preprocessing (MUST match your training transforms)
preprocess = transform = transforms.Compose([
    transforms.ToTensor(), # Image is already 128x128 from Step 1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def lambda_handler(event, context):
    try:
        # 3. Parse the incoming request
        # The frontend sends a JSON body with a base64 string
        body = json.loads(event['body'])
        image_data = body['image']
        
        # Decode the base64 string
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 4. Transform and Predict
        input_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, prediction = torch.max(probabilities, 0)
        
        # Mapping numerical output to class names
        classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        result = classes[prediction.item()]
        score = confidence.item()

        # 5. Return the response with CORS headers
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # Required for website connectivity
            },
            'body': json.dumps({
                'diagnosis': result,
                'confidence': f"{score:.2f}",
                'message': "Analysis complete"
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e)})
        }
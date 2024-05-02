from torch import load as tload
from torch import no_grad as tno_grad
from torch import nn as nn
from torch import hub as thub
from torch import no_grad as no_grad
from cv2 import imread as cimread
from cv2 import cvtColor as cvtColor
from cv2 import COLOR_BGR2RGB as COLOR_BGR2RGB
from torch import max as tmax
from torchvision import transforms, models

import json
from urllib.parse import urlparse
import boto3
import os
import tempfile

# device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu' # AWS Lambda is CPU only
class_label_map = {
    0: "apple_pie",
    1: "baby_back_ribs",
    2: "baklava",
    3: "beef_carpaccio",
    4: "beef_tartare",
    5: "beet_salad",
    6: "beignets",
    7: "bibimbap",
    8: "bread_pudding",
    9: "breakfast_burrito",
    10: "bruschetta",
    11: "caesar_salad",
    12: "cannoli",
    13: "caprese_salad",
    14: "carrot_cake",
    15: "ceviche",
    16: "cheese_plate",
    17: "cheesecake",
    18: "chicken_curry",
    19: "chicken_quesadilla",
    20: "chicken_wings",
    21: "chocolate_cake",
    22: "chocolate_mousse",
    23: "churros",
    24: "clam_chowder",
    25: "club_sandwich",
    26: "crab_cakes",
    27: "creme_brulee",
    28: "croque_madame",
    29: "cup_cakes",
    30: "deviled_eggs",
    31: "donuts",
    32: "dumplings",
    33: "edamame",
    34: "eggs_benedict",
    35: "escargots",
    36: "falafel",
    37: "filet_mignon",
    38: "fish_and_chips",
    39: "foie_gras",
    40: "french_fries",
    41: "french_onion_soup",
    42: "french_toast",
    43: "fried_calamari",
    44: "fried_rice",
    45: "frozen_yogurt",
    46: "garlic_bread",
    47: "gnocchi",
    48: "greek_salad",
    49: "grilled_cheese_sandwich",
    50: "grilled_salmon",
    51: "guacamole",
    52: "gyoza",
    53: "hamburger",
    54: "hot_and_sour_soup",
    55: "hot_dog",
    56: "huevos_rancheros",
    57: "hummus",
    58: "ice_cream",
    59: "lasagna",
    60: "lobster_bisque",
    61: "lobster_roll_sandwich",
    62: "macaroni_and_cheese",
    63: "macarons",
    64: "miso_soup",
    65: "mussels",
    66: "nachos",
    67: "omelette",
    68: "onion_rings",
    69: "oysters",
    70: "pad_thai",
    71: "paella",
    72: "pancakes",
    73: "panna_cotta",
    74: "peking_duck",
    75: "pho",
    76: "pizza",
    77: "pork_chop",
    78: "poutine",
    79: "prime_rib",
    80: "pulled_pork_sandwich",
    81: "ramen",
    82: "ravioli",
    83: "red_velvet_cake",
    84: "risotto",
    85: "samosa",
    86: "sashimi",
    87: "scallops",
    88: "seaweed_salad",
    89: "shrimp_and_grits",
    90: "spaghetti_bolognese",
    91: "spaghetti_carbonara",
    92: "spring_rolls",
    93: "steak",
    94: "strawberry_shortcake",
    95: "sushi",
    96: "tacos",
    97: "takoyaki",
    98: "tiramisu",
    99: "tuna_tartare",
    100: "waffles"
}

TOTAL_CATEGORIES = 101

thub.set_dir('/tmp/torch_hub')

def download_file_from_s3(s3_url):
    s3_client = boto3.client('s3')
    parsed = urlparse(s3_url)
    bucket_name = parsed.netloc.split('.')[0]
    s3_path = parsed.path.lstrip('/')
    
    fd, temp_file_path = tempfile.mkstemp()
    os.close(fd)
    s3_client.download_file(bucket_name, s3_path, temp_file_path)
    
    return temp_file_path

def handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))
        s3_url = body.get('image_path')
        
        if not s3_url:
            raise ValueError("image_path is required")
        
        local_image_path = download_file_from_s3(s3_url)
        predicted_label = identifyObject(local_image_path)
        
        os.remove(local_image_path)
        
        return {
            'statusCode': 200,
            'headers': {
                "Access-Control-Allow-Headers": "Content-Type",
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*"
            },
            'body': json.dumps({'predicted_label': predicted_label})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

class FoodModel(nn.Module):
    def __init__(self, num_classes=TOTAL_CATEGORIES):
        super(FoodModel, self).__init__()
        self.resnet = models.resnet50()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
    
data_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def identifyObject(image_path):
    model_path = '/var/task/FoodModel.pt'
    loaded_model = FoodModel()
    loaded_model.load_state_dict(tload(model_path, map_location=device))
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    image = cimread(image_path)
    image = cvtColor(image, COLOR_BGR2RGB)
    input_image = data_trans(image).unsqueeze(0).to(device)

    with tno_grad():
        output = loaded_model(input_image)

    _, predicted_class = tmax(output, 1)
    predicted_label = class_label_map[predicted_class.item()]
    return predicted_label

# For Testing Locally From Public S3 Bucket
if __name__ == "__main__":
    event = {
        'body': json.dumps({'image_path': 'https://myfirstawsbucketjustformeandonlyme.s3.us-east-2.amazonaws.com/beignets.jpg'})
    }
    print(handler(event, None))
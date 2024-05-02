import requests
import json

def get_nutrition_data(food_name):
    # Nutritionix API endpoint
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"

    # API headers
    headers = {
        'Content-Type': 'application/json',
        'x-app-id': 'cdf85b33',  # replace with your app id
        'x-app-key': 'e202f3b5f1a54b169b11f7df3c8395b9'  # replace with your app key
    }

    # API body
    body = {
        "query": food_name
    }

    # Send a POST request to the Nutritionix API
    response = requests.post(url, headers=headers, data=json.dumps(body))

    # Parse the response as JSON
    data = response.json()

    # Extract the nutrition data
    foods = data.get("foods", [])
    if not foods:
        return None
    
    first_food = foods[0]
    nutrition_data = {
        "nf_calories": first_food.get("nf_calories", 0),
        "nf_total_fat": first_food.get("nf_total_fat", 0),
        "nf_protein": first_food.get("nf_protein", 0),
        "nf_total_carbohydrate": first_food.get("nf_total_carbohydrate", 0)
    }

    return nutrition_data

def handler(event, context):
    try:
        body = json.loads(event.get('body', '{}'))
        actual_body = body.get('body')
        food_name = actual_body.get('predicted_label')

        if not food_name:
            raise ValueError("predicted_label is required")

        nutrition_data = get_nutrition_data(food_name)

        data = {
            'statusCode': 200,
            'headers': {
                "Access-Control-Allow-Headers": "Content-Type",
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
            },
            'body': json.dumps({'nutrition_data': nutrition_data})
        }

    except Exception as e:
        data = {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

    return data

import requests

def fetch_satellite_image(api_key, latitude, longitude, zoom=15, width=640, height=480):
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom,
        "size": f"{width}x{height}",
        "maptype": "satellite",
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        with open("satellite_image.jpg", "wb") as f:
            f.write(response.content)
        print("Satellite image saved successfully.")
    else:
        print(f"Failed to fetch satellite image. Status code: {response.status_code}")

api_key = 'YOUR_API_KEY' # Enter the API_Key
latitude = 37.7749
longitude = -122.4194

fetch_satellite_image(api_key, latitude, longitude)
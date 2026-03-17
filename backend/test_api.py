import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "Planet_Radius": 2.5,
    "Planet_Mass": 10,
    "Orbital_Period": 20,
    "Semi_Major_Axis": 0.12,
    "Planet_Density": 3.5,
    "Equilibrium_Temp": 600,
    "Stellar_Temp": 5200,
    "Stellar_Luminosity": 0.4,
    "Stellar_Metallicity": 0,
    "StarType_A": 0,
    "StarType_F": 0,
    "StarType_G": 0,
    "StarType_K": 1,
    "StarType_M": 0
}

response = requests.post(url, json=data)

print(response.json())
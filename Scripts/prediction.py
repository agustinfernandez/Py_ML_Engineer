import json
import pandas as pd
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import pickle
import sys



def up_json(file):
    with open(file, 'r') as file2:
    
        data = json.load(file2)
        df = pd.DataFrame([data])
        df.to_csv("df.csv", index=False)
    return df

def prediction(data):
    #Feature engineering años del establecimiento
    
    data['Outlet_Establishment_Year'] = 2020 - data['Outlet_Establishment_Year']
    
    #LIMPIEZA: Unificando etiquetas para 'Item_Fat_Content'
    
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
    
    #LIMPIEZA: de faltantes en el peso de los productos
    
    productos = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
    for producto in productos:
        moda = (data[data['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
        data.loc[data['Item_Identifier'] == producto, 'Item_Weight'] = moda
    
    #LIMPIEZA: de faltantes en el tamaño de las tiendas
    outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
    for outlet in outlets:
        data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'
    
    # FEATURES ENGINEERING: creando categorías para 'Item_Type'
    data['Item_Type'] = data['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
     'Seafood': 'Meats', 'Meat': 'Meats',
     'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
     'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
     'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

    # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
    data.loc[data['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'
    
    #FEATURES ENGINEERING: Codificando los niveles de precios de los productos
    intervalos = [(31.289, 94.012), (94.012, 142.247), (142.247, 185.856),(185.856, 266.888)]
    valores_asignados = [1, 2, 3, 4]

    # Crear una nueva columna que contenga los valores asignados según los intervalos
    data['Item_MRP'] = pd.cut(data['Item_MRP'], bins=[intervalo[0] for intervalo in intervalos] + [float('inf')], labels=valores_asignados, right=False)
    
    # Se utiliza una copia de data para separar los valores codificados en un dataframe distinto.
    
    dataframe = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()
    
    # Codificación de variables ordinales
    dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
    dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos
    
    #FEATURES ENGINEERING: Codificación de variables nominales
    todos_valores = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']

    # Agregar columnas dummy faltantes con valores inicializados en 0
    for valor in todos_valores:
        dataframe[f'Outlet_Type_{valor}'] = 0

    # Marcar con 1 la columna correspondiente al valor presente en el registro de ejemplo
    dataframe.loc[0, f'Outlet_Type_{dataframe["Outlet_Type"].iloc[0]}'] = 1

    
    # Eliminación de variables que no contribuyen a la predicción por ser muy específicas y por repetición("outlet_type")
    dataset = dataframe.drop(columns=['Item_Identifier', 'Outlet_Identifier',"Outlet_Type"])
    
    
    
    with open('model.pkl', 'rb') as archivo_pkl:
        modelo_cargado = pickle.load(archivo_pkl)
        
    
    prediction = modelo_cargado.predict(dataset)
    
    return prediction

class ArgumentosInsuficientesError(Exception):
    pass

if __name__ == "__main__":
    
    try:
        
        if len(sys.argv) < 2:
            raise ArgumentosInsuficientesError("Se esperan al menos un argumento.")

        json_start =  sys.argv[1]
        json_csv_file = up_json(json_start)
        pred = prediction(json_csv_file)
        print(f"La predicción realizada es {pred}.")
        
        

    except ArgumentosInsuficientesError as e:
        print(f"Error: {e}")
        print("Uso: python mi_script.py <archivo_json>")
        sys.exit(1)  
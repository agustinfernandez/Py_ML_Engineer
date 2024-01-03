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



def up_preprocesing(data_train, data_test):
    
    # Identificando la data de train y de test, para posteriormente unión y separación
    data_train = pd.read_csv(data_train)
    data_test = pd.read_csv(data_test)

    data_train['Set'] = 'train'
    data_test['Set'] = 'test'
    data = pd.concat([data_train, data_test], ignore_index=True, sort=False)
    
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
    pd.qcut(data['Item_MRP'], 4,).unique()
    data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])
    
    # Se utiliza una copia de data para separar los valores codificados en un dataframe distinto.
    
    dataframe = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()
    
    # Codificación de variables ordinales
    dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
    dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos
    
    #FEATURES ENGINEERING: Codificación de variables nominales
    dataframe = pd.get_dummies(dataframe, columns=['Outlet_Type'])
    
    # Eliminación de variables que no contribuyen a la predicción por ser muy específicas
    dataset = dataframe.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
    # División del dataset de train y test
    df_train = dataset.loc[data['Set'] == 'train']
    df_test = dataset.loc[data['Set'] == 'test']

    # Eliminando columnas sin datos
    
    df_train.drop(['Set'], axis=1, inplace=True)
    df_test.drop(['Item_Outlet_Sales'], axis=1, inplace=True)
    df_test.drop(['Set'], axis=1, inplace=True)

    # Guardando los datasets
    df_train.to_csv("train_final.csv", index=False)
    df_test.to_csv("test_final.csv", index=False)

    return df_train

def training_process(df_train):
    
    seed = 28
    model = LinearRegression()

    # División de dataset de entrenaimento y validación
    X = df_train.drop(columns='Item_Outlet_Sales')#[['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
    x_train, x_val, y_train, y_val = train_test_split(X, df_train['Item_Outlet_Sales'], test_size = 0.3, random_state=seed)

    # Entrenamiento del modelo
    model.fit(x_train,y_train)

    # Predicción del modelo ajustado para el conjunto de validación
    pred = model.predict(x_val)

    # Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
    mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
    R2_train = model.score(x_train, y_train)
    print('Métricas del Modelo:')
    print('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

    mse_val = metrics.mean_squared_error(y_val, pred)
    R2_val = model.score(x_val, y_val)
    print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

    print('\nCoeficientes del Modelo:')
    # Constante del modelo
    print('Intersección: {:.2f}'.format(model.intercept_))

    # Coeficientes del modelo
    coef = pd.DataFrame(x_train.columns, columns=['features'])
    coef['Coeficiente Estimados'] = model.coef_
    print(coef, '\n')
    
    with open('model.pkl', 'wb') as archivo_pkl:
        pickle.dump(model, archivo_pkl)
    
    return "Entrenamiento realizado con éxito."

class ArgumentosInsuficientesError(Exception):
    pass

        
if __name__ == "__main__":

    try:
        
        if len(sys.argv) < 3:
            raise ArgumentosInsuficientesError("Se esperan al menos dos argumentos.")

        data_train = sys.argv[1]
        data_test = sys.argv[2]

        pre_po =  up_preprocesing(data_train, data_test)
        training_process(pre_po)

    except ArgumentosInsuficientesError as e:
        print(f"Error: {e}")
        print("Uso: python training.py <train_file> <test_file>")
        sys.exit(1)  

   

# Py_ML_Engineer
Process to deploy one regresion model
-----------------------------------

En la carpeta Script podrán encontrar 2 .py. Por un lado, training.py, que deberá ejecutarse en consola de la siguiente manera:

python traning.py ruta/datasetentrenamiento.csv ruta/datasetesteo.csv 

Este script guardará un modelo ("model.pkl") que se utilizará para hacer las predicciones correspondientes en el script prediction.py.

Tal y como se solicita, el scritp prediction.py deberá ejecutarse en consola (Lo ideal sería pasar como parámetro el modelo, pero así no estaba especificado):

python prediction.py ruta/json_file.json


Aquí se devolverá la predicción para los registros que contenga el .json.


------------------------------------
## Ideas acerca de una posible puesta en producción y apificación.

Si se busca llevar el modelo a producción, es necesario que tengamos en claro que necesitaremos implementar una solución robusta. Para eso, es necesario poder integrar diferentes etapas del deployment, que aseguren que el modelo pueda entregar predicciones precisas y que encuentre constantemente actualizado. A su vez, podría ser necesario contar con una herramienta que facilite el acceso a las predicciones. 
A la hora de pensar este proceso, podríamos nombrar una serie de herramientas que serían indispensables para cada etapa del proceso. Por un lado, para llevar a cabo nuestros procesos de CI/CD y de entrenamiento contínuo, podríamos utilizar como disparador de los flujos de trabajo, diferentes github actions. En lo que respecta al proceso de CI/CD podríamos tener montado el proceso en Docker, y desplegarlo en algún servicio CLoud como podría ser CloudRun.
Para el entreamiento contínuo, será necesario trabajar con data version control para serializar nuestro modelo (así como con la herramienta elegida para crear y entrenar los modelos, en  este caso scikit-learn). Con Continuos Machine Learning podríamos ir siguiendo las métricas arrojadas por nuestro modelo.
Podríamos también requerir de algún servicio de storage para trabajar con nuestra data, integrándola a nuestro proceso. S3 o cloud storage podrían ser de utilidad.

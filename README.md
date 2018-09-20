finding_waldo_yolo
Scripts de configuración para YOLOv2 YOLOv3 para buscar a Waldo.
* **image_augmentation.py:** Script en Python3 para aumentar el tamaño del conjunto inicial de imágenes. Basado en imgaug, si se quiere hacer uso de este script instalar el paquete [https://imgaug.readthedocs.io/en/latest/source/installation.html]
* **proccess_dataset.py:** Script en Python3 que genera los conjuntos de pruebas y entrenamiento para la red.
* **waldo.data:** Archivo con la metadata de la red a entrenar.
* **waldo.names:** Archivo con los nombres de las clases
* **yolov2-waldo.cfg:** Configuración de la red YOLOv2.
* **yolov3-waldo.cfg:** Configuración con la red YOLOv3.
* **data/:** Carpeta con las imágenes de Waldo, para este ejemplo se trabajará solo con imagenes *.jpg en caso de agregar mas mantener el formato.
* **data/aug:** Carpeta de salida del script de aumentación de imágenes.
* **validate/:** Carpeta con imágenes de validación del modelo.
### Generar imágenes
   ```
   #! finding_waldo_yolo/
   python ./image_augmentation.py 
   ```
   
### Generar datasets
``` 
#! finding_waldo_yolo/
python ./proccess_dataset.py [images_location_dir] [darknet_search_path]
``` 
### Tutorial


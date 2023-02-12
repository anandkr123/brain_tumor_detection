# Brain tumor classification

Sample images from Validation data set

![brain_tumor_validation_images](https://user-images.githubusercontent.com/23450113/216850715-89dcbf76-fc84-4ed1-bafb-174b94903b2e.png)


## Dataset source 
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?resource=download


## Create model using Transfer learning
- Tensorflow hub
- RestNet_v1 feature layer
- Keras sequential api

## Train and validation set

- Epochs          --> 10
- Batch size      --> 4
- Optimizer       --> Adam()
- Loss function   --> BinaryCrossentropy
- Trace metrics   --> ['accuracy', 'Precision', 'Recall']


## Classification results **after 10 epochs on validation set** 

![Screenshot from 2023-02-06 00-04-34](https://user-images.githubusercontent.com/23450113/216851078-6d1069c9-b338-4d2c-8bfb-b3003067ad90.png)

## Deployed the saved keras model using tensorflow serving using docker image tensorflow/serving

### Test the results on validation datset 

#### Image with no brain tumor classified as 0

- Image
![result_no_tumor](https://user-images.githubusercontent.com/23450113/218318290-b39e8197-89c7-4e79-8540-dbef4b841f8b.png)

- Results
![Screenshot from 2023-02-12 15-52-04](https://user-images.githubusercontent.com/23450113/218318435-1e382a8f-907e-4f90-9166-3bba47a2c5dd.png)


#### Image with brain tumor classified as 1

-Image 
![result_tumor](https://user-images.githubusercontent.com/23450113/218318362-963f1dd7-4095-49d9-bda0-ee19f864b638.png)

-Result
![Screenshot from 2023-02-12 15-46-06](https://user-images.githubusercontent.com/23450113/218318455-33043faa-0c57-4af7-93b1-df98478360ac.png)
ts

# Brain_tumor_detection

Sample images from Validation data set

![brain_tumor_validation_images](https://user-images.githubusercontent.com/23450113/216850715-89dcbf76-fc84-4ed1-bafb-174b94903b2e.png)

## Dataset source 
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?resource=download

## Create model using Transfer learning, from **tensorflow hub, RestNet_v1** feature layer

## Train and validation set

- Epochs          --> 10
- Batch size      --> 4
- optimizer       --> Adam()
- Loss function   --> BinaryCrossentropy
- Trace metrics   --> ['accuracy', 'Precision', 'Recall']

## Classification results **after 10 epochs on validation set** 

![Screenshot from 2023-02-06 00-04-34](https://user-images.githubusercontent.com/23450113/216851078-6d1069c9-b338-4d2c-8bfb-b3003067ad90.png)

# Convolutional Neural Network
Breast Cancer Prediction on histopathology images

## Intro
The most common type of breast cancer is called invasive ductal carcinoma (IDC). Doctors have to be very careful to identify and classify different types of breast cancer correctly, and sometimes they use machines to help them do it quicker and more accurately. When doctors want to assess how aggressive a particular case of breast cancer is, they usually look at the subtype called invasive ductal carcinoma (IDC), which is actually the most common type of breast cancer. To do this, they examine specific regions of a sample taken from the patient. In order to automate this process and use machines to assign an aggressiveness grade, one of the initial steps is to identify and outline the exact areas that contain IDC within the sample.

## Dataset
Image data from https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

The original dataset contained 162 whole mount slide images of specimens from patients with breast cancer. These images were scanned at 40x magnification, and from them, researchers extracted a total of 277,524 smaller patches, each measuring 50 x 50 pixels. Of these patches, 198,738 were from areas of the slides that tested negative for invasive ductal carcinoma (IDC), while the remaining 78,786 patches were from areas that tested positive for IDC.

The file names for each patch contain useful information, such as the patient ID (which is "u" followed by a unique identifier like "10253_idx5"), as well as the x and y coordinates of the patch's location on the original slide. Additionally, each file name includes a "class" designation, where 0 represents a patch that does not contain IDC, and 1 represents a patch that does contain IDC. For example, a patch with the file name "10253_idx5_x1351_y1101_class0.png" would indicate that it was taken from a patient with ID "10253_idx5," and it does not contain IDC.

 e.g. 10253_idx5_x1001_y1301_class0.png
        
     patient_id    |     |     |     |
        
            x-coordinate |     |     |
     
                 y-coordinate  |     |
         
          cancer class[No 0/ Yes 1]  |
         
                                 file type

image examples ![10253_idx5_x1001_y1001_class0](https://user-images.githubusercontent.com/100010968/201340965-cdb25d25-2af6-41d3-9a58-fd1d4a4bfbde.png)
![10253_idx5_x501_y401_class1](https://user-images.githubusercontent.com/100010968/201340968-bb5c503b-cb7c-43d6-b6d9-5ccd132fd231.png)


## Process Data
Separate images into pos and neg, saving in different folder and assign labels onto the images

## CNN model

    Conv2D -> Conv2D -> Conv2D -> Batch norm -> maxPool -> Dropout (start again for 3 times) -> flatten -> Dense ('relu') -> softmax

The first convolutional layer learns basic features like edges, lines, and curves. The second layer combines these features to detect more complex shapes and structures, such as textures and shapes of objects. The third layer further combines these features to identify even more complex patterns and structures in the input image. The input to each layer changes as the network learns. This can cause the distribution of inputs to each layer to shift, a problem known as "internal covariate shift." This shift can cause the network to take longer to train, or even to stop learning entirely. 

In order to improve the training stability, adding batch normalisation to calculates the mean and standard deviation of the inputs in a batch, and normalises the inputs to have zero mean and unit variance reducing the impact of internal covariate shift. I also added Maxpooling to reduce the number of parameters and computation required by the neural network, which in turn can improve its efficiency and ability to generalize to new data and dropping out random neurons to maintains the models robustness.

After completing the above process 3 times, we flatten the layer to transform the input tensor into 1D array which then can be fed into the connected layer then we have our results decision by the softmax function.


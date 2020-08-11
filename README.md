# Semantic-Segmentation-of-Teeth-in-Panoramic-X-ray-Image
The aim of this study is automatic semantic segmentation and measurement total length of teeth in one-shot panoramic x-ray image by using deep learning method with U-Net Model and binary image analysis in order to provide diagnostic information for the management of dental disorders, diseases, and conditions.

The authors of this article are Selahattin Serdar Helli and Andaç Hamamcı  with the Department of Biomedical Engineering, Faculty of Engineering, Yeditepe University, Istanbul, Turkey.

U-Net Network ref - 	Olaf Ronneberger, Philipp Fischer, and .omas Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, 2015, pp. 234–241.

DATASET ref - 	H. Abdi, S. Kasaei, and M. Mehdizadeh, “Automatic segmentation of mandible in panoramic x-ray,” J. Med. Imaging, vol. 2, no. 4, p. 44003, 2015
Link[https://data.mendeley.com/datasets/hxt48yk462/1]
<img src="https://github.com/SerdarHelli/Semantic-Segmentation-of-Teeth-in-Panoramic-X-ray-Image/blob/master/Viewing_Estimations/Figures/example.png" alt="Results" width="1024" height="512">
This example of the model’s output has an effective segmentation map.
<img src="https://github.com/SerdarHelli/Semantic-Segmentation-of-Teeth-in-Panoramic-X-ray-Image/blob/master/Viewing_Estimations/Figures/exampleofcca.png" alt="Results" width="1024" height="512">
In this example of binary image processing, teeth have successfully labeled
<img src="https://github.com/SerdarHelli/Semantic-Segmentation-of-Teeth-in-Panoramic-X-ray-Image/blob/master/Viewing_Estimations/Figures/Architecture.png" alt="Results" width="1024" height="512">
Our neural network architecture. Each blue box corresponds to a multi-channel feature map. The number of channels is denoted on top of the
box. The x-y-size is provided at the lower-left edge of the box. White boxes represent copied feature maps. The arrows denote the different operations

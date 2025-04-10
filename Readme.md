# YOLOv8 Suland Project for Domain Adaptation

This project builds upon the work of [1], where YOLOv8s and YOLOv8n models were successfully applied to the ”SurfLandmine” dataset. Although these models performed well with in-distribution (IID) data, they exhibited high false positive rates and struggled with out-of-distribution (OOD) data, highlighting the need for improved generalization. The primary goal of the project is to enhance the model’s generalization ability across diverse backgrounds, as collecting real-world landmine data in
various environments is expensive, dangerous, and logistically challenging. To overcome this limitation, we propose a data augmentation pipeline that will artificially generate diverse training data.

The landmine image patches will first be segmented from the existing training data using the Segment Anything Model (SAM), a model chosen for its strong generalization capabilities. These segmented landmine patches will then be pasted onto new, varied backgrounds, simulating different environmental conditions. Several augmentation techniques will be applied, including hue and saturation adjustments, translation, scaling, flipping, rotation, and mosaic augmentation to furtherenhance diversity.

Once these augmented landmine images are generated, they will be integrated back into the training data. The YOLOv8s model will then be retrained on this newly augmented dataset, with the goal of improving its ability to detect landmines in out-of-distribution data.


# References: 
 [1] Vivoli, E.; Bertini, M.; Capineri, L. Deep Learning-Based Real-Time Detection of Surface Landmines Using Optical Imaging. Remote Sens. 2024, 16, 677. https://doi.org/10.3390/rs16040677

 
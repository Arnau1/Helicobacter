# Helicobacter detection through medical imaging

### Concept
Project from the Vision and Learning subject from the Artificial Intelligence degree at UAB. 

The aim is to be capable of detecting whether a patient has or not the bacteria _H. Pylori_. The Quiron Dataset provided in the course, which can be found on the Virtual Campus, has been used. Patient images are separated into multiple patches, which can have bacteria or not.

The main assumption behind the approach used is that when bacteria are present patches have more red pixels. However, simply counting pixels does not yield good results. Detection through an autoencoder trained so that it is capable of replicating the structure of patches but not red color is a more effective approach.


![Example1](https://github.com/user-attachments/assets/f190ba3e-ea70-4ecb-b466-d46aa209230f)
![Example2](https://github.com/user-attachments/assets/4f6ca58e-a552-4037-a7d8-21608942ecb7)



### Results
Results are measured for patch and patient classification. In patch classification, only a test set is classified, since no more labelled data was available. On patient classification, a test set and a chunk of raw unseen data are classified. 
#### Patch classification
![image](https://github.com/user-attachments/assets/9f50ae49-dc2b-477f-8d31-0e92751f25da)
#### Patient classification
Test set


![image](https://github.com/user-attachments/assets/fbfa93f7-aff3-4fbe-ace3-f91555a3f8b0)

Unseen data


![image](https://github.com/user-attachments/assets/fa339748-95b1-4e20-93f9-ca9f7f2bde54)

The over-classification of patients as not infected on unseen data may be due to the sparsity of infected patches on raw data.
##### For further information read the report

### References

[1] Ronneberger, O., Fischer, P. and Brox, T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation, arXiv.org. Available at: https://arxiv.org/abs/1505.04597 (Accessed: 10 November 2024).

[2] Cano, P. et al. (2023) Diagnosis of helicobacter pylori using autoencoders for the detection of anomalous staining patterns in immunohistochemistry images, arXiv.org. Available at: https://arxiv.org/abs/2309.16053 (Accessed: 10 November 2024).

[3] https://stackoverflow.com/

[4] https://chatgpt.com/



Mustapha El Aichouni (1668936), Arnau Solé Porta (1630311), Josep Bonet Saez (1633723) and Jordi Longaron (1630483)


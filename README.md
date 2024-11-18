# Helicobacter detection through medical imaging

### Concept
Project from the Vision and Learning subject from the Artificial Intelligence degree at UAB. 

The aim is to be capable of detecting whether a patient has or not the bacteria _H. Pylori_. The Quiron Dataset provided in the course, which can be found on the Virtual Campus, has been used. Patient images are separated into multiple patches, which can have bacteria or not.

The main assumption behind the approach used is that when bacteria are present patches have more red pixels. However, simply counting pixels does not yield good results. Detection through an autoencoder trained so that it is capable of replicating the structure of patches but not red color is a more effective approach.


![Example2](https://github.com/user-attachments/assets/4f6ca58e-a552-4037-a7d8-21608942ecb7)



### Results
Results are measured for patch and patient classification. In patch classification, only a small labelled set is classified, since no more labelled data was available. On patient classification, a chunk of raw unseen data is classified. 
#### Patch classification
![image](https://github.com/user-attachments/assets/ecf8e59b-afa2-4393-8bf9-75a697901bf5)

#### Patient classification
![image](https://github.com/user-attachments/assets/0a7522c0-92b1-4d9d-bb4a-c002b8799dfa)


##### For further information read the report

### References

[1] Cano, P. et al. (2023) Diagnosis of helicobacter pylori using autoencoders for the detection of anomalous staining patterns in immunohistochemistry images, arXiv.org. Available at: https://arxiv.org/abs/2309.16053 (Accessed: 10 November 2024).

[2] https://stackoverflow.com/

[3] https://chatgpt.com/


Mustapha El Aichouni (1668936), Arnau Solé (1630311), Josep Bonet (1633723) and Jordi Longaron (1630483)


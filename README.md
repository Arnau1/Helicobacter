# Helicobacter detection through medical imaging

Project from the Vision and Learning subject from the Artificial Intelligence degree at UAB. 

### Concept

The aim is to be capable of detecting whether a patient has or not the bacteria _H. Pylori_. The Quiron Dataset provided in the course, which can be found on the Virtual Campus, has been used. Patient images are separated into multiple patches, which can have bacteria or not.

The main assumption behind the approach used is that when bacteria are present patches have more red pixels. However, simply counting pixels does not yield good results. It is more effective to train an autoencoder to reconstruct sane patches and use the reconstruction error, which will be higher in infected patches, as a metric.


![Example2](https://github.com/user-attachments/assets/4f6ca58e-a552-4037-a7d8-21608942ecb7)



### Results
Results are measured for patch and patient classification. In patch classification, only a small labelled set is classified, since no more labelled data was available. On patient classification, a chunk of raw unseen data is classified. 
#### Patch classification
![image](https://github.com/user-attachments/assets/ecf8e59b-afa2-4393-8bf9-75a697901bf5)

#### Patient classification
![image](https://github.com/user-attachments/assets/0a7522c0-92b1-4d9d-bb4a-c002b8799dfa)


For further information read the report


Mustapha El Aichouni (1668936), Arnau Solé (1630311), Josep Bonet (1633723) and Jordi Longaron (1630483)


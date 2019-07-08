## Towards intelligent imaging: Role of applied AI

The goal of this post is to develop broad range of necessary AI-technologies to develop new data acqusition and reconstructions which not only allow efficient/fast data acquisitions but also synthesis of novel contrasts to allow more "effective diagnosis" followed with automatic image intereption in an end-to-end setting, a marked differently and radical approach.  This would require moving away from the traditional approaches and representation by adopting a more generic embedded-space representation, a following road map outlines the development statergy. 

 ## Road map

1. Develop good representation underlying objects aka. structures of organs to embedding's via employing computer vision algorithms to develop embeddings 

2. Design novel contrast from the learned representations for effective diagnosis learning for complex probability distributions via attention modelling or baysien modelling 

3. Combine multi-modal data with ultimate goal of better risk stratification along with other datasets (e.g. radiology report read via NLP) via automated intepretation   

 #### Algorithms implemented 
 
 #### Automatic (unsupervised) object (organ) detection/segmentation in medical images (via combination DeepLab-3, U-NET, GAN)
 #### Using segmentation obtain "image" embeddings (WIP: need more datasets), these would allow development highly of efficient acqusition 
 #### Employing embeddings and policy-based-learning (DDPG) for synthesis of optimal image acquisition of varying contrast (WIP-see RI repo for a simple implementation)
 #### Attention modelling e.g. Show attend and tell, model: which can generate captions for images, this would be the final layer for automatic inpretation of data, feeding back to a decision model (DDPG: Actor-Critic, Gaussian Processes), to learn new acqusition parameters based on the current observation


## docker image: jehillparikh/betamlstack:v2 (for all dependency employed in this project)

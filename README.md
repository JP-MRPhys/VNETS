## Towards intelligent imaging: Role of applied AI

### Background

MRI/CT/US are very powerful scanners employed to investigate and diagnose a range of diseases. MRI scans in particular are acquired by employing specific parameters in a "sequence" to encode in data in arbitrary space known as "k-space" and then by applying mathematical transforms (mainly fourier, recently wavelets, and dictornary) to obtain images of different constrast (T1w,T2w,T2*, DTI, etc). 

To reconstruct a "clean" image i.e. without artifacts, k-space encoding rules must be followed, which leads to slow "encoding/acqusitions" and longer scans duration. As a result a huge amount of effort has been devoted to increase efficiency data collection over the last three decades. Recent developments suggest, neural networks, which act as function estimators allow enhance reconstruction have potentail replace  long encoding process to a quicker via using transfer learning to unify reconstruction process between different modalities. These report demonstrate traditional fourier based reconstruction appraoches can be avoided and the neural networks are generic i.e. applicatable to other domains (CT, PET) etc. 

Neural Networks/Deep Learning also been applied classification of the medical images mainly as computer vision "problem".   

The goal of this post is to develop broad range of necessary AI-technologies to develop new data acqusition and reconstructions which not only allow efficient/fast data acquisitions but also synthesis of novel contrasts to allow more "effective diagnosis" followed with automatic image intereption in an end-to-end setting, a marked differently and radical approach.  This would require moving away from the traditional approaches and representationn by adopting a more generic embedded-space representation, a following road map outlines the development statergy. 

 ## Road map

1. Develop good representation underlying objects aka. structures of organs to embedding's via employing computer vision algorithms. 

2. Employ embedding distribution to isolate specific organs with effective representation so that can be employed on current hardware (e.g. MRI/PET/CT/US scanners), similar to word2vec or glove 

3. Design novel contrast from the learned representations for effective diagnosis (policy learning) learning for complex probability distributions or attention modelling or baysien modelling 

4. Combine multi-modal data with ultimate goal of better risk stratification along with other datasets (e.g. radiology report read via NLP) via automated intepretation   

This git-hub repository highlight key algorithms and provides core implementations. These are briefly summaried below. 

 #### Automatic (unsupervised) object (organ) detection/segmentation in medical images (via combination DeepLab-3, U-NET, GAN)
 #### Using segmentation obtain "image" embeddings (WIP: need more datasets), these would allow development highly of efficient acqusition 
 #### Employing embeddings and policy-based-learning (DDPG) for synthesis of optimal image acquisition of varying contrast (WIP-see RI repo for a simple implementation)
 #### Attention modelling e.g. Show attend and tell, model: which can generate captions for images, this would be the final layer for automatic inpretation of data, feeding back to a decision model (DDPG: Actor-Critic?), to learn new acqusition parameters based on the current observation

This work is highly experimental and not validated for any clinical/diagonistic use. Trained model/software support cannot be provided.  Publication (WIP). 

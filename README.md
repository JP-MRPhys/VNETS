## Towards intelligent imaging: Role of applied AI and generative models

The goal of this post is to develop broad range of necessary AI-technologies to develop new data acqusition and reconstructions which not only allow efficient/fast data acquisitions but also synthesis of novel contrasts to allow more "effective diagnosis" followed with automatic image intereption in an end-to-end setting, a marked differently and radical approach.  This would require moving away from the traditional approaches and representation by adopting a more generic embedded-space representation, a following road map outlines the development statergy. 

 ### Road map

1. Develop good representation underlying objects aka. structures of organs to embedding's via employing computer vision algorithms to develop embeddings/latent representation
 
   Algos: GAN, U-NET, DENSE-nets  

2. Design novel contrast from the latent representations for effective diagnosis learning for complex probability distributions
  
   Update sub-goal: assess feasiblity of generative models plus RI 
   
   Important papers: 
   1. Spiral (https://github.com/deepmind/spiral), W-GAN-GP+RL
   2. World Models (https://worldmodels.github.io/) VAE+RNN
   3. Non RL learning approach based on T/R imaging AutoSEQ (http://www.enc-conference.org/portals/0/Abstracts2019/ENC20198520.4608VER.2.pdf)




### docker image: jehillparikh/betamlstack:v2 (for all dependency employed in this project)


UPDATE: 01 November 2019


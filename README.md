## Towards intelligent imaging: Role of applied AI and generative models

The goal of this post is to develop broad range of necessary AI-technologies to develop new data acqusition and reconstructions which not only allow efficient/fast data acquisitions but also synthesis of novel contrasts to allow more "effective diagnosis" followed with automatic image intereption in an end-to-end setting, a marked differently and radical approach.  This would require moving away from the traditional approaches and representation by adopting a more generic embedded-space representation, a following road map outlines the development statergy. 

 ### Road map

1. Develop good representation underlying objects aka. structures of organs to embedding's via employing computer vision algorithms to develop embeddings/latent representation
 
   Algos: GAN, U-NET, DENSE-nets  

2. Design novel contrast from the latent representations for effective diagnosis learning for complex probability distributions
   
   Alogs: infoGAN, attention models (BERT-WIP)
   
   Update sub-goal: assess feasiblity of generative models plus RI 
   
   Important papers: 
   1. Spiral (https://github.com/deepmind/spiral), W-GAN-GP+RL
   2. World Models (https://worldmodels.github.io/) VAE+RNN
   3. Non RL learning approach based on T/R imaging AutoSEQ (http://www.enc-conference.org/portals/0/Abstracts2019/ENC20198520.4608VER.2.pdf)

3. Combine multi-modal data with ultimate goal of better risk stratification along with other datasets (e.g. radiology report read via NLP) via automated intepretation 
  
    Algos: Show attend and tell image-to-text, (BERT-based disease embeddings), Research (Knowlegde Graphs)  

4. Develop framework for safe AI
   
   Alogs: Baysien Networks for development of robust and safe 



### docker image: jehillparikh/betamlstack:v2 (for all dependency employed in this project)


UPDATE: 01 Oct 2019

There is recent literature which make 

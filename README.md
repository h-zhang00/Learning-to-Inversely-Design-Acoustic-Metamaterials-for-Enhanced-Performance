# Learning-to-Inversely-Design-Acoustic-Metamaterials-for-Enhanced-Performance
H. Zhang*, J. Liu, W. Ma, H. Yang, Y. Wang, H. Yang, H. Zhao, D. Yu, J. Wen*

https://link.springer.com/article/10.1007/s10409-023-22426-x

## Abstract
Elastic metamaterials are popularly sought to realize numerous special functions such as vibration control and wave manipulation among which sound absorption is a typical task fulfilled by acoustic metamaterials. Inverse designing metamaterials with machine learning approaches has been under the spotlight thanks to the data-driven experience-free advantages and become one of the important design paradigms. Nevertheless, the existing works mostly concentrate on validating the reproduction accuracy of the neural networks on trained data and very few have explored their ability on designing for enhanced properties. To this end, our work studies the competence of the proposed inverse design framework in enhancing the acoustic performance of a three-dimensional mixed-size cavity-based waterborne sound absorptive metamaterial. With forward and inverse networks in the framework, the target sound absorption spectra (100-10000 Hz) are taken as inputs into the inverse network during training and a corresponding structure is output with the best matching spectra which is subsequently fed into the forward network for acoustic property evaluation and loss calculation. The trained forward network is shown to possess excellent generalization capabilities by highly accurately predicting for structures with “unseen” beyond-range parameters compared to the training set. Most importantly, the inverse network is delightfully capable of spontaneously adopting beyond-range structural parameters to ensure meeting the acoustic target whose mean sound absorption coefficient is higher than any of the data in the training set, hence inverse designing for enhanced performance. The inverse design accuracy is dramatically improved from only 9.2% of mean squared errors being <0.0001 to 99.6% with beyond-range exploration. A case study is presented to demonstrate the significant difference beyond-range exploration makes for inverse designing aiming at enhanced performance. It is hoped that this work will serve as an inspiration for the design and optimization of elastic metamaterials with enhanced performance for future work.

## Requirements
CUDA==10.1.105
python==3.8
visdom==0.1.8.9
torch==1.5.0 

## Cite
Zhang, H., Liu, J., Ma, W. et al. Learning to inversely design acoustic metamaterials for enhanced performance. Acta Mech. Sin. 39, 722426 (2023). https://doi.org/10.1007/s10409-023-22426-x


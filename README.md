# FEATURE EXTRACTION FOR HYPOGLYCEMIC EVENTS DETECTION BASED ON ECG USING TRANSFER LEARNING 

Hypoglycemia is a condition that occurs when sugar levels in the blood are low. The blood glucose level is considered low when it drops below 3.3mmol / L [1]. Hypoglycemia affects the electrophysiology of the heart. Hence, monitoring blood glucose levels is vital for healthy subjects and crucial for patients suffering from diabetes. The current study used Artificial Intelligence (AI) to extract high-level features from raw electrocardiogram (ECG) signals to automatically detect hypoglycemic events

## INTRODUCTION 

Hypoglycemia is one of the most important complications of diabetes treatment [2]. The pervasiveness of diabetes has reached high proportions in most parts of the world. According to the World Health Organization (WHO), nearly 220 million people are diagnosed with diabetes. Epidemiologic evidence insinuates that unless preventive measures are carried out, the global prevalence will continue to shoot [3]. Various glucose monitoring systems are available in the market to measure blood glucose concentration. However, such systems have many drawbacks in terms of functionality and reliability. A study has been conducted to detect hypoglycemia by monitoring changes in ECG and skin conductance using sensors [4]. A robust real-time monitoring system utilizing Continuous Glucose Monitoring Devices (CGMs) was developed in paper [5]. However, the system was not developed commercially due to low efficiency and lack of sensitivity in the detection of unrecognized hypoglycemia. The main drawback of many methods that utilize deep learning is the lack of data available. 

Recent performances achieved by deep learning models in pattern recognition tasks have motivated researchers to implement techniques such as transfer learning in medical image and signal processing to overcome the problem of inadequate data. In this regard, deep learning models have produced accurate results in the ECG domain using Recurrent Neural Networks (RNN), specifically Long Short Term Memory Networks (LSTM) [6] and Convolutional Neural Network (CNN)  [7]. The main advantage of using deep learning is the fact that deep neural networks can identify high-level features, therefore eliminating the need to extract features manually. Utilizing this merit helps us develop robust systems that can take ECG signals as input and output glucose levels for the detection of hypoglycemia. 

In paper [8], it was shown that a personalized medicinal approach and AI could be used to detect nocturnal hypoglycemia using a few heartbeats of raw ECG signal recorded using wearable devices for 14 consecutive days. Additionally, the paper presents a visualization technique to identify which part of the ECG signal is associated with hypoglycemia events.

The study proposed in this paper aims at developing a method to extracts high-level features from ECG excerpts for the automatic detection of low levels of glucose in practical settings. To the best of the author’s knowledge, the proposed study is a novel approach providing reliable results and overcoming the limitations of the lack of data available by incorporating transfer learning from the 2-dimensional domain. The image classification and object detection domains consist of immense amounts of data in contrast to the ECG signal domain when the amount of data available are significantly small. These domains contain enough data to train and classify images and find high-level feature maps that can represent complex image patterns. The features that are learnt from such classifiers can be transferred to the ECG domain if the 1-D ECG signal is transformed into a 2-D spectrogram [9].

## BACKGROUND

### The challenge of Hypoglycemia detection

The most prevalent method for glucose level detection in the blood is consist of analyzing blood droplets resulted from a finger prick. However, this method does not monitor glucose levels continuously. Moreover, this method is cumbersome, expensive, and inaccurate. As an alternative, CGMs are used to monitor glucose levels continuously based on glucose in the interstitial fluid. While these devices provide relatively accurate results, they are highly expensive and unattractive for pre-diabetic patients. Moreover, most CGMs require finger prick calibration, which is not very reliable. 

The details that need to be analyzed in an ECG signal are fine-grained and identical; therefore, have patterns that are hard to detect. This problem could be solved with the availability of large datasets. However, the datasets existing in the ECG domain contain a small amount of data. When a small amount of data is used for training, the model is susceptible to overfitting and may fail to generalize and detect patterns in unseen data.   

Therefore, the amount of data available, validation accuracy and reliability of hypoglycemic events on unseen data propose a challenge in training a deep neural network. However, for the same reasons, we incorporate a technique known as “Transfer Learning” and “Off-the-Shelf CNN Features” for accurate detection of hypoglycemia on previously unseen data. 

### Transfer Learning 

Knowledge learnt from a pattern in one domain is transferable to another domain. The Transfer learning technique enables us to transfer this knowledge from one domain to another and used in the latter domain for classification. A deep CNN is used for automatic feature extraction. The layers inside the network consist of feature maps that are learnt during the training on the original dataset. Moreover, these feature maps hold information regarding the patterns in that dataset. These feature maps can be used to extract feature in another dataset. These off-the-shelf features are strong enough to extract features in another domain [10].

In this study, we incorporate a Deep Neural Network, DenseNet that learn from millions of images via the ImageNet dataset. The knowledge learnt from this large dataset is transferred to the ECG domain and is used for feature extraction. Consequently, these high-level features can be used to detect hypoglycemic events.

###  DenseNet

A DenseNet is a deep feed-forward neural network in which each layer is connected to every other layer [11]. This type of model architecture helps reduce the problem of vanishing-gradient. The state-of-the-art DenseNet model performs highly accurate object detection tasks. The DenseNet consists of 4 dense blocks of variable length. In our study, we incorporate a pre-trained DenseNet which has 161 convolutional layers and analyze the output of these layers for feature extraction.   

## METHODOLOGY

### Method Overview

To detect hypoglycemic events using ECG excerpts, we first cut the ECG signal recordings based on the glucose levels recorded by CGMs. Our model was evaluated on 5-minutes ECG excerpts. The first 200 beats in the given excerpt were considered as input to the DenseNet. This is considered as one data instance. Each instance of data is transformed into an image using spectrograms. These images are fed into the DenseNet model, and high-level features are extracted by analyzing the output of 12 convolutional layers.

### Dataset

The dataset used in our study is extracted from the open D1NAMO dataset [12]. The dataset was acquired on 20 healthy subjects and 9 subjects diagnosed with Type-1 diabetes. The data is collected in a practical setting with the Zephyr BioHarness 3 wearable device. The device can measure ECG, breathing and accelerometer signals. Additionally, the dataset contains glucose measurements and annotated food pictures. The device also measures additional metrics such as Hearth Rate (HR), Breathing Rate BR, activity level, posture, etc. The device measures ECG at a rate of 250 Hz and up to 54.89 mV. 

### Data selection

ECG excerpts of 5 minutes were annotated with the help of CGM readings. A glucose concentration lower than 4 mmol/L is considered to be ‘lower glucose level’, and a glucose concentration between 4 mmol/L and 7.5 mmol/L is considered ‘normal glucose level’. Heartbeats corresponding to the extracted ECG signals are considered for training. 

### Signal Transformation 

Since we incorporate a pre-trained deep neural network which is trained on images and used as a feature extractor for ECG signals, every instance of data must be transformed into an image. We use spectrograms to carry out this transformation. Spectrograms apprehend variations in the power of the signals in every image by taking the Fourier Transform (FT) of each partition of the ECG signal

### Feature selection

Feature extraction is carried out by examining the output of convolutional layers inside the DenseNet model. The extracted features are considered as feature vectors. We select 12 layers and extract feature vectors from the output of each convolutional layer. This is evident as each layer contains different feature maps which are activated due to unique patterns. 

## CONCLUSION 

The study advances the feasibility of a real-time, non-invasive system to detect hypoglycemia using short excerpts of ECG signals and the technique of transfer learning. Efforts have been made to incorporate a deep neural network, pre-trained on millions of images, is used as a feature extractor on the ECG signals. Transfer learning helps us overcome the problem of the lack of data available in the ECG domain. Hence, the features maps learnt from the ImageNet dataset is powerful enough to represent the spectrograms of ECG signals. These high-level features can be utilized to detect low levels of glucose in the blood using a few raw ECG-signals recorded using a wearable device. 

## REFERENCES 

[1] 	D. M. Nathan, "The Diabetes Control and Complications Trial/Epidemiology of Diabetes Interventions and Complications Study at 30 Years: Overview," Diabetes Care, pp. 9-16, 2013. 

[2] 	M. M.-T. M. P. a. B. L. Gita Shafiee, "The importance of hypoglycemia in diabetic patients," Journal of Diabetes & Metabolic Disorders , pp. 11-17, 2012. 

[3] 	P. Z. J. S. K G M M Alberti, "International Diabetes Federation: a consensus on Type 2 diabetes prevention," Diabetic Medicine , pp. 451-463, 2007. 

[4] 	B. W. Carlos Eduardo Ferrante do Amaral, "Current development in non-invasive glucose monitoring," Medical Engineering & Physics, pp. 541-549, 2008. 

[5] 	D. R. I. C. N. (. S. Group, "Evaluation of Factors Affecting CGMS Calibration," Diabetes Technology & Therapeutics, vol. 8, no. 3, 2006. 

[6] 	Ö. Yildirim, "A novel wavelet sequence based on deep bidirectional LSTM network model for ECG signal classification," Computers in Biology and Medicine , pp. 189-202, 2018. 

[7] 	A. Y. H. M. H. C. B. A. Y. N. Pranav Rajpurkar, "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks," Computer Vision and Pattern Recognition, 2017. 

[8] 	S. S. A. P. &. L. P. Mihaela Porumb, "Precision Medicine and Artificial Intelligence: A Pilot Study on Deep Learning for Hypoglycemic Events Detection based on ECG," Scientific Reports , 2020. 

[9] 	S. T. J.-S. Y. Milad Salem, "ECG Arrhythmia Classification Using Transfer Learning from 2-Dimensional Deep CNN Features," in IEEE Biomedical Circuits and Systems (BIOCAS), 2018. 

[10] 	H. A. J. S. S. C. Ali Sharif Razavian, "CNN Features off-the-shelf: an Astounding Baseline for Recognition," Computer Vision and Pattern Recognition, 2014. 

[11] 	G. Huang, Z. Liu, L. V. D. Maaten and K. Q. Weinberger, "Densely Connected Convolutional Networks," in Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 

[12] 	J.-E. R. S. B. J.-P. C. J. R. M. S. Fabien Dubosson, "The open D1NAMO dataset: A multi-modal dataset for research on non-invasive type 1 diabetes management," Informatics in Medicine Unlocked, vol. 13, pp. 92-100, 2018. 




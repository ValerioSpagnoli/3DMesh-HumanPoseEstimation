# A Simple Neural Network Approach for 3D Human Mesh Keypoints Estimation with a Novel Dataset

This repository presents a novel approach for 3D human mesh keypoint estimation, featuring a newly developed dataset and a simple neural network architecture.

## Project Overview
This project tackles the challenge of extracting keypoints from 3D human meshes for pose estimation. We address the scarcity of suitable datasets by introducing the **Human Mesh Keypoints Extraction dataset**, derived from the *Human Body Segmentation dataset*. Our approach utilizes a neural network that leverages *mesh edge features* to predict 12 essential body joint keypoints. A core innovation lies in employing **Hungarian Matching** within the loss function to handle the unordered nature of keypoints across meshes, ensuring accurate matching between predicted and ground truth keypoints.

## Features
* **New Dataset Creation**: We developed the **Human Mesh Keypoints Extraction dataset** from the *Human Body Segmentation dataset*, addressing a critical gap in suitable datasets for 3D human pose estimation using meshes.
* **Simple Neural Network Architecture**: The project proposes a *small and efficient neural network* designed to predict *12 keypoints* representing critical human body joints by leveraging mesh edge features.
* **Keypoint Association Solution**: The challenge of unordered keypoints is effectively resolved using *Hungarian Matching in the loss function*, which ensures optimal correspondence between predicted and ground truth keypoints.

## Dataset
The **Human Mesh Keypoints Extraction dataset** was created by deriving it from the *Human Body Segmentation dataset*. This dataset addresses the lack of suitable resources for 3D human pose estimation, especially those based on mesh structures. It consists of **370 training models and 18 test models**, with meshes segmented into eight body part labels. Our unique algorithm identifies keypoints by interpreting the connections between these segmented parts as human body joints. The final annotated dataset comprises 292 meshes, each with 12 keypoints.

<p align="center">
  <img src="https://github.com/user-attachments/assets/de9ddf0e-2fb5-4694-93ec-35c68f85a305" />
</p>

Illustrative examples derived from the various steps of the algorithm used to create the dataset are presented. In sub-figure a, the useful vertices of the mesh are shown, representing the vertices that belong to two distinct segmentation classes. In sub-figure b, the connected components of the useful edges are displayed, i.e., the edges that have at least one useful vertex. Sub-figure c highlights the vertices within the connected components. Finally, in sub-figure d, the resulting keypoints of the mesh are shown.

## Network Architecture & Loss
<p align="center">
  <img src="https://github.com/user-attachments/assets/08f411fb-e597-48d5-a108-c7d5ffa9d4eb" />
</p>

The proposed architecture is a straightforward convolutional neural network (CNN) designed for **3D keypoint prediction from mesh edge features**. The network takes a tensor of edge features as input and outputs the 3D coordinates for 12 keypoints. To address the unordered nature of keypoints, the **Sum of Distances Loss with Hungarian Matching** is employed in the loss function.

## Results

<p align="center">
  <img src="https://github.com/user-attachments/assets/a3d25e42-1178-4c3a-90a9-114c76bb69f5" />
</p>

The model's performance is promising despite the limited dataset size and compact architecture. Notably, the *Percentage of Correct Keypoints* (PCK) with a threshold of tpck​=0.1 is over 84%, indicating high accuracy in keypoint prediction across the test set. Visual results generally show good performance, with predicted keypoints closely matching ground truth across diverse poses. However, some imperfections are observed in more challenging or non-standard postures, such as an less accurate prediction for the right knee in one example.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d4c6335f-f9bf-486a-9092-1987e74784dd" />
</p>

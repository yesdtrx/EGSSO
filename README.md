# EGSSO
Source code for "Entropy-Guided Search Space Optimization for Efficient Neural Network Pruning".
The code implementation of this project references the https://github.com/ultralytics/yolov5 project and the https://github.com/Gumpest/YOLOv5-Multibackbone-Compression project, and we would like to express our gratitude.
Neural network pruning is essential for deploying deep learning models on resource-constrained devices by reducing computational demands and memory usage.
![graphical_abstrast](https://github.com/user-attachments/assets/9d587c60-5582-4187-9a31-689ecaca6c57)
 By improving the subnetwork search algorithm through refining the search space, we enhance the efficiency of the search process and the performance of the subnetworks.
The refined search space focuses the search on more promising regions, thereby reducing computational overhead and improving the quality of the pruned networks. Through iterative optimization, the optimal subnetwork is identified and fine-tuned to achieve the final pruned model.
Experimental results on benchmark datasets demonstrate that our method significantly outperforms other state-of-the-art pruning methods, achieving substantial improvements in both accuracy and computational efficiency.  This entropy-guided pruning strategy offers a robust and effective solution for neural network compression, making it suitable for various deep learning applications.

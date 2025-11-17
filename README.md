# EuroSAT Satellite Image Clustering for Agroecological Assessment

## Introduction
This project demonstrates the application of unsupervised machine learning techniques, specifically cluster analysis, to high-resolution satellite imagery from the EuroSAT dataset. The primary goal is to group images into categories representing different land cover types, which an agricultural management agency can then use for rapid assessment of vulnerable regions following extreme weather events.

## Problem Statement
An agricultural management agency needs to conduct a rapid assessment of land cover status in a vulnerable region, but lacks up-to-date baseline data. They have access to thousands of high-resolution satellite images. The objective is to use cluster analysis to identify land cover types. It is hypothesized that clusters representing "bare soil," "industrial area," or "stagnant water" are indicators of damaged or at-risk agroecological zones.

## Dataset
The dataset used is the **EuroSAT dataset**, sourced from Kaggle. It consists of high-resolution satellite images classified into 10 distinct land cover categories (e.g., AnnualCrop, Forest, River, SeaLake, Industrial, Residential).

Dataset Link: [EuroSAT Dataset on Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

## Step-by-Step Summary

### 1. Data Download and Loading
   - The EuroSAT dataset was downloaded using the `kagglehub` library.
   - Image paths were collected and categorized based on their directory structure, corresponding to the 10 land cover classes.

     <img width="1001" height="340" alt="Screenshot 2025-11-16 at 11 55 10 p m" src="https://github.com/user-attachments/assets/087b840a-e6ed-4aba-91f6-a8165a4ef812" />


### 2. Image Preprocessing
   - **Flattening**: Each 64x64x3 (RGB) image was converted into a 1D vector of 12288 pixel values.
   - **Scaling**: Pixel values (0-255) were scaled to a 0-1 range by dividing by 255.0.

### 3. Dimensionality Reduction (PCA)
   - **Initial PCA**: Performed PCA on the scaled image data to analyze explained variance.
   - **Variance Analysis**: Plotted cumulative explained variance and a scree plot to determine the optimal number of principal components.
   - **Component Selection**: Based on the Kaiser Criterion (eigenvalues > 1), the first 17 principal components were retained, explaining over 75% of the total variance. This reduced the dimensionality from 12288 to 17.
   - **PCA Transformation**: The data was transformed into this 17-dimensional PCA space (`df_images_pca`).

<img width="851" height="545" alt="img1" src="https://github.com/user-attachments/assets/1f217ef9-e8a5-4703-996e-246fe74f4650" />
<img width="847" height="545" alt="img2" src="https://github.com/user-attachments/assets/b98c8d24-f649-4da3-9a63-df4ee7f8565d" />

### 4. Clustering Approaches
   - **K-Means Clustering**: Applied K-Means with varying numbers of clusters (K=2 to 12). The elbow method and silhouette score analysis suggested an optimal `K=3`.
<img width="851" height="545" alt="img3" src="https://github.com/user-attachments/assets/00e3f93f-16b9-4c71-9e40-606f1f3f8785" />
<img width="733" height="507" alt="img4" src="https://github.com/user-attachments/assets/4cc39b5a-9d46-46cf-9e9e-0e0961e2ff79" />
<img width="686" height="507" alt="img5" src="https://github.com/user-attachments/assets/5a2389eb-96e1-419b-b339-4046060e6157" />

   - **DBSCAN Clustering**: Utilized `Optuna` for hyperparameter tuning (`eps` and `min_samples`) to maximize the silhouette score. Despite tuning, DBSCAN identified a large portion of data as noise and yielded suboptimal clustering.
<img width="1215" height="526" alt="img6" src="https://github.com/user-attachments/assets/cb20ce56-caa1-454b-b037-85a245fc8ee6" />

   - **Gaussian Mixture Models (GMM)**: Explored GMM with a range of components. BIC and AIC scores indicated `K=4` as a suitable number of clusters.
<img width="611" height="386" alt="img7" src="https://github.com/user-attachments/assets/b4df2903-9383-487d-bfeb-4c4756efbe2e" />
<img width="1453" height="1177" alt="img8" src="https://github.com/user-attachments/assets/d88dc1b6-e56e-4e4f-ba1a-1aa431f9c403" />
<img width="811" height="699" alt="img9" src="https://github.com/user-attachments/assets/ab23e334-1f3b-497a-9d0e-3aeeba6baa33" />

### 5. Model Evaluation and Profiling (Agroecological Connection)
   - **Internal Validation**: GMM outperformed K-Means and DBSCAN due to its probabilistic approach and adaptability to data distribution.
   - **External Validation**: Compared GMM clusters with the original EuroSAT classes using a confusion matrix and various metrics.

<img width="1060" height="854" alt="img10" src="https://github.com/user-attachments/assets/28bf9bc4-b9e7-4aaa-ba17-8912ef58449a" />

## Results and Conclusions

### Model Selection (GMM)
- K-Means ($K=3$) provided an overly simplistic segmentation.
- DBSCAN struggled to find coherent density structures, classifying most data as noise.
- **GMM ($K=4$) was selected** for its superior performance, adapting well to the elliptical covariance of the data. It achieved a robust semantic separation into four meaningful categories:
    - **Cluster 0 (Soil/Anthropic):** High reflectance zones (bare soil, industrial, urban).
    - **Cluster 1 (Agriculture):** Mixed texture mosaics (crops, pastures).
    - **Cluster 2 (Water):** Dark homogeneous surfaces (rivers, sea).
    - **Cluster 3 (Dense Vegetation):** Dark green zones (forests).

### Model Verdict and Metrics
- **External validation** confirmed the robustness of the GMM model.
- **Normalized Mutual Information (NMI)**: A solid 0.3040, indicating that the clustering retains approximately 30.4% of the structural information from the 10 real classes, despite reducing complexity by 60% (10 classes to 4 clusters).
- **Adjusted Rand Index (ARI)**: A lower score (0.1531) was expected as ARI penalizes grouping different classes, which was an intentional outcome for semantic consolidation.
- **V-Measure**: 0.3040, demonstrating a balance between homogeneity and completeness.
    - **Completeness (0.4142)**: High, indicating the model successfully captured instances of real classes within clusters (e.g., all 'Water' images effectively grouped). 
    - **Homogeneity (0.2401)**: Lower, reflecting the intentional merging of semantically similar true classes into single clusters (e.g., 'Industrial' and 'Residential' into 'Soil/Anthropic').

### Separation of Crops vs. Forests
- The GMM algorithm **successfully created separate clusters for crop fields and forests**.
- GMM was sensitive enough to differentiate the texture patterns of **Dense Vegetation (Cluster 3)** from **Agriculture mosaics (Cluster 1)**. This distinction is crucial for the agricultural agency's use case, enabling independent monitoring of active crop zones versus forest reserves.

### Anomaly Detection Strategy
Based on this segmentation, a state transition alert system is proposed for the agricultural agency:
- **Drought Alert:** Transition from **Group 1 (Crops)** to **Group 0 (Fertile Soil)**.
- **Flood Alert:** Transition from **Group 0 or 1** to **Group 2 (Water)**.
- **Deforestation Alert:** Transition from **Group 3 (Forests)** to **Group 0 or 1**.

This clustering approach provides a valuable tool for rapid agroecological assessment, enabling proactive management and response to environmental changes.

## Author

* **Juan Guillermo Gómez**
* Linkedin: [@jggomezt](https://www.linkedin.com/in/jggomezt/)

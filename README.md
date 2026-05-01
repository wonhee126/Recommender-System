# Recommender System

This repository contains practical exercises and mini-projects related to recommendation systems.



## ✨ Skills Covered

**🛠️ Tech Stack**

- Python / pandas / numpy  
- scikit-learn (CountVectorizer, cosine_similarity) 
- tensorflow / keras 
- matplotlib / seaborn



 **Techniques**

- Day1

  : **From Content-Based Filtering Practice** 

  - Content-Based Filtering 
  - Cosine Similarity 
  - Text Vectorization using CountVectorizer  

- Day2

  : **Fashion Session-based Recommendation Practice**

  -  **30-Day Time Window**: Only the latest 30 days of data were used to filter out outdated trends and focus on current seasonal demand.
  -  **Trend-Centric Filtering**:
     - **Short Sessions (<2 clicks)**: Removed to ensure meaningful intent.
     - **Unpopular Items (<5 clicks)**: Removed to focus on statistically significant fashion trends.

- Day3_Project

  : **MovieLens Session-based Recommendation Practice****

  - 14-Day Test Window**: The final 14 days were used as a test set to ensure a statistically stable evaluation.

  - **Stable Trend Logic**: Because movie popularity doesn't fluctuate daily, a 2-week window provides a more reliable measure of how well the model understands user preference.
  - **Session Parallelism**: Multiple user sessions are processed simultaneously within a single mini-batch to maximize throughput.
  - **Recursive Cleansing**: Removed short sessions and unpopular items repeatedly to ensure data quality.

- Project

  : **MovieLens Deep Learning Recommendation: AutoInt, AutoInt+**

  - **AutoInt vs. AutoInt+**:
    - Both models were implemented and compared.
    - **Result**: **AutoInt showed superior performance** over AutoInt+ in this specific movie dataset, suggesting that the raw attention mechanism was sufficient to capture the underlying patterns without the additional complexity of the 'Plus' variant.

  

  - **Feature Embedding**: High-dimensional categorical data (User ID, Movie ID, Genres, etc.) were compressed into 16-dimensional dense vectors for efficient processing.
  - **Residual Connections**: Used within attention layers to prevent the vanishing gradient problem and allow the model to retain low-order feature information.

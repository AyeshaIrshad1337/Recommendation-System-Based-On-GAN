# Recommendation System by using GANs

This project implements multiple recommendation models for implicit feedback using the MovieLens dataset. The implemented models include Bayesian Personalized Ranking (BPR), Adversarial Personalized Ranking (APR), CollaGAN, and Adversarial Collaborative Autoencoder (ACAE). The implementation includes several evaluation metrics: Precision, Recall, Hit Ratio (HR), Normalized Discounted Cumulative Gain (NDCG), Mean Average Precision (MAP), and F1 Score. The results are saved in a PDF file containing the evaluation metrics and plots.

## File Structure
```
├── NCF/  
│   ├── Neural Collaborative Filtering.ipynb
├── papers/  
│   ├── GAN Based RS.pdf
├── data/  
│   ├── ratings.csv  
│   ├── movies.csv
├── models/  
│   ├── bpr.py  
│   ├── amr.py  
│   ├── collagan.py  
│   ├── acae.py  
│   ├── apr.py  
├── utils/  
│   ├── data_loader.py  
│   ├── evaluation.py  
├── main.py  
├── README.md  
```

## How to Use

To use this project, follow these steps:

1. Make sure you have Python version 3.8 installed on your system.

2. Clone the repository to your local machine:
    ```
    git clone <repository-url>
    ```

3. Navigate to the project directory:
    ```
    cd <foldername>
    ```

4. Install the required dependencies by running the following command:
    ```
    pip install -r requirements.txt
    ```

5. Run the application:
    ```
    python main.py
    ```

## Evaluation Metrics
The implemented evaluation metrics include:

- **Precision@K:** The fraction of relevant items among the top-K recommended items.  
- **Recall@K:** The fraction of relevant items that have been recommended in the top-K results.  
- **Hit Ratio (HR)@K:** Measures if at least one of the top-K items is relevant.  
- **NDCG@K:** Normalized Discounted Cumulative Gain at K, which accounts for the position of the hit by assigning higher scores to hits at top ranks.  
- **MAP@K:** Mean Average Precision at K, which computes the average precision scores for the top-K items.  
- **F1 Score@K:** The harmonic mean of Precision@K and Recall@K.  

## Output
The results, including evaluation metrics and plots for each model, are saved in a PDF file named `model_evaluation_plots.pdf`. The PDF contains:
- Plots for Precision, Recall, NDCG, Hit Ratio, F1 Score, and MAP over epochs for each model.
- Tables summarizing the evaluation metrics for each model on the test dataset.
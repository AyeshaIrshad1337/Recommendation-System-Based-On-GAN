# Bayesian Personalized Ranking with MovieLens Dataset

This project implements Bayesian Personalized Ranking (BPR) for implicit feedback recommendation using the MovieLens dataset. The implementation includes several evaluation metrics: Precision, Recall, Hit Ratio (HR), Normalized Discounted Cumulative Gain (NDCG), Mean Average Precision (MAP), and F1 Score.

## File Structure
├── models/  
│ ├── bpr.py  
├── utils/  
│ ├── data_loader.py  
│ ├── evaluation.py  
├── main.py  
├── README.md  
## How to Use

To use this project, follow these steps:

1. Make sure you have Python version 3.8 installed on your system.

2. Clone the repository to your local machine:
   
3. Navigate to the project directory:
    ```
    cd foldername
    ```

4. Install the required dependencies by running the following command:
    ```
    pip install -r requirements.txt
    ```

5. Run the application:
    ```
    python main.py
    ```

# Evaluation Metrics
The implemented evaluation metrics include:  
  
- **Precision@K:** The fraction of relevant items among the top-K recommended items.  
- **Recall@K:** The fraction of relevant items that have been recommended in the top-K results.  
- **Hit Ratio (HR)@K:** Measures if at least one of the top-K items is relevant.    
- **NDCG@K:** Normalized Discounted Cumulative Gain at K, which accounts for the position of the hit by assigning higher scores to hits at top ranks.  
- **MAP@K:** Mean Average Precision at K, which computes the average precision scores for the top-K items.  
- **F1 Score@K:** The harmonic mean of Precision@K and Recall@K.


from utils.data_loader import load_data, split_data, interaction_matrix
from models.bpr import BPR
from models.amr import BPRModel, train_amr
from models.collagan import CollaGAN
from models.acae import ACAE
from models.apr import APRModel,train_apr  
import numpy as np
import tensorflow as tf
import implicit
from scipy.sparse import csr_matrix
from utils.evaluation import evaluate_implicit_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def convert_to_implicit_format(interactions):
    return csr_matrix(interactions)

def plot_metrics(precisions, recalls, ndcgs, hits, f1_scores, map_scores, model_name, pdf):
    epochs = range(1, len(precisions) + 1)
    
    plt.figure(figsize=(18, 18))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs, precisions, label='Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision over Epochs')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, recalls, label='Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title(f'{model_name} - Recall over Epochs')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, ndcgs, label='NDCG')
    plt.xlabel('Epochs')
    plt.ylabel('NDCG')
    plt.title(f'{model_name} - NDCG over Epochs')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, hits, label='HIT')
    plt.xlabel('Epochs')
    plt.ylabel('HIT')
    plt.title(f'{model_name} - HIT over Epochs')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'{model_name} - F1 Score over Epochs')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs, map_scores, label='MAP')
    plt.xlabel('Epochs')
    plt.ylabel('MAP')
    plt.title(f'{model_name} - MAP over Epochs')
    plt.legend()
    
    plt.tight_layout()
    pdf.savefig() 
    plt.close()  
def add_evaluation_results_table(model_name, evaluation_results, pdf):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('tight')
    ax.axis('off')
    table_data = [
        ["Metric", "Value"],
        ["Precision@10", f"{evaluation_results['precision@10']:.4f}"],
        ["Recall@10", f"{evaluation_results['recall@10']:.4f}"],
        ["Hit Ratio@10", f"{evaluation_results['hit_ratio@10']:.4f}"],
        ["NDCG@10", f"{evaluation_results['ndcg@10']:.4f}"],
        ["MAP@10", f"{evaluation_results['map@10']:.4f}"],
        ["F1 Score@10", f"{evaluation_results['f1_score@10']:.4f}"]
    ]
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    ax.set_title(f'{model_name} - Evaluation Results', fontweight="bold")
    pdf.savefig()
    plt.close()
def evaluate_implicit_model_and_return_results(model, test_interactions):
    precisions, recalls, ndcgs, hit_ratios, f1_scores, maps = evaluate_implicit_model(model,  test_interactions, k=10)
    results = {
        "precision@10": precisions,
        "recall@10": recalls,
        "hit_ratio@10": hit_ratios,
        "ndcg@10": ndcgs,
        "map@10": maps,
        "f1_score@10": f1_scores
    }
    return results    
def main():
    df = load_data('ratings.csv')
    train, test = split_data(df)
    interactions_matrix = interaction_matrix(train)

    n_users, n_items = interactions_matrix.shape

    # Convert interaction matrix to implicit format
    train_interactions = convert_to_implicit_format(interactions_matrix.values)
    test_interactions = convert_to_implicit_format(interaction_matrix(test).values)

    with PdfPages('model_evaluation_plots.pdf') as pdf:
        # Example: Using BPR model from implicit library
        model = implicit.bpr.BayesianPersonalizedRanking(factors=10, iterations=100)
        model.fit(train_interactions)
        # Evaluate BPR model
        evaluation_results = evaluate_implicit_model_and_return_results(model, test_interactions)
        add_evaluation_results_table('BPR Model', evaluation_results, pdf)
        epochs=100
        # Example: Using AMR model
        bpr_model = BPRModel(n_users, n_items, embedding_dim=10)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # Train AMR model
        precisions, recalls, ndcgs, hits, f1_scores, map_scores = train_amr(bpr_model, optimizer, convert_to_implicit_format(interactions_matrix.values), epochs=epochs, batch_size=256)
        plot_metrics(precisions, recalls, ndcgs, hits, f1_scores, map_scores, 'AMR Model', pdf)
        # Evaluate AMR model
        evaluation_results = evaluate_implicit_model_and_return_results(bpr_model, test_interactions)
        add_evaluation_results_table('AMR Model', evaluation_results, pdf)
        
        # Example: Using CollaGAN model
        collagan = CollaGAN(n_users, n_items, embedding_dim=10)
        precisions, recalls, ndcgs, hits, f1_scores, map_scores = collagan.train(convert_to_implicit_format(interactions_matrix.values),  epochs=epochs, batch_size=128)
        plot_metrics(precisions, recalls, ndcgs, hits, f1_scores, map_scores, 'CollaGAN Model', pdf)
        # Evaluate CollaGAN model
        evaluation_results = evaluate_implicit_model_and_return_results(collagan, test_interactions)
        add_evaluation_results_table('CollaGAN Model', evaluation_results, pdf)
        
        # Example: Using ACAE model
        acae = ACAE(n_users, n_items, embedding_dim=10)
        precisions, recalls, ndcgs, hits, f1_scores, map_scores = acae.train(convert_to_implicit_format(interactions_matrix.values), epochs=epochs, batch_size=6)
        plot_metrics(precisions, recalls, ndcgs, hits, f1_scores, map_scores, 'ACAE Model', pdf)
        # Evaluate ACAE model
        evaluation_results = evaluate_implicit_model_and_return_results(acae, test_interactions)
        add_evaluation_results_table('ACAE Model', evaluation_results, pdf)
        
        # Example: Using APR model
        apr_model = APRModel(n_users, n_items, embedding_dim=10)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        print("Its Training")
        # Train APR model
        precisions, recalls, ndcgs, hits, f1_scores, map_scores = train_apr(apr_model, optimizer,convert_to_implicit_format(interactions_matrix.values), epochs=epochs, batch_size=6)
        plot_metrics(precisions, recalls, ndcgs, hits, f1_scores, map_scores, 'APR Model', pdf)
        # Evaluate APR model
        evaluation_results = evaluate_implicit_model_and_return_results(apr_model, test_interactions)
        add_evaluation_results_table('APR Model', evaluation_results, pdf)
if __name__ == "__main__":
    main()
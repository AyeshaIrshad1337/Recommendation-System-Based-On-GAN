import numpy as np

def dcg(relevance_scores):
    return np.sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])

def map(model, test_interaction, k=10):
    aps = []
    for user in range(test_interaction.shape[0]):
        user_interaction = test_interaction[user].nonzero()[1]
        if len(user_interaction) == 0:
            continue
        scores = model.recommend(user, test_interaction, N=k, filter_already_liked_items=False)
        top_k_items = [x[0] for x in scores]
        hits = 0
        precisions = []
        for i, item in enumerate(top_k_items):
            if item in user_interaction:
                hits += 1
                precisions.append(hits / (i + 1))
        if precisions:
            aps.append(np.mean(precisions))
        else:
            aps.append(0)
    return np.mean(aps)

def eval_at_k(model, test_interactions, k=10):
    recalls = []
    precisions = []
    ndcgs = []
    hits = []
    for user in range(test_interactions.shape[0]):
        user_interactions = test_interactions[user].nonzero()[1]
        if len(user_interactions) == 0:
            continue
        scores = model.recommend(user, test_interactions, N=k, filter_already_liked_items=False)
        top_k_items = [x[0] for x in scores]
        hits.append(1.0 if len(set(top_k_items) & set(user_interactions)) > 0 else 0.0)
        precisions.append(len(set(top_k_items) & set(user_interactions)) / k)
        recalls.append(len(set(top_k_items) & set(user_interactions)) / len(user_interactions))
        relevance_scores = [1 if item in user_interactions else 0 for item in top_k_items]
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        if dcg(ideal_relevance_scores) == 0:
            ndcgs.append(0)
        else:
            ndcgs.append(dcg(relevance_scores) / dcg(ideal_relevance_scores))
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs), np.mean(hits)

def f1_score_at_k(model, interactions, k):
    precision, recall , _ ,_= eval_at_k(model, interactions, k)
    
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)
def evaluate_implicit_model(model, interactions, k=10):
    precisions, recalls, ndcgs, hits = eval_at_k(model, interactions, k)
    maps = map(model, interactions, k)
    f1_score = f1_score_at_k(model, interactions, k)
    print(f'Precision@{k}: {precisions}')
    print(f'Recall@{k}: {recalls}')
    print(f'NDCG@{k}: {ndcgs}')
    print(f'Hit Score@{k}: {hits}')
    print(f'F1 Score@{k}: {f1_score}')
    print(f'MAP@{k}: {maps}')
    return precisions, recalls, ndcgs, hits, f1_score, maps


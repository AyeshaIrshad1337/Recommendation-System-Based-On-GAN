import numpy as np

def precision_at_k(model, interactions, k):
    precisions = []
    for user in range(interactions.shape[0]):
        user_interactions = interactions[user].nonzero()[0]
        scores = [(item, model.predict(user, item)) for item in range(interactions.shape[1])]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = [item for item, _ in scores[:k]]
        precisions.append(len(set(top_k) & set(user_interactions)) / k)
    return np.mean(precisions)

def recall_at_k(model, interactions, k):
    recalls = []
    for user in range(interactions.shape[0]):
        user_interactions = interactions[user].nonzero()[0]
        scores = [(item, model.predict(user, item)) for item in range(interactions.shape[1])]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = [item for item, _ in scores[:k]]
        recalls.append(len(set(top_k) & set(user_interactions)) / len(user_interactions))
    return np.mean(recalls)

def hit_ratio_at_k(model, interactions, k):
    hit_ratios = []
    for user in range(interactions.shape[0]):
        user_interactions = interactions[user].nonzero()[0]
        scores = [(item, model.predict(user, item)) for item in range(interactions.shape[1])]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = [item for item, _ in scores[:k]]
        hit_ratios.append(1.0 if len(set(top_k) & set(user_interactions)) > 0 else 0.0)
    return np.mean(hit_ratios)

def ndcg_at_k(model, interactions, k):
    def dcg(relevance_scores):
        return np.sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])
    
    ndcgs = []
    for user in range(interactions.shape[0]):
        user_interactions = interactions[user].nonzero()[0]
        if len(user_interactions) == 0:
            continue
        scores = [(item, model.predict(user, item)) for item in range(interactions.shape[1])]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = [item for item, _ in scores[:k]]
        relevance_scores = [1 if item in user_interactions else 0 for item in top_k]
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        if dcg(ideal_relevance_scores) == 0:
            ndcgs.append(0)
        else:
            ndcgs.append(dcg(relevance_scores) / dcg(ideal_relevance_scores))
    return np.mean(ndcgs)

def map_at_k(model, interactions, k):
    aps = []
    for user in range(interactions.shape[0]):
        user_interactions = interactions[user].nonzero()[0]
        scores = [(item, model.predict(user, item)) for item in range(interactions.shape[1])]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = [item for item, _ in scores[:k]]
        hits = 0
        precisions = []
        for i, item in enumerate(top_k):
            if item in user_interactions:
                hits += 1
                precisions.append(hits / (i + 1))
        if precisions:
            aps.append(np.mean(precisions))
        else:
            aps.append(0)
    return np.mean(aps)


def f1_score_at_k(model, interactions, k):
    precision = precision_at_k(model, interactions, k)
    recall = recall_at_k(model, interactions, k)
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)
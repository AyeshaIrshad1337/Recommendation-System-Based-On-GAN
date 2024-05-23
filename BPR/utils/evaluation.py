import numpy as np
def evaluation(model, interactions , k):
    precisions=[]
    recalls =[]
    hit_ratios=[]
    ndcgs=[]
    aps=[]
    for user in range(interactions.shape[0]):
        user_interactions = interactions[user].nonzero()[0]
        scores =[(item, model.predict(user, item )) for item in range(interactions.shape[1])]
        scores.sort(key=lambda x:x[1], reverse=True)
        top_k = [item for item, _ in scores[:k]]
        precisions.append(len(set(top_k) & set(user_interactions))/k)
        recalls.append(len(set(top_k) & set(user_interactions))/len(user_interactions))
        hit_ratios.append(1.0 if len(set(top_k) & set(user_interactions)) > 0 else 0.0)
        relevance_scores = [1 if item in user_interactions else 0 for item in top_k]
        ideal_relevance_scores = sorted(relevance_scores, reverse=True)
        ndcgs.append(dcg(relevance_scores) / dcg(ideal_relevance_scores))
        hits = 0
        precisions2 = []
        for i, item in enumerate(top_k):
            if item in user_interactions:
                hits += 1
                precisions2.append(hits / (i + 1))
        if precisions2:
            aps.append(np.mean(precisions2))
        else:
            aps.append(0)
    precision=np.mean(precisions)
    recall = np.mean(recalls)
    hit_ratio = np.mean(hit_ratios)
    ndcg = np.mean(ndcgs)
    ap= np.mean(aps)
    
            
    return precision, recall , hit_ratio, ndcg , ap
def dcg(relevance_scores):
        return np.sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])
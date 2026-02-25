import time
import math
from typing import List, Dict
from modules.llm import retrieve_info
from modules.storage import qdrant_ops



def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """Calcola il Reciprocal Rank per una singola query."""
    for rank, ret_id in enumerate(retrieved_ids, start=1):
        if ret_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Calcola la Recall@k per una singola query."""
    retrieved_k = retrieved_ids[:k]
    # Quanti dei documenti rilevanti abbiamo trovato nei primi k?
    hits = sum(1 for gt_id in ground_truth_ids if gt_id in retrieved_k)
    return hits / len(ground_truth_ids) if ground_truth_ids else 0.0

def calculate_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Calcola l'nDCG@k (assumendo rilevanza binaria 1/0)."""
    dcg = 0.0
    idcg = 0.0
    
    # Calcolo DCG (Discounted Cumulative Gain)
    for i, ret_id in enumerate(retrieved_ids[:k]):
        if ret_id in ground_truth_ids:
            # Formula standard: (2^rel - 1) / log2(rank + 1). Se rel=1, 2^1 - 1 = 1
            dcg += 1.0 / math.log2(i + 2) 

    # Calcolo IDCG (Ideal DCG - ovvero il ranking perfetto)
    ideal_hits = min(len(ground_truth_ids), k)
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def calculate_hit_rate(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Restituisce 1.0 se almeno un documento rilevante è nei top K, altrimenti 0.0"""
    retrieved_k = retrieved_ids[:k]
    for gt_id in ground_truth_ids:
        if gt_id in retrieved_k:
            return 1.0
    return 0.0


def run_evaluation(eval_dataset: List[Dict], retrieve_func, k: int = 5):
    """
    Esegue l'eval quantitativo.
    `eval_dataset`: Lista di dizionari [{'query': "...", 'gt_ids': ["id1", "id2"]}]
    `retrieve_func`: Funzione del tuo RAG che prende una query testuale e ritorna i top K IDs.
    """
    results = {
        "mrr": [],
        f"recall@{k}": [],
        f"ndcg@{k}": [],
        f"hit_rate@{k}": [],
        "latency_sec": []
    }
    
    print(f"🚀 Inizio valutazione su {len(eval_dataset)} query (k={k})...")
    
    for item in eval_dataset:
        query = item['query']
        gt_ids = item['gt_ids']
        

        start_time = time.time()
        

        retrieved_ids = retrieve_func(query, top_k=k)

        print(f"\n❓ Query: {query}")
        print(f"✅ Atteso (GT): {gt_ids}")
        print(f"🤖 Trovati:     {retrieved_ids}") 
        
        latency = time.time() - start_time
        
        # Calcolo Metriche
        results["mrr"].append(calculate_mrr(retrieved_ids, gt_ids))
        results[f"recall@{k}"].append(calculate_recall_at_k(retrieved_ids, gt_ids, k))
        results[f"ndcg@{k}"].append(calculate_ndcg_at_k(retrieved_ids, gt_ids, k))
        results[f"hit_rate@{k}"].append(calculate_hit_rate(retrieved_ids, gt_ids, k))
        results["latency_sec"].append(latency)

    # Aggregazione finale (Media)
    metrics_summary = {
        "MRR": sum(results["mrr"]) / len(results["mrr"]),
        f"Recall@{k}": sum(results[f"recall@{k}"]) / len(results[f"recall@{k}"]),
        f"nDCG@{k}": sum(results[f"ndcg@{k}"]) / len(results[f"ndcg@{k}"]),
        f"HitRate@{k}": sum(results[f"hit_rate@{k}"]) / len(results[f"hit_rate@{k}"]),
        "Average Latency (s)": sum(results["latency_sec"]) / len(results["latency_sec"])
    }
    
    return metrics_summary


if __name__ == "__main__":

    dataset = [
        {
            "query": "Per cosa sta DAFS?", 
            "gt_ids": ["6aca4e31-4a08-ea8c-51a8-2a37f13e372f", "05c546fd-835b-9b17-c8df-d652219c2b8d", "2b5822f2-5c89-cb9e-e28a-2522b8a136d6", "4d2e064f-3d7e-ad60-da05-d4da4aa8d459", "496941ed-6586-0cf2-c160-bd9976f4ea6f", "3f71c29f-1376-2c06-539d-f7fb1dc3732f", "3c2d1429-a108-ca29-26bc-7ccddbda3c38" ] 
        },
        {
            "query": "In the HSI client what did it change with regards to the previous version?", 
            "gt_ids": ["4206a36c-1148-a126-5f90-a7452264ffa5", "df8afb7d-5ed5-c527-01d5-3b7fe76e990a", "7e39b947-9409-47a1-b1e3-3ef48ab8d0d6", "594cae11-2667-b2df-f459-687ddbc41b41"]
        },
        {
            "query": "What is the difference between the two options for how PBRDSK groups data and flushes it to file?",
            "gt_ids": ["00c1187e-b1fd-1aeb-b242-67e1d3c059c5", "835a5d73-5ebc-2984-a61c-d6ad4ed8f66a"]
        },
        {
            "query": "What are the 3 core differences in ApmsRtDIP compared to the previous approach?",
            "gt_ids": ["72be81f0-13d7-b7eb-a2fc-c7fc6126f634", "e801ab01-df92-2523-7711-e5a5c1917365", "55c70526-b8c9-4a8e-5d38-6dca0dcdbc57", "95b2629a-9470-9c41-b81c-9ac99cb6f655"]
        },
        {
            "query" : "What is the difference between 2.X e 3.X?",
            "gt_ids": ["f00506a2-20d6-9405-c837-806bb2b77e5d", "e6647ddc-6272-b078-c409-aedf868b6470", "d4e40f92-aefc-e3be-06f8-fe03ca42e6d8", "a06bcba6-a450-db95-25a8-c7d0bff49ca4", "4db2dbec-8740-7e4a-0854-29e673c6d38f", "895b4822-dc36-7c51-1054-d984f010aa22"]
        },
        {
            "query" : "How is the playback structure of HSI server?",
            "gt_ids": ["00c1187e-b1fd-1aeb-b242-67e1d3c059c5", "835a5d73-5ebc-2984-a61c-d6ad4ed8f66a", "3401343b-2bf3-4107-eb24-aa8cf7a60487", "3d4124a0-7c39-92f3-681b-b090bd6d8946", "4918f26f-0138-48d0-f2fa-9480ab578d30", "2f983d5c-e54f-fb21-22c2-c9691df66cb7", "cee0d36c-14d8-b70b-b8e5-dd62a47370f3"]

        },
        {
            "query": "How many packets are inside a record?",
            "gt_ids": ["2f983d5c-e54f-fb21-22c2-c9691df66cb7", "ac9bde89-d478-2416-b554-9369833db462", "cee0d36c-14d8-b70b-b8e5-dd62a47370f3"]
        },
        {
            "query": "What does the power explorer do?",
            "gt_ids": ["86ab204f-771e-2542-e33f-1a37368fa5b7", "f2f2f92b-22d3-df31-98a6-1c998a7c4e20", "0ef96ab8-8f9a-d0e9-d561-99150eac5bb7", "371e990f-a4ef-2d50-a53e-cfe4ee9b9131", "86ab204f-771e-2542-e33f-1a37368fa5b7"]

        },
        {
            "query": "Which methods are necessary for process management?",
            "gt_ids": ["0d219aa3-fff4-7c1c-b8e0-a6803fd82b72", "4a29c7c5-13b2-f8de-5451-d60a3abdb85f", "ca67b8ff-f779-01b8-3c67-2a21a2185a09", "9d978c74-9b14-02fa-c698-ba4b5ec44322", "0f9d1dcc-8084-af8c-aa8f-2ae8a854dc0e"]

        }
    ]
    
    rag_system = retrieve_info.InfoRetriever(db_wrapper=qdrant_ops.VectorDB(collection_name="abb_video_collection_v2"))

    summary = run_evaluation(dataset, retrieve_func=rag_system.gt_eval_ids, k=5)
    
    print("\n📊 --- RISULTATI VALUTAZIONE QUANTITATIVA ---")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import Preprocess
from kmeans import KMeansClustering


if __name__ == "__main__":
    preprocessor = Preprocess(
        "https://raw.githubusercontent.com/chaitanya-basava/CS6375-004-Assignment-1-data/main/bbchealth.txt",
        ['id', 'datetime', 'tweet']
    )

    sse, exp = [], []

    for exp_idx, k in enumerate([5, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]):
        print(f"running for K value: {k}")

        preprocessed_tweets = preprocessor()
        kmeans = KMeansClustering(n_clusters=k, max_iter=1000000)

        preprocessed_tweets['cluster_id'] = kmeans.fit(preprocessed_tweets['tokens'])

        cluster_names_df = pd.DataFrame({
            'cluster_id': range(len(kmeans.get_centroids())),
            'cluster_name': kmeans.get_centroids()
        })
        cluster_groups = preprocessed_tweets.groupby('cluster_id').size().reset_index(name='count')

        print(pd.merge(cluster_groups, cluster_names_df, on='cluster_id', how="inner", validate="many_to_many"))
        print(f"SSE: {kmeans.sse}")
        print("---"*25)

        exp.append(exp_idx)
        sse.append(kmeans.sse)

    plt.plot(exp, sse)
    plt.show()

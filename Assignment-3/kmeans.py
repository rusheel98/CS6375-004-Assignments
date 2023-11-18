import numpy as np
import pandas as pd

from preprocess import Preprocess


def jaccard_distance(arg1, arg2):
    intersection = len(arg1.intersection(arg2))
    union = len(arg1.union(arg2))
    if union == 0:
        print(arg1, arg2)

    return 1 - intersection / union


class KMeansClustering:
    def __init__(self, n_clusters=2, max_iter=100, random_state=42):
        self.sse = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None

    def initialize_centroids(self, x):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(len(x))
        centroids = [x[idx] for idx in random_idx[:self.n_clusters]]
        return centroids

    def compute_distances(self, x):
        distances = np.zeros((self.n_clusters, len(x)))
        for i, centroid in enumerate(self.centroids):
            for j, k in enumerate(x):
                distances[i, j] = jaccard_distance(set(centroid), set(k))
        return distances

    def update_centroids(self, x, _labels):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster = [x[j] for j in range(len(x)) if _labels[j] == i]
            if cluster:
                new_centroid = set(max(cluster, key=cluster.count))
                if len(new_centroid) == 0:
                    print(cluster)
                new_centroids.append(new_centroid)
        return new_centroids

    def fit(self, x):
        self.centroids = self.initialize_centroids(x)
        _labels = np.array([])

        for _ in range(self.max_iter):
            distances = self.compute_distances(x)
            _labels = np.argmin(distances, axis=0)
            new_centroids = self.update_centroids(x, _labels)
            if all(new_centroid == centroid for new_centroid, centroid in zip(new_centroids, self.centroids)):
                break
            self.centroids = new_centroids

        self.sse = self.compute_sse(x, _labels)

        return _labels

    def predict(self, x):
        distances = [jaccard_distance(set(centroid), set(x)) for centroid in self.centroids]
        cluster = np.argmin(distances, axis=0)
        return cluster

    def get_centroids(self):
        return self.centroids

    def compute_sse(self, x, _labels):
        sse = 0.0
        for i in range(len(self.centroids)):
            cluster_points = [x[j] for j in range(len(x)) if _labels[j] == i]
            for point in cluster_points:
                sse += jaccard_distance(set(self.centroids[i]), set(point)) ** 2
        return sse


if __name__ == "__main__":
    preprocessor = Preprocess(
        "https://raw.githubusercontent.com/chaitanya-basava/CS6375-004-Assignment-1-data/main/bbchealth.txt",
        ['id', 'datetime', 'tweet']
    )
    preprocessed_tweets = preprocessor()[:100]

    kmeans = KMeansClustering(n_clusters=10, max_iter=100)

    preprocessed_tweets['cluster_id'] = kmeans.fit(preprocessed_tweets['tokens'])

    cluster_names_df = pd.DataFrame({
        'cluster_id': range(len(kmeans.get_centroids())),
        'cluster_name': kmeans.get_centroids()
    })
    cluster_groups = preprocessed_tweets.groupby('cluster_id').size().reset_index(name='count')

    print(pd.merge(cluster_groups, cluster_names_df, on='cluster_id', how="inner", validate="many_to_many"))
    print(kmeans.sse)

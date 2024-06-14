import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import math
class Cluster:
    def __init__(self, data_pairs: list, clusters: int = 0):
        self.data_pairs = data_pairs
        self.reduced_features = None
        self.clusters = clusters

    def get_embedding(self, embedder) -> None:
        embedding = embedder.encode()
        self.reduced_features = embedding

    def pca(self) -> None:
        pca = PCA(n_components=len(self.data_pairs))
        self.reduced_features = pca.fit_transform(self.reduced_features)

    def find_best_clusters(self) -> None:
        max_clusters = 20
        max_silhouette = -1
        best_clusters = 0

        for n_clusters in range(2, max_clusters + 1):
            if n_clusters > len(self.data_pairs) - 1:
                break
            kmeans = KMeans(n_clusters=n_clusters, random_state=11)
            clusters = kmeans.fit_predict(self.reduced_features)
            silhouette = silhouette_score(self.reduced_features, clusters)
            
            print(f"Number of clusters: {n_clusters}, Silhouette score: {silhouette}")
            
            if silhouette > max_silhouette:
                max_silhouette = silhouette
                best_clusters = n_clusters
        self.clusters = best_clusters
        print(f"最佳聚类数目为: {best_clusters}, 最大轮廓系数: {max_silhouette}")
        print(f"另一种计算最佳聚类数目：",magic(max_clusters))
    def fit(self) -> None:
        kmeans_best = KMeans(n_clusters=self.clusters, random_state=42)
        clusters = kmeans_best.fit_predict(self.reduced_features)
        for i, sentence_pair in enumerate(self.data_pairs):
            print(f"Sentence pair: {sentence_pair}\nCluster label: {clusters[i]}")

class Embedding:
    def __init__(self, data_pairs: list):
        self.data_pairs = data_pairs

    def encode(self) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses")

class BGEEmbedding(Embedding):
    def __init__(self, data_pairs: list):
        super().__init__(data_pairs)

    def encode(self) -> np.ndarray:
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        sentence_features = [
            model.encode(' '.join([item['query'], item['sentence']]), batch_size=12, max_length=8192)['dense_vecs']
            for item in self.data_pairs
        ]
        return sentence_features

class MiniLMEmbedding(Embedding):
    def __init__(self, data_pairs: list):
        super().__init__(data_pairs)

    def encode(self) -> np.ndarray:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        sentence_features = model.encode([' '.join([item['query'], item['sentence']]) for item in self.data_pairs])
        return sentence_features

def magic(num: float) -> float:
    return math.floor(math.sqrt(num / 2) % 1e3)

def main() -> None:
    #chatgpt生成测试数据
    data_pairs = [
    {"query": "Latest news on global warming", "sentence": "Scientists have reported a significant rise in global temperatures, highlighting the urgent need for climate action."},
    {"query": "Recent advancements in AI technology", "sentence": "AI algorithms are now capable of diagnosing diseases with greater accuracy than human doctors."},
    {"query": "Current stock market trends", "sentence": "The stock market has seen unprecedented volatility amid economic uncertainties and geopolitical tensions."},
    {"query": "Results of the World Cup finals", "sentence": "The World Cup finals ended in a thrilling match, with the underdog team clinching victory in the final minutes."},
    {"query": "New policies on renewable energy", "sentence": "Governments worldwide are increasing investments in renewable energy to combat climate change and reduce dependence on fossil fuels."},
    {"query": "Breakthroughs in cancer research", "sentence": "Researchers have developed a new treatment that shows promise in significantly reducing tumor sizes in patients."},
    {"query": "Impact of social media on mental health", "sentence": "Studies indicate a strong correlation between excessive social media use and increased levels of anxiety and depression among teenagers."},
    {"query": "Technological innovations in agriculture", "sentence": "Precision farming and the use of drones are revolutionizing the agricultural sector, leading to higher yields and reduced waste."},
    {"query": "Highlights of the NBA finals", "sentence": "The NBA finals were marked by outstanding performances, with several players breaking previous records."},
    {"query": "Advances in renewable energy storage", "sentence": "New battery technologies are enabling longer storage times for renewable energy, making it more reliable and accessible."},
    {"query": "Economic impact of the COVID-19 pandemic", "sentence": "The global economy is gradually recovering from the pandemic, but challenges such as supply chain disruptions and inflation remain."},
    {"query": "Ethical considerations in AI development", "sentence": "As AI systems become more advanced, ensuring ethical use and preventing biases in algorithms have become critical issues."},
    {"query": "Major transfers in the football transfer window", "sentence": "This year's football transfer window saw some surprising moves, with top players switching clubs for record-breaking fees."},
    {"query": "Climate change and its effects on polar regions", "sentence": "Melting ice caps and rising sea levels are some of the most visible impacts of climate change on the polar regions."},
    {"query": "Innovations in electric vehicle technology", "sentence": "Electric vehicles are becoming more efficient and affordable, with new models offering longer ranges and faster charging times."},
    {"query": "Developments in quantum computing", "sentence": "Quantum computing is poised to revolutionize various industries by solving complex problems much faster than traditional computers."},
    {"query": "Strategies for improving mental health in the workplace", "sentence": "Companies are increasingly implementing mental health programs to support their employees and improve overall productivity."},
    {"query": "Olympic Games results and records", "sentence": "The recent Olympic Games were notable for several record-breaking performances across various sports."},
    {"query": "Role of AI in healthcare", "sentence": "AI is transforming healthcare by enabling personalized medicine and improving diagnostic accuracy through data analysis."},
    {"query": "Efforts to combat plastic pollution", "sentence": "Initiatives to reduce plastic waste, such as banning single-use plastics and promoting recycling, are gaining momentum globally."}
]
    test = Cluster(data_pairs)
    test.get_embedding(BGEEmbedding(data_pairs))
    test.pca()
    test.find_best_clusters()
    test.fit()


main()
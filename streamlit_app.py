import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

def clustering_results():
    # Load dataset

    url = 'https://raw.githubusercontent.com/JasonObiso/Clustering-Mini-Project/main/Clustering-Mini-Project/obesity_estimation_data.csv'
    data = pd.read_csv(url)
    X_scaled = data.drop(columns=['NObeyesdad'])  # Assuming 'NObeyesdad' is the target column

    # Determine optimal number of clusters
    inertia = []
    silhouette_scores = []
    max_clusters = 10
    for k in range(2, max_clusters + 1):
        model = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
        model.fit(X_scaled)
        inertia.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, model.labels_))

    # Plot the elbow method graph
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(2, max_clusters + 1), inertia, marker='o')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    st.pyplot(fig1)

    # Explanation for Elbow Method graph
    st.write("""
    The Elbow Method graph helps in determining the optimal number of clusters for the dataset. 
    It shows the relationship between the number of clusters and the within-cluster sum of squares (inertia). 
    The optimal number of clusters is often associated with the "elbow point," where the rate of decrease in inertia sharply changes.
    """)

    # Plot the silhouette score graph
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score')
    st.pyplot(fig2)

    # Explanation for Silhouette Score graph
    st.write("""
    The Silhouette Score graph provides insights into the quality of clustering by measuring how similar 
    an object is to its own cluster compared to other clusters. A higher silhouette score indicates better 
    separation between clusters. The optimal number of clusters is often associated with the highest average 
    silhouette score.
    """)

    # Determine optimal number of clusters based on known obesity levels and silhouette score
    num_clusters = 7  
    final_model = KMeans(n_clusters=num_clusters, n_init=10, max_iter=300, random_state=42)
    final_model.fit(X_scaled)

    # Visualize clusters using PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    fig3, ax3 = plt.subplots()
    for i in range(num_clusters):
        ax3.scatter(X_pca[final_model.labels_ == i, 0], X_pca[final_model.labels_ == i, 1], label=f'Cluster {i + 1}')
    ax3.set_title('Clusters Visualized using PCA')
    ax3.set_xlabel('Principal Component 1')
    ax3.set_ylabel('Principal Component 2')
    st.pyplot(fig3)

    # Explanation for PCA
    with st.expander("What is PCA?"):
        st.write("""
        Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data 
        into a lower-dimensional space while preserving the most important information. It achieves this by finding the 
        principal components, which are orthogonal vectors that represent the directions of maximum variance in the data. 
        PCA is commonly used for visualization and data compression.
        """)

    # Reduce dimensionality using t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    for i in range(num_clusters):
        ax4.scatter(X_tsne[final_model.labels_ == i, 0], X_tsne[final_model.labels_ == i, 1], label=f'Cluster {i + 1}')
    ax4.set_title('Clusters Visualized using t-SNE')
    ax4.set_xlabel('t-SNE Component 1')
    ax4.set_ylabel('t-SNE Component 2')
    st.pyplot(fig4)

    # Explanation for t-SNE
    with st.expander("What is t-SNE?"):
        st.write("""
        t-Distributed Stochastic Neighbor Embedding (t-SNE) is a nonlinear dimensionality reduction technique 
        particularly well-suited for embedding high-dimensional data into low-dimensional space for visualization. 
        It aims to preserve the local structure of the data by modeling pairwise similarities between data points 
        in the high-dimensional space and their low-dimensional counterparts. t-SNE is commonly used to visualize 
        clusters or groups of data points in a lower-dimensional space.
        """)

    # Print counts of points in each cluster
    unique_labels, label_counts = np.unique(final_model.labels_, return_counts=True)
    for label, count in zip(unique_labels, label_counts):
        st.write(f"Cluster {label + 1}: {count} points")

    # Count occurrences of each unique value in the 'NObeyesdad' column
    value_counts = data['NObeyesdad'].value_counts()

    # Create bar plot
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    value_counts.plot(kind='bar', ax=ax5)
    ax5.set_xlabel('Obesity Levels')
    ax5.set_ylabel('Count')
    ax5.set_title('Number of entries per Obesity Level')
    ax5.tick_params(axis='x', rotation=180) 
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    st.pyplot(fig5)

    # Evaluate the model
    inertia = final_model.inertia_
    silhouette_avg = silhouette_score(X_scaled, final_model.labels_)
    silhouette_values = silhouette_samples(X_scaled, final_model.labels_)

    st.write(f"Inertia: {inertia}")
    st.write(f"Average Silhouette Score: {silhouette_avg}")

    # Plot silhouette scores for each sample
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(num_clusters):
        ith_cluster_silhouette_values = silhouette_values[final_model.labels_ == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / num_clusters)
        ax6.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=0.7)
        ax6.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax6.set_title("Silhouette plot for each cluster")
    ax6.set_xlabel("Silhouette score")
    ax6.set_ylabel("Cluster label")
    st.pyplot(fig6)

    # Explanation for Silhouette Plot
    st.write("""
    The Silhouette plot provides a visual representation of silhouette scores for each sample in each cluster. 
    Silhouette scores measure how similar an object is to its own cluster compared to other clusters. 
    Higher silhouette scores indicate better separation between clusters. 
    In this plot, each bar represents the silhouette score for a single data point. 
    The silhouette scores are sorted within each cluster and represented by different colors.
    """)

    # Members
    st.markdown("<h4 style='text-align: center;'><br>Team Tea Members:</h4>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Glyke Kinney P. Alivio | Chlarence Callelero | James Epijul Costan | Jason Carlo O. Obiso</p>", unsafe_allow_html=True)

def main():
    st.markdown("<h1 style='text-align: center;'>Obesity Risk Assessment Model</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Clustering App Mini Project Results | Team Tea</p>", unsafe_allow_html=True)
    clustering_results()

if __name__ == "__main__":
    main()

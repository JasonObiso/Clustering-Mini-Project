import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def main():
    st.title('Classification/Clustering App')

    # Load your dataset or generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    df = pd.DataFrame(X, columns=['X', 'Y'])

    # Display the data
    st.subheader('Data:')
    st.write(df)

    # Add sidebar widgets for model parameters
    k = st.sidebar.slider('Number of clusters (k)', min_value=2, max_value=10, value=4)

    # Train the clustering model
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    df['Cluster'] = kmeans.labels_

    # Display the clustered data
    st.subheader('Clustered Data:')
    st.write(df)

if __name__ == '__main__':
    main()

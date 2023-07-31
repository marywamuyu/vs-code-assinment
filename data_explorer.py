import streamlit as st
st.title("mary wamuyu mini project")

import pandas as pd
def load_iris_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    data = pd.read_csv(url, header=None, names=column_names)
    return data

data = load_iris_data()

st.title("Iris Dataset Explorer")
st.write("Explore the Iris dataset by answering the following questions:")
if st.checkbox("Show raw data"):
    st.subheader("Iris Data")
    st.dataframe(data)
average_sepal_length_by_species = data.groupby("species")["sepal_length"].mean()
st.subheader("Average Sepal Length by Species")
st.write(average_sepal_length_by_species)
import plotly.express as px

st.subheader("Scatter Plot: Compare Two Features")
x_feature = st.selectbox("Select X-axis feature:", data.columns[:-1])
y_feature = st.selectbox("Select Y-axis feature:", data.columns[:-1])
scatter_plot = px.scatter(data, x=x_feature, y=y_feature, color="species", title="Scatter Plot")
st.plotly_chart(scatter_plot)
st.subheader("Filter Data by Species")
selected_species = st.multiselect("Select species to filter:", data["species"].unique())
filtered_data = data[data["species"].isin(selected_species)]
st.subheader("Filtered Data")
st.dataframe(filtered_data)
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("Pair Plot for Selected Species")
if len(selected_species) > 1:
    sns.pairplot(filtered_data, hue="species", palette="Set1")
      
else:
     sns.pairplot(data, hue="species")
st.pyplot()
st.subheader("Distribution of Selected Features")
selected_feature = st.selectbox("Select a feature to display its distribution:", data.columns[:-1])

plt.figure(figsize=(8, 6))
sns.histplot(data=filtered_data, x=selected_feature, hue="species", kde=True, palette="Set1")
plt.xlabel(selected_feature)
plt.title(f"Distribution of {selected_feature}")
st.pyplot()


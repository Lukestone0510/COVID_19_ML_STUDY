import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('CovidDataset.csv')
df = df.replace({'Yes': 1, 'No': 0})
df

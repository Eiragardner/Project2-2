import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

class dataPlots:
    def __init__(self, data):
        self.data = data
        
    def pricePlotOriginalSet(self):
        plt.figure(figsize=(10,6))
        plt.hist(self.data['Price'], bins=50, edgecolor='black')
        plt.title("Distribution of House Prices")
        plt.xlabel("Price")
        plt.ylabel("Number of Listings")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def priceBoxPlotOriginalSet(self):
        plt.figure(figsize=(10, 2))
        sns.boxplot(self.data["Price"])
        plt.title("Boxplot of House Prices")
        plt.xlabel("Price")
        plt.tight_layout()
        plt.show()
    
    def qqPlot(self, column='Price'):
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        values = self.data[column].dropna()

        plt.figure(figsize=(6, 6))
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {column}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


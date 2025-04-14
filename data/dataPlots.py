import matplotlib.pyplot as plt
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

import pandas as pd
from data.dataOptimization import dataOptimization
from data.dataPlots import dataPlots

class Main:
    def __init__(self, filepath='prepared_data.csv'):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        # Create optimizer and plotter objects
        self.optimizer = dataOptimization(self.data)
        self.plotter = dataPlots(self.data)

    def run(self):
        # Example usage of your classes
        self.optimizer.optimize(max_remove=336)
        #self.optimizer.train_and_evaluate()
       # self.plotter.pricePlotOriginalSet()
       # self.plotter.priceBoxPlotOriginalSet()

if __name__ == "__main__":
    app = Main()
    app.run()
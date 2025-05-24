import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style("darkgrid")


def plot_from_csv(filepath):

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(filepath)

    # Define the bin edges
    bins = [40, 50, 60, 70, 80, 90, 100]

    # Create the histogram using Seaborn
    sns.set(style='darkgrid')
    sns.histplot(data=df, x='Exam Scores', bins=bins, kde=False)

    # Set the x-axis tick labels to show the bin ranges
    plt.xticks(bins, [f'[{bins[i]}-{bins[i + 1]})' for i in range(len(bins) - 1)] + [f'[{bins[-1]}+]'])

    # Add axis labels and a plot title
    plt.xlabel('Exam Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of Exam Scores')

    # Show the plot
    plt.show()


def get_histogram_frequency_counts(filepath):
    # read the csv file into a pandas dataframe
    df = pd.read_csv(filepath)

    # define the bin ranges
    bins = [40, 50, 60, 70, 80, 90, 100]

    # create the histogram
    hist, bins = np.histogram(df['Exam Scores'], bins=bins)

    # print the counts for each bin range
    for i in range(len(hist)):
        print(f"{bins[i]}-{bins[i + 1]}: {hist[i]}")


if __name__ == "__main__":
    csvfile = "CSS practical scores.csv"
    # plot_from_csv(csvfile)
    get_histogram_frequency_counts(csvfile)

import matplotlib.pyplot as plt

def plot_fig(data, cp_position, file_name):
    """
    Plots data with specified positions marked with red vertical lines.
    
    Args:
        data (list or pd.Series): The dataset to be plotted.
        position (list of int): Indices in the data that should be highlighted with red lines.
        file_name (str): The path and filename where the plot should be saved as an image.
    """
    
    # Create a figure and an axes object
    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot the data
    ax.plot(data, label='Data Points', color='blue')

    # Draw red vertical lines at specified positions
    for pos in cp_position:
        ax.axvline(x=pos, color='red', linestyle='--', label='Change Point' if pos == cp_position[0] else "")

    # Add a legend to the plot. Include the red line label only once.
    ax.legend()

    # Set the title and labels of the plot
    ax.set_title('Data Plot with Change Points')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    # Save the plot to a file
    plt.savefig(file_name)

 

# Select numerical columns (replace or modify with your own column selection logic)
numerical_columns = df.select_dtypes(include=["number"]).columns

# Check if there are any numerical columns
if numerical_columns.empty:
    print("No numerical columns to plot!")
else:
    # Number of columns for subplots
    n_cols = len(numerical_columns)
    n_rows = 1  # Adjust this based on how many columns you have

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, 5))

    # If only one column, axes is a single axis, so we make it iterable
    if n_cols == 1:
        axes = [axes]

    # Iterate over numerical columns and plot the barplot
    for ax, col in zip(axes, numerical_columns):
        sns.barplot(hue='COVID-19', y=col, data=df, ax=ax)
        ax.set_title(f'{col} by COVID-19 Status')
        ax.set_xlabel('COVID-19 Status')
        ax.set_ylabel(f'{col}')

    plt.tight_layout()
    plt.show()

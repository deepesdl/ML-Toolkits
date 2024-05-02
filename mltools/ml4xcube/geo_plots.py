import geopandas as gpd
import matplotlib.pyplot as plt


def plot_geo_data(
        df, column_to_plot, title='Geographic Plot', label='Geographic Plot [K]', color_map='viridis',
        save_fig=False, file_name='plot.png', fig_size=(15, 10), edge_color='black', base_map='naturalearth_lowres',
        marker='o', vmin=None, vmax=None, ticks=None):
    """
    Plots geographic data from a DataFrame with a base map for context.

    Parameters:
        df (DataFrame): DataFrame containing the latitude and longitude and data to plot.
        column_to_plot (str): Name of the column which contains the data to visualize.
        title (str): Title of the plot.
        label (str): Legend label for the plot.
        color_map (str): Color map to use for the plot.
        save_fig (bool): If True, saves the figure to a file.
        file_name (str): Name of the file to save the plot to, if save_fig is True.
        fig_size (tuple): Size of the figure to create.
        edge_color (str): Color of the edges of the base map.
        marker (str): Marker style.
        vmin (float): Minimum data value that corresponds to the lower limit of the colormap.
        vmax (float): Maximum data value that corresponds to the upper limit of the colormap.
        ticks (list): List of two elements defining the min and max values for color bar ticks.
    """
    plt.ioff()

    # Check if the necessary columns exist in the dataframe
    if 'lon' not in df.columns or 'lat' not in df.columns:
        raise ValueError("DataFrame must contain 'lon' and 'lat' columns")

    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

    legend_kwds = {'shrink': 0.5, 'label': label}
    if ticks and isinstance(ticks, list) and len(ticks) == 2:
        legend_kwds['ticks'] = ticks

    # Load the world map
    world = gpd.read_file(gpd.datasets.get_path(base_map))

    # Plotting
    ax = world.plot(figsize=fig_size, color='white', edgecolor=edge_color)
    gdf.plot(ax=ax, column=column_to_plot, cmap=color_map, legend=True, marker=marker,
             markersize=0.1, legend_kwds=legend_kwds, vmin=vmin, vmax=vmax, rasterized=True)

    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    if save_fig:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()


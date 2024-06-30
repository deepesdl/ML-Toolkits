import geopandas as gpd
import matplotlib.pyplot as plt


def plot_slice(
        df, var_to_plot, xdim, ydim, title='Geographic Plot', label='Geographic Plot [K]', color_map='viridis',
        xlabel='Longitude', ylabel='Latitude', save_fig=False, file_name='plot.png', fig_size=(15, 10),
        edge_color='black', base_map='naturalearth_lowres', marker='o', vmin=None, vmax=None, ticks=None) -> None:
    """
    Plots data cube slice from a DataFrame with an optional base map for context.

    Args:
        df (DataFrame): DataFrame containing the latitude and longitude and data to plot.
        var_to_plot (str): Name of the column which contains the data to visualize.
        xdim (str): Name of the x dimension to plot.
        ydim (str): Name of the y dimension to plot.
        title (str): Title of the plot.
        label (str): Legend label for the plot.
        color_map (str): Color map to use for the plot.
        save_fig (bool): If True, saves the figure to a file.
        file_name (str): Name of the file to save the plot to, if save_fig is True.
        fig_size (tuple): Size of the figure to create.
        edge_color (str): Color of the edges of the base map.
        marker (str): Marker style.
        vmin (Optional[float]): Minimum data value that corresponds to the lower limit of the colormap.
        vmax (Optional[float]): Maximum data value that corresponds to the upper limit of the colormap.
        ticks (Optional[List]): List of two elements defining the min and max values for color bar ticks.

    Returns:
        None
    """
    plt.ioff()

    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[xdim], df[ydim]))

    # Load the world map
    world = gpd.read_file(gpd.datasets.get_path(base_map))

    # Plotting
    fig, ax = plt.subplots(figsize=fig_size)
    world.plot(ax=ax, color='white', edgecolor=edge_color)

    plot = gdf.plot(ax=ax, column=var_to_plot, cmap=color_map, legend=False, marker=marker,
                    markersize=0.1, vmin=vmin, vmax=vmax, rasterized=True)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label(label)
    if ticks and isinstance(ticks, list) and len(ticks) == 2:
        cbar.set_ticks(ticks)

    if save_fig:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

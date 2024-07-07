import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Tuple
from ml4xcube.preprocessing import apply_filter
from scipy.ndimage import binary_erosion, binary_dilation


def plot_slice(
    ds: xr.DataArray, var_to_plot: str, xdim: str, ydim: str, filter_var: str ='land_mask', title: str ='Slice Plot',
    label: str ='Cube Slice', color_map: str ='viridis', xlabel: str ='Longitude', ylabel: str ='Latitude',
    save_fig: bool = False, file_name: str ='plot.png', fig_size: Tuple[int, int] =(15, 10), vmin: float = None,
    vmax: float = None, ticks: List[float] = None
) -> None:
    """
    Plots a slice of data from a DataArray with an optional land mask for context.

    Args:
        ds (xarray.DataArray): DataArray containing the latitude, longitude, and data to plot.
        var_to_plot (str): Name of the variable to visualize.
        xdim (str): Name of the x dimension to plot.
        ydim (str): Name of the y dimension to plot.
        filter_var (str): Name of the variable used as a land mask for filtering. D
        title (str): Title of the plot.
        label (str): Legend label for the plot.
        color_map (str): Color map to use for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_fig (bool): If True, saves the figure to a file. Defaults to False.
        file_name (str): Name of the file to save the plot to, if save_fig is True.
        fig_size (tuple): Size of the figure to create.
        vmin (float): Minimum data value that corresponds to the lower limit of the colormap.
        vmax (float): Maximum data value that corresponds to the upper limit of the colormap.
        ticks (List[float]): List of two elements defining the min and max values for color bar ticks.

    Returns:
        None
    """
    plt.ioff() # Turn off interactive plotting

    # Extract the min and max values for the x and y dimension
    x_min, x_max = ds[xdim].min().item(), ds[xdim].max().item()
    y_min, y_max = ds[ydim].min().item(), ds[ydim].max().item()
    extent = [x_min, x_max, y_min, y_max]

    # Convert the variable to plot to a numpy array and reshape
    ds_dict = {}
    ds_dict[var_to_plot] = ds[var_to_plot].to_numpy()
    original_shape = ds_dict[var_to_plot].shape
    ds_dict[var_to_plot] = ds_dict[var_to_plot].reshape((1, *original_shape))

    # Create the plot
    fig, ax = plt.subplots(figsize=fig_size)

    # Apply the land mask if specified
    if filter_var is not None and filter_var in ds:
        # Convert the land mask to a numpy array and reshape
        ds_dict[filter_var] = ds[filter_var].to_numpy()
        ds_dict[filter_var] = ds_dict[filter_var].reshape((1, *original_shape))

        # Apply the land mask filter
        ds_dict = apply_filter(ds_dict, filter_var, drop_sample=False)

        # Extract the land mask
        mask = ds_dict[filter_var][0]

        # Erode the mask to move the border inward
        eroded_mask = binary_erosion(mask, iterations=1)

        # Calculate the border by subtracting the eroded mask from the original mask
        mask_border = mask & ~eroded_mask

        # Dilate the border to thicken it
        mask_border = binary_dilation(mask_border, iterations=1)

        # Plot the land borders
        ax.imshow(~mask_border, cmap='gray', extent=extent, alpha=0.9)


    # Plot the variable of interest as a heatmap
    c = ax.imshow(ds_dict[var_to_plot][0], cmap=color_map, vmin=vmin, vmax=vmax, extent=extent)

    # Set the title and lables
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add a colorbar with the specified label
    cbar = plt.colorbar(c, ax=ax, shrink=0.5)
    cbar.set_label(label)

    # Set colorbar ticks if specified
    if ticks and isinstance(ticks, list) and len(ticks) == 2:
        cbar.set_ticks(ticks)

    # Save the figure if save_fig is True
    if save_fig:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()
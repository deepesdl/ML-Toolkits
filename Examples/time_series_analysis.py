import torch
import numpy as np
import xarray as xr
import torch.nn.functional as F
from typing import Tuple
from torch import nn
from xcube.core.store import new_data_store
from ml4xcube.training.pytorch import Trainer
from ml4xcube.splits import assign_block_split
from ml4xcube.datasets.multiproc_sampler import MultiProcSampler
from ml4xcube.datasets.pytorch import PTXrDataset, prep_dataloader


class UNet(nn.Module):
    """
    U-Net architecture for image-like input data. The network consists of a
    contracting path (encoder), a bottleneck, and an expanding path (decoder).
    """
    def __init__(self):
        """
        Initialize U-Net layers: Conv2D, BatchNorm2D, and ConvTranspose2D layers for
        contracting, bottleneck, and expanding paths.
        """
        super(UNet, self).__init__()
        # Contracting Path
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Bottleneck
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)

        # Expanding Path
        self.upconv1    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv5      = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn5        = nn.BatchNorm2d(64)
        self.upconv2    = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv6      = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn6        = nn.BatchNorm2d(32)
        self.upconv3    = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv7      = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn7        = nn.BatchNorm2d(16)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        # Contracting Path
        c1 = F.relu(self.bn1(self.conv1(x)))
        p1 = F.max_pool2d(c1, 2)
        c2 = F.relu(self.bn2(self.conv2(p1)))
        p2 = F.max_pool2d(c2, 2)
        c3 = F.relu(self.bn3(self.conv3(p2)))
        p3 = F.max_pool2d(c3, 2)

        # Bottleneck
        b = F.relu(self.bn4(self.conv4(p3)))

        # Expanding Path
        up1     = self.upconv1(b)
        concat1 = torch.cat([up1, c3], dim=1)
        c5      = F.relu(self.bn5(self.conv5(concat1)))
        up2     = self.upconv2(c5)
        concat2 = torch.cat([up2, c2], dim=1)
        c6      = F.relu(self.bn6(self.conv6(concat2)))
        up3     = self.upconv3(c6)
        concat3 = torch.cat([up3, c1], dim=1)
        c7      = F.relu(self.bn7(self.conv7(concat3)))

        # Final Convolution
        out = self.final_conv(c7)
        return out


def map_function(chunk) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a chunk of data into input (X) and target (y) tensors.

    Args:
        chunk (list[dict]): List of dictionaries, where each dictionary contains data arrays
                            for 'air_temperature_2m', 'gross_primary_productivity',
                            'radiation_era5', and 'latent_energy'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            X: Concatenated tensor of input features.
            y: Tensor of target values for air temperature.
    """
    # Extract the first two elements from each sample for the features
    at  = np.concatenate([d['air_temperature_2m'][:, :2, :, :] for d in chunk], axis=0)
    gpp = np.concatenate([d['gross_primary_productivity'][:, :2, :, :] for d in chunk], axis=0)
    re  = np.concatenate([d['radiation_era5'][:, :2, :, :] for d in chunk], axis=0)
    le  = np.concatenate([d['latent_energy'][:, :2, :, :] for d in chunk], axis=0)

    # Concatenate all features along the channel dimension
    X = np.concatenate([at, gpp, re, le], axis=1)

    # Extract the last element of 'air_temperature_2m' as the dependent variable
    y = np.concatenate([d['air_temperature_2m'][:, 2, :, :] for d in chunk], axis=0)

    # Add an extra dimension to y to make it (32, 1, 16, 16)
    y = y[:, np.newaxis, :, :]

    return torch.tensor(X), torch.tensor(y)


def prepare_dataset_creation() -> xr.Dataset:
    """
    Prepare the dataset for creation by fetching the data, calculating statistics, and assigning a land mask.

    Returns:
        xr.Dataset: The prepared dataset.
    """
    data_store = new_data_store("s3", root="esdl-esdc-v2.1.1", storage_options=dict(anon=True))
    dataset    = data_store.open_data('esdc-8d-0.083deg-184x270x270-2.1.1.zarr')
    start_time = "2002-05-21"
    end_time   = "2003-08-01"
    ds         = dataset[["air_temperature_2m", "gross_primary_productivity", "radiation_era5", "latent_energy"]].sel(
        time=slice(start_time, end_time),
        lon=slice(0, 190),
        lat=slice(90, 0)
    )
    print("Dataset Dimensions:", ds.dims)



    # block sampling
    xds = assign_block_split(
        ds         = ds,
        block_size = [("time", 12), ("lat", 135), ("lon", 135)],
        split      = 0.7
    )
    return xds




def main() -> None:
    """
    Main function to prepare the dataset and create training/testing datasets.
    """
    ds = prepare_dataset_creation()

    # Preprocess data and split into training and testing sets
    sampler = MultiProcSampler(
        ds          = ds,
        train_cube  = 'train_cube2.zarr',
        test_cube   = 'test_cube2.zarr',
        sample_size = [('time', 3), ('lat', 16), ('lon', 16)],
        nproc       = 5,
        chunk_size  = (64, 3, 16, 16),
        array_dims  = ('samples', 'time', 'lat', 'lon'),
        drop_nan    = 'if_all_nan',
        fill_method = 'sample_mean',
        data_fraq   = 0.1,
        scale_fn    = None
    )

    train_ds, test_ds = sampler.get_datasets()

    train_set = PTXrDataset(train_ds)
    test_set  = PTXrDataset(test_ds)

    train_loader = prep_dataloader(train_set, callback=map_function)
    test_loader = prep_dataloader(test_set, callback=map_function)

    lr = 0.1
    epochs = 2

    reg_model = UNet()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(reg_model.parameters(), lr=lr, weight_decay=1e-3)
    best_model_path = './Unet_0.1.pth'

    ## Trainer instance
    trainer = Trainer(
        model      = reg_model,
        train_data = train_loader,
        test_data  = test_loader,
        optimizer  = optimizer,
        loss       = mse_loss,
        model_path = best_model_path,
        epochs     = epochs
    )

    ## Start training
    reg_model = trainer.train()


if __name__ == "__main__":
    main()




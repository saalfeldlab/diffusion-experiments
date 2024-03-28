import os
from urllib.parse import urlparse, urlunparse

import mlflow
import yaml
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet, CellMapDatasets3Das2D, CellMapDataset3Das2D

from exp06.config import (
    ExperimentConfig,
    TrackingConfig,
)
from exp06.utility import flatten_dict, get_repo_and_commit_cwd
from typing import Literal, Sequence, Optional
import warnings
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import imageio
import math
import logging
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", module="pydantic_ome_ngff")  # line104

def get_next_sample(existing: Sequence[str], digits=None):
    if len(existing) < 1:
        next_sample=0
        if digits is None:
            digits = 5
            logger.info(f"Number of digits not specified and no strings given to derive from. Defaulting to {digits}.")
    else:
        next_sample = max(int(s) for s in list(existing)) + 1
        if digits is None:
            digits = len(list(existing)[0])            
        elif digits != len(list(existing)[0]):
            raise ValueError(f"Specified to use {digits} digits but found string with {len(list(existing)[0])} digits")
    next_sample_str = "{num:0{digits}d}".format(num=next_sample, digits=digits)
    return next_sample_str

def track(config: TrackingConfig):
    parsed_uri = urlparse(config.tracking_uri)
    if parsed_uri.scheme == "file":
        print(os.path.exists(parsed_uri.path))  # this is just accessing it for autofs
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    if config.continue_run_id is None:
        with mlflow.start_run(run_name=config.run_name) as run:
            return run.info.run_id
    else:
        #    with mlflow.start_run(run_id = config.continue_run_id) as run:
        return config.continue_run_id

def sample_real():
    with open("experiment_config.yaml") as config_file:
        yaml_data = yaml.safe_load(config_file)
    config = ExperimentConfig(**yaml_data)

    run_id = track(config.tracking)
    with mlflow.start_run(run_id=run_id):
        architecture = config.architecture.get_constructor()(**config.architecture.dict())
        diffusion = config.diffusion.get_constructor()(
            architecture, image_size=config.image_size, **config.diffusion.dict()
        )
        data_args = config.data.dict()
        del data_args["data_type"]
        data_args["image_size"] = config.image_size
        data_args["dask_workers"] = 0
        dataset = config.data.get_constructor()(**data_args)
        class_list = data_args["class_list"]
        class_colors =  {
            "ecs": (50,50,50),
            "pm": (100,100,100),
            "mito_mem": (255,128,0),
            "mito_lum": (128,64,0),
            "mito_ribo": (220,172,104),
            "golgi_mem": (0,132,255),
            "golgi_lum": (0,66,128),
            "ves_mem": (255,0,0),
            "ves_lum": (128,0,0),
            "endo_mem": (0,0,255),
            "endo_lum": (0,0,128),
            "lyso_mem": (255,216,0),
            "lyso_lum": (128, 108,0),
            "ld_mem": (134,164,247),
            "ld_lum": (79,66,252),
            "er_mem": (57,215,46),
            "er_lum": (51,128,46),
            "eres_mem": (85,254,219),
            "eres_lum": (6,185,157),
            "ne_mem": (9,128,0),
            "ne_lum": (5,77,0),
            "np_out": (175,249,111),
            "np_in": (252,144,211),
            "hchrom": (168,55,188),
            "nhchrom": (84,23,94),
            "echrom": (204,0,102),
            "nechrom": (102,0,51),
            "nucpl": (255,0,255),
            "nucleo": (247,82,104),
            "mt_out": (255,255,255),
            "mt_in": (128,128,128),}
        class_luts = {k:make_lut(v) for k,v in class_colors.items()}
        parsed_artifact_uri = urlparse(mlflow.get_artifact_uri())
        trainer = Trainer(
            diffusion,
            dataset,
            results_folder=parsed_artifact_uri.path,
            **config.training.dict(),
        )
        a = next(trainer.dl).cpu().detach()
        a = trainer.model.unnormalize(a)
        a = a.mul(255).clamp_(0,255).to(torch.uint8)
        save_path = "/groups/saalfeld/home/heinrichl/material/diffusion_movies/grids4x4/real/"
        if os.path.exists(save_path):
            samples = os.listdir(save_path)
            existing = [int(s[1:]) for s in samples]
            if len(existing) < 1:
                sample_id = 0
            else:
                sample_id = max(existing)+1
        else:
            sample_id = 0
        for s in range(int(a.shape[0]/16)):
            sample_name = f"s{sample_id+s:03d}"
            sample = a[s*16:(s+1)*16,...]
            sample_grids = utils.make_grid(sample, 4).numpy()
            print(sample_grids.shape)
            os.makedirs(f"{save_path}{sample_name}", exist_ok=True)

            result = variant1(sample_grids[:31,...], class_list, class_luts)
            raw = sample_grids[31,...]
            imageio.imwrite(f"{save_path}{sample_name}/labels.png", result)
            imageio.imwrite(f"{save_path}{sample_name}/raw.png", raw)

ChannelOptions = Literal["colorized", "split_channels"]
TimestepReturn = Literal["timeseries", "final_timestep"]

def run(checkpoint: int, num_samples: int, timesteps: TimestepReturn, channels: ChannelOptions, sample_digits:Optional[int]=None):
    """_summary_

    Args:
        checkpoint (int): _description_
        num_samples (int): _description_
        timesteps (TimestepReturn): _description_
        channels (ChannelOptions): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    # Load configuration from YAML file
    with open("experiment_config.yaml") as config_file:
        yaml_data = yaml.safe_load(config_file)
    config = ExperimentConfig(**yaml_data)
    run_id = track(config.tracking)
    
    with mlflow.start_run(run_id=run_id):
        architecture = config.architecture.get_constructor()(**config.architecture.dict())
        diffusion = config.diffusion.get_constructor()(
            architecture, image_size=config.image_size, **config.diffusion.dict()
        )
        data_args = config.data.dict()
        del data_args["data_type"]
        data_args["pre_load"] = False
        data_args["image_size"] = config.image_size
        dataset = config.data.get_constructor()(**data_args)
        parsed_artifact_uri = urlparse(mlflow.get_artifact_uri())
        if parsed_artifact_uri.scheme != "file":
            raise NotImplementedError(
                f"Using a {parsed_artifact_uri.scheme} connection to save artifacts is not implemented"
            )
        trainer = Trainer(
            diffusion,
            dataset,
            results_folder=os.path.join(parsed_artifact_uri.path, "checkpoints"),
            **config.training.dict(),
        )
        trainer.load(checkpoint)

        samples = trainer.ema.ema_model.sample(num_samples, return_all_timesteps=timesteps=="timeseries").cpu().detach()
        print(samples.shape)
        zarr_relative_path = os.path.join("checkpoints", "samples", f"{timesteps}.zarr")
        group= os.path.join(channels, f"grid_{num_samples}")
        samples_zarr = zarr.group(store=zarr.DirectoryStore(os.path.join(parsed_artifact_uri.path, zarr_relative_path)))
        zarr_grp = samples_zarr.require_group(f"{channels}/grid_{num_samples}")
        next_sample = get_next_sample(zarr_grp.keys(), digits=sample_digits)
        print(next_sample)
        sample_grp = zarr_grp.require_group(next_sample)
        if timesteps == "timeseries":
            time_digits = len(str(diffusion.sampling_timesteps))
            for t in range(diffusion.sampling_timesteps+1):
                img_grp = sample_grp.require_group(name="t{time:0{time_digits}d}".format(time=t, time_digits=time_digits))
                sample = samples[:,t, ...]
                #sample_to_zarr(img_grp, samples[:, t, ...], data_args, channels)    
        elif timesteps == "final_timestep":
            img_grp = sample_grp 
            #sample_to_zarr(img_grp, samples, data_args, channels)
        
def sample_to_zarr(zarr_img_grp, sample, data_args, channels):
    samples_per_row = int(math.sqrt(num_samples))
    if len(data_args["class_list"]) > 0:
        label = sample[:len(data_args["class_list"]),...]
        if channels == "colorized":
            label = rgbify(label)
        img_grp.create_dataset("labels", utils.make_grid(label, samples_per_row))
    if data_args["include_raw"]:
        raw = sample[-1,...]
        img_grp.create_dataset("raw", utils.make_grid(raw, sample_per_row))
                
        
        # for t in range(251):
        #     sample_group.create_dataset(name=f"t{t:03d}", data=sample_grids[t])
        # return next_sample
        # samples_zarr.create_dataset(name=f"s{next_sample:03d}", data=sample_grids)
        
        
        
        # for ch in range(32):
        #     print(ch)
        #     print(samples[:,-1,ch, ...].min())
        #     print(samples[:,-1, ch, ...].max())
        # print(type(samples))
        # print(samples.shape)
        # print(samples.dtype)
        # samples_zarr = zarr.group(store=zarr.DirectoryStore(os.path.join(parsed_artifact_uri.path, "timesamples_300.zarr")))
        # for sid, timeseries in enumerate(samples):
        #     sample_group = samples_zarr.create_group(f"s{sid:03d}")
        #     for t, sample in enumerate(timeseries):
        #         sample_group.create_dataset(name=f"t{t:03d}", data=sample)
def make_lut(color, ncolors=256):
    entry = np.linspace(0,1,ncolors)
    lut = []
    for value in entry:
        lut.append(tuple(int(round(value*ch)) for ch in color))
    return np.array(lut).astype(np.uint8)
def make_rgb_timeseries(sample_no):
    with open("experiment_config.yaml") as config_file:
        yaml_data = yaml.safe_load(config_file)
    config = ExperimentConfig(**yaml_data)
    run_id = track(config.tracking)
    data_args = config.data.dict()
    class_list = data_args["class_list"]
    sample_name = f"s{sample_no:03d}"
    with mlflow.start_run(run_id=run_id):
        parsed_artifact_uri = urlparse(mlflow.get_artifact_uri())
    samples_zarr = zarr.group(store=zarr.DirectoryStore(os.path.join(parsed_artifact_uri.path, "timesamplegrids4x4_300.zarr")))
    timeseries = samples_zarr[sample_name]
    
        #print(samples)
    class_colors =  {
            "ecs": (50,50,50),
            "pm": (100,100,100),
            "mito_mem": (255,128,0),
            "mito_lum": (128,64,0),
            "mito_ribo": (220,172,104),
            "golgi_mem": (0,132,255),
            "golgi_lum": (0,66,128),
            "ves_mem": (255,0,0),
            "ves_lum": (128,0,0),
            "endo_mem": (0,0,255),
            "endo_lum": (0,0,128),
            "lyso_mem": (255,216,0),
            "lyso_lum": (128, 108,0),
            "ld_mem": (134,164,247),
            "ld_lum": (79,66,252),
            "er_mem": (57,215,46),
            "er_lum": (51,128,46),
            "eres_mem": (85,254,219),
            "eres_lum": (6,185,157),
            "ne_mem": (9,128,0),
            "ne_lum": (5,77,0),
            "np_out": (175,249,111),
            "np_in": (252,144,211),
            "hchrom": (168,55,188),
            "nhchrom": (84,23,94),
            "echrom": (204,0,102),
            "nechrom": (102,0,51),
            "nucpl": (255,0,255),
            "nucleo": (247,82,104),
            "mt_out": (255,255,255),
            "mt_in": (128,128,128),
    }

    # for cls in class_colors.keys():
    #     class_colors[cls] = tuple(np.array(class_colors[cls])/255.))
    class_luts = {k:make_lut(v) for k,v in class_colors.items()}
    os.makedirs(f"/groups/saalfeld/home/heinrichl/material/diffusion_movies/grids4x4/{sample_name}/labels/", exist_ok=True)
    os.makedirs(f"/groups/saalfeld/home/heinrichl/material/diffusion_movies/grids4x4/{sample_name}/raw/", exist_ok=True)
    for t in range(251):
        sample = timeseries[f"t{t:03}"][:31,...]
        result = variant1(sample, class_list, class_luts)
        raw = timeseries[f"t{t:03}"][31,...]
        imageio.imwrite(f"/groups/saalfeld/home/heinrichl/material/diffusion_movies/grids4x4/{sample_name}/labels/t{t:03}.png", result)
        imageio.imwrite(f"/groups/saalfeld/home/heinrichl/material/diffusion_movies/grids4x4/{sample_name}/raw/t{t:03}.png", raw)
        #plt.imshow(result,vmin=0,vmax=255)
        #plt.savefig(f"/groups/saalfeld/home/heinrichl/material/diffusion_movies/grids/s000/labels/t{t:03}.png")
    

def variant1(sample, class_list, class_luts):
    global_threshold = 10
    # print("BEFORE:",sample.min(), sample.max(), sample.mean())
    sample[sample<=global_threshold] = 0
    # print("AFTER:",sample.min(), sample.max(), sample.mean())
    class_id_array = np.argmax(sample, axis=0, keepdims=True)
    values = np.take_along_axis(sample, class_id_array, axis=0)
    rgb_image = np.zeros((sample.shape[1], sample.shape[2], 3), dtype=np.uint16)
    # print(rgb_image.shape)
    normalizing_image = np.zeros((1, sample.shape[1], sample.shape[2]), dtype=np.uint8)
    
    #normalizing_image = np.zeros_like(sample)
    for class_id, class_name in enumerate(class_list):
        class_bin_arr = class_id_array == class_id
        normalizing_image += class_bin_arr.astype(np.uint8)
        lut = class_luts[class_name]
        this_class_arr = np.copy(values)
        this_class_arr[np.logical_not(class_bin_arr)] = 0
        # print("A", class_bin_arr.shape)
        # print("B", values.shape)
        # print("C", this_class_arr.shape)
        # print("D", np.unique(this_class_arr))
        # print("E", np.unique(values))
        # print("F", np.unique(lut[this_class_arr]))
        # print("G", lut[this_class_arr.squeeze()].shape)
        # print("H", rgb_image.shape)
        rgb_image += lut[this_class_arr.squeeze()]
    # print("I", np.unique(class_id_array))
    #rgb_image= rgb_image/np.stack([normalizing_image,]*3, axis=2)
    return rgb_image.astype(np.uint8)
    
if __name__ == "__main__":
#    sample_no = run()
#    make_rgb_timeseries(sample_no)
    #real()
    run(200, 17, "timeseries", "split_channels")
    #make_rgb_timeseries()

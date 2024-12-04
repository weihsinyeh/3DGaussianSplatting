#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, test, separate_sh, output_dir):
    makedirs(output_dir, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, test=test, separate_sh=separate_sh)["render"]
        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(output_dir, view.image_name))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, test : bool, separate_sh: bool, output_dir: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, test=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not test:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, test, separate_sh, output_dir)

        if test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, test, separate_sh, output_dir)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    # model_path = "./gaussian-splatting/output/f27ffa1f-1" # 0.25
    # model_path = "./gaussian-splatting/output/92057955-b" # random
    # model_path = "./gaussian-splatting/output/d81ea2d1-e"
    # model_path = "/tmp2/r13922043/gaussian-splatting/output/26c8f247-4" #  60000
    # model_path = "/tmp2/r13922043/gaussian-splatting/output/d79f13d7-4" # 100000
    model_path = "./bestmodel"
    
    model       = ModelParams(parser, sentinel=True, model_path=model_path)
    pipeline    = PipelineParams(parser)
    parser.add_argument("--iteration",  default=-1, type=int)
    parser.add_argument("--test",       default=True, type=bool)
    parser.add_argument("--quiet",      action="store_true")
    parser.add_argument("--output_dir", type=str)
    args = get_combined_args(parser)
    print("Rendering " + model_path)
    print("Images will be saved to ",args.output_dir)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.test, SPARSE_ADAM_AVAILABLE, args.output_dir)
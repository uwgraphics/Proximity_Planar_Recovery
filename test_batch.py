"""
Test that the batched version of the differentiable method gives the same output as the 
non-batched version.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from differentiable_method import render
from differentiable_method_batched import render_batched
from util import ZONE_SPEC

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    # example reference histogram
    # the below line tells black formatter not to format this section
    # fmt: off
    reference_hist = [
        6, 3, 3, 4, 5, 5, 5, 3, 3, 3, 8, 26, 203, 1674, 27531, 54617, 32175, 14796, 9119, 6999,
        5506, 4713, 3826, 3344, 2875, 2436, 2110, 1692, 1549, 1343, 1288, 1202, 1077, 1007, 970,
        875, 761, 781, 765, 689, 669, 567, 605, 542, 517, 511, 456, 435, 458, 436, 400, 387, 366,
        367, 313, 326, 308, 299, 286, 253, 275, 248, 246, 219, 228, 221, 201, 189, 210, 162, 170,
        173, 164, 176, 142, 153, 127, 127, 116, 138, 115, 114, 105, 115, 108, 107, 99, 117, 96, 97,
        85, 103, 80, 97, 72, 64, 67, 79, 68, 44, 71, 43, 53, 60, 51, 62, 56, 47, 58, 50, 38, 38, 34,
        31, 37, 29, 33, 36, 21, 29, 28, 31, 18, 20, 29, 22, 15, 23
    ]
    # fmt: on

    forward_params = {
        "surface_albedo": torch.tensor(3.4409680366516113),
        "edge_brightness": torch.tensor(0.5996025204658508),
        "corner_brightness": torch.tensor(0.407014936208725),
        "bin_offset": torch.tensor(9.523497581481934),
        "crosstalk_scale": torch.tensor(0.009766248054802418),
        "specular_weight": torch.tensor(0.14681196212768555),
        "specular_exponent": torch.tensor(1.8378312587738037),
        "dc_offset": torch.tensor(0.0),
        "soft_hist_sigma": torch.tensor(0.5),
        "impulse_x_scale": torch.tensor(0.2788732647895813),
        "impulse_y_offset": torch.tensor(135.14208984375),
        "bin_size": torch.tensor(0.014131013303995132),
        "saturation_point": torch.tensor(315.8522033691406),
    }

    num_test_cases = 200
    test_aois = torch.tensor(np.random.uniform(0, np.pi / 8, num_test_cases))
    test_azimuths = torch.tensor(np.random.uniform(0, np.pi, num_test_cases))
    test_z_dists = torch.tensor(np.random.uniform(0, 1, num_test_cases))

    non_batched_hists = render(
        aoi=test_aois[0],
        azimuth=test_azimuths[0],
        z_dist=test_z_dists[0],
        impulse_response=torch.tensor(reference_hist).to(torch.float64),
        zone_spec=ZONE_SPEC,
        samples_per_zone=256 * 9,
        device=device,
        **forward_params
    )

    batched_forward_params = {k: v.expand(num_test_cases) for k, v in forward_params.items()}

    batched_hists = render_batched(
        aoi=test_aois,
        azimuth=test_azimuths,
        z_dist=test_z_dists,
        impulse_response=torch.tensor(reference_hist).to(torch.float64).unsqueeze(0).expand(num_test_cases, -1),
        zone_spec=ZONE_SPEC,
        samples_per_zone=256 * 9,
        device=device,
        **batched_forward_params
    )[0, :, :]

    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            ax[i, j].plot(non_batched_hists[i * 3 + j].cpu().numpy(), label="non-batched")
            ax[i, j].plot(batched_hists[i * 3 + j].cpu().numpy(), label="batched")
            ax[i, j].legend()
    
    plt.show()


if __name__ == "__main__":
    main()

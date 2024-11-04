import numpy as np
import torch
from tqdm import tqdm

from direct_method import direct_method
from util import ZONE_SPEC, angle_between_vecs


def render(
        aoi,
        azimuth,
        z_dist,
        impulse_response,
        surface_albedo,
        edge_brightness,
        corner_brightness,
        bin_offset,
        bin_size,
        crosstalk_scale,
        specular_weight,
        specular_exponent,
        dc_offset,
        soft_hist_sigma,
        impulse_x_scale,
        impulse_y_offset,
        saturation_point,
        zone_spec,
        samples_per_zone,
        device
    ):
    # convert aoi, azimuth, and z_dist to plane_a and plane_d (ax+d=0 form where d is positive)
    plane_a = torch.tensor([1, 0, 0]) * torch.cos(azimuth) * torch.sin(aoi) + \
        torch.tensor([0, 1, 0]) * torch.sin(azimuth) * torch.sin(aoi) + \
        torch.tensor([0, 0, 1]) * torch.cos(aoi)
    plane_d = plane_a[2] * z_dist

    image = torch.zeros(9, 128)

    gains = torch.tensor([1, 0, 1, 0, 0, 0, 1, 0, 1]) * corner_brightness * surface_albedo + \
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0]) * edge_brightness * surface_albedo + \
            torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0]) * surface_albedo

    # rescale the impulse response
    # first subtract the y offset and clamp any negative values
    impulse_response = impulse_response.to(device)
    impulse_response = impulse_response - impulse_y_offset
    impulse_response = torch.clamp(impulse_response, min=0)
    # then normalize it so that the max is 1
    impulse_response = impulse_response / impulse_response.max()
    # then scale it along the x axis by impulse_x_scale - this requires resampling and interpolating
    scaled_impulse = resample(impulse_response, impulse_x_scale)

    for i, single_zone_spec in enumerate(zone_spec):
        # sample rays from zone
        zone = Zone(single_zone_spec["center_px"], [single_zone_spec["width_px"], single_zone_spec["height_px"]])
        rays = zone.sample_ray_directions(samples_per_zone)

        # render all the rays to get a distance and intensity measure for each ray
        dists, intensities = render_rays(
            torch.from_numpy(rays).to(device),
            plane_a,
            plane_d,
            specular_weight,
            specular_exponent,
            saturation_point,
            gains[i],
            device
        )

        # apply some constant offset to the distances (this should be the same as offsetting the
        # final histogram by some amount?)
        dists = dists + (bin_offset * bin_size)

        # create a histogram of the distances using differentiable soft hist function
        hist_max = bin_size * 128
        sigma = soft_hist_sigma * bin_size
        hist = soft_hist(dists, size=128, vmin=0, vmax=hist_max, weights=intensities, sigma=sigma, device=device)[0]

        # convolve the impulse response - pad the start but cut off the end
        hist = torch.nn.functional.conv1d(
            hist.reshape(1, 1, -1),
            torch.flip(scaled_impulse, dims=[0]).reshape(1, 1, -1),
            padding=len(scaled_impulse)-1
        ).reshape(-1)[:len(hist)] # flatten to 1d and cut off the end

        image[i] = hist

    # apply crosstalk between zones
    image = apply_crosstalk(image, crosstalk_scale)

    # add the dc offset - maybe this should override the function rather than add to it?
    # e.g. maybe it should be hist = max(dc_offset, hist)
    image = image + dc_offset

    return image

def render_rays(rays, plane_a, plane_d, specular_weight, specular_exponent, saturation_point, gain, device):
    """
    The first dimension of each input tensor is the number of rays
    """
    num_rays = rays.shape[0]
    plane_as = plane_a.repeat(num_rays, 1).to(device)
    plane_ds = plane_d.repeat(num_rays, 1).to(device)

    pts = intersect_lines_planes(
        torch.zeros(num_rays, 1).to(device),
        rays,
        plane_as,
        plane_ds
    )

    # because the camera is at the origin, the distance to the point is its norm
    dists = torch.norm(pts, dim=1)

    # intensity falloff is inverse square
    incident_light = 1 / dists**2
    # measured_light = 1 - torch.exp(-incident_light * saturation_point)
    measured_light = saturation_point * (1 - torch.exp(-(incident_light * gain / saturation_point)))

    # phong lighting model where camera and light are co-located
    # diffuse weight is assumed to be inverse of specular weight so that the weights don't fight the
    # camera gain during optimization. Because plane_d is always positive in our convention, 
    # plane_a will always be the normal for the wrong side of the plane, so take the opposite
    diffuse_term = torch.sum(-rays * -plane_as, dim=1) * (1 - specular_weight)

    r_hat = 2*torch.sum(-rays * -plane_as, dim=1).unsqueeze(1) * -plane_as + rays
    specular_dotprod = torch.sum(-rays * r_hat, dim=1)
    # if the specular dot product is negative, set the specular term to zero
    specular_dotprod = torch.where(specular_dotprod < 0, torch.tensor(0.0).to(device), specular_dotprod)
    specular_term = specular_dotprod ** specular_exponent * specular_weight

    intensities = measured_light * (diffuse_term + specular_term)

    return dists, intensities

def apply_crosstalk(image, crosstalk_scale):
    """
    Apply cross-zone "crosstalk" to a multi-zone rendered image. For each zone, add a fraction of 
    the sum of the other zones' histograms. This seems to occur because the lens is not perfect,
    and some portion of the light from other zones may bounce around inside the lens and end up
    hitting other zones.
    """
    sum_hist = torch.sum(image, dim=0)
    return image + sum_hist * crosstalk_scale

def soft_hist(d, size, vmin, vmax, sigma=None, weights=None, device='cpu'):
    # For each data point, construct a gaussian centered at that data point, and sum all these
    weights = weights.flatten() if weights is not None else torch.ones_like(d)
    bin_size = (vmax - vmin) / size
    sigma = sigma if sigma is not None else bin_size / 2

    centers = torch.arange(0, size).to(device) * bin_size + vmin + bin_size / 2

    # d is now dimension (N, 1), centers is dimension (1, N)
    x = torch.unsqueeze(d, 0) - torch.unsqueeze(centers, 1)
    x = 1 / torch.sqrt(2 * np.pi * sigma ** 2) * torch.exp(-1 / 2 * (x / sigma) ** 2) * weights
    x = x.sum(dim=1) * bin_size
    return x.unsqueeze_(0)    

def intersect_lines_planes(p0, p1, plane_a, plane_d):
    """
    Vectorized version: plane_a and plane_d are (N, 3) and p0 and p1 are (N, 3)
    From https://stackoverflow.com/a/18543221/8841061
    """
    u = p1 - p0
    dot = torch.sum(plane_a * u, dim=1)

    # find a point on the plane
    p_co = plane_a * plane_d

    w = p0 - p_co
    fac = -torch.sum(plane_a * w, dim=1) / dot
    u = u * fac.unsqueeze(1)
    return p0 + u

class Zone:
    """ Zone with multiple SPAD pixels. """

    pixel_width = 16.8      # width (x) of a single SPAD pixel (unit: um)
    pixel_height = 38.8     # height (y) of a single SPAD pixel (unit: um)
    focal_distance = 400    # distance from optical center to lens (unit: um)

    def __init__(self, center=(0, 0), shape=(4, 2)):
        """
        Args:
            center (float List[2]): xy-coordinates of zone center.
            shape (int List[2]): zone dimension (e.g., (4, 2)).
        """
        center = [-center[0] * self.pixel_width, -center[1] * self.pixel_height]
        
        self.center = center
        self.shape = shape

        # zone size
        width = self.pixel_width * shape[0]
        height = self.pixel_height * shape[1]

        # bottom-left and top-right corner of zone
        zone_x0 = center[0] - width / 2
        zone_y0 = center[1] - height / 2
        zone_x1 = center[0] + width / 2
        zone_y1 = center[1] + height / 2

        # bottom-left and top-right corner of rectangle at unit distance
        # from optical center 
        x0 = -zone_x1 / self.focal_distance
        y0 = -zone_y1 / self.focal_distance
        x1 = -zone_x0 / self.focal_distance
        y1 = -zone_y0 / self.focal_distance

        # FoV (in radians)
        self.xfov = np.arctan(x1) - np.arctan(x0)
        self.yfov = np.arctan(y1) - np.arctan(y0)

        v00 = np.array([x0, y0, 1])
        v01 = np.array([x0, y1, 1])
        v10 = np.array([x1, y0, 1])
        v11 = np.array([x1, y1, 1])

        n0 = np.cross(v00, v10)
        n1 = np.cross(v10, v11)
        n2 = np.cross(v11, v01)
        n3 = np.cross(v01, v00)
        n0 /= np.linalg.norm(n0)
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        n3 /= np.linalg.norm(n3)

        g0 = np.arccos(np.dot(n0, n1))
        g1 = np.arccos(np.dot(n1, n2))
        g2 = np.arccos(np.dot(n2, n3))
        g3 = np.arccos(np.dot(n3, n0))

        b0, b1 = n0[-1], n2[-1]
        k = 2 * np.pi - g2 - g3
        S = g0 + g1 - k # solid angle

        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.b0, self.b1 = b0, b1
        self.k = k
        self.S = S

    def get_xfov(self):
        """ Return FoV (in radians) along x-axis. """
        return self.xfov

    def get_yfov(self):
        """ Return FoV (in radians) along y-axis. """
        return self.yfov

    def _stratified_uv_sample(self, n):
        """ Stratified sampling from unit square. """
        tics = np.linspace(0, n, n + 1)[:-1]
        grid_v, grid_u = np.meshgrid(tics, tics)
        grid = np.stack((grid_u, grid_v), axis=-1).reshape(-1, 2)
        uv = (grid + np.random.rand(*grid.shape)) / n
        return uv

    def sample_ray_directions(self, num_rays):
        n = int(np.sqrt(num_rays))
        assert n ** 2 == num_rays
        uv = self._stratified_uv_sample(n)  # (n**2, 2)
        u, v = uv[:, 0], uv[:, 1]

        au = u * self.S + self.k
        fu = (np.cos(au) * self.b0 - self.b1) / np.sin(au)
        cu = ((fu > 0) * 2 - 1) / np.sqrt(fu ** 2 + self.b0 ** 2)
        cu = np.clip(cu, -1, 1)

        xu = -cu / np.sqrt(1 - cu ** 2)
        xu = np.clip(xu, self.x0, self.x1)

        d = np.sqrt(xu ** 2 + 1)
        h0 = self.y0 / np.sqrt(d ** 2 + self.y0 ** 2)
        h1 = self.y1 / np.sqrt(d ** 2 + self.y1 ** 2)
        hv = h0 + v * (h1 - h0)
        yv = (hv * d) / np.sqrt(1 - hv ** 2 + 1e-8)
        yv = np.clip(yv, self.y0, self.y1)

        r = np.stack((xu, yv, np.ones(n ** 2)), -1) # (n**2, 3)
        r /= np.linalg.norm(r, axis=-1, keepdims=True)
        return r

def resample(x, scale_factor, sigma=torch.scalar_tensor(1.0)):
    """
    Resample a function along the x axis to "squish" or "stretch" it
    """

    impulse_bin_centers = torch.arange(x.shape[0]).to(torch.float32).repeat(x.shape[0], 1).transpose(0, 1)
    sample_bin_centers = torch.arange(x.shape[0]).to(torch.float32).repeat(x.shape[0], 1) * scale_factor

    gaussians = 1 / torch.sqrt(2 * np.pi * sigma ** 2) * torch.exp(-(1 / 2) * ((sample_bin_centers -  impulse_bin_centers)**2 / (sigma ** 2))) * x

    # gaussians is a 128x128 array where [0,:] is a gaussian centered on bin 0, [1,:] is a gaussian centered on bin 1, etc.
    # sum along the rows (gaussians) to get the output signal
    return gaussians.sum(dim=1)

def l2_normalized_offset_loss(rendered_images, gt_images, offset_level=200):
    """
    DC offset is somewhat unpredicatble, so instead of trying to model it this loss can be used to
    effectively ignore it. Loss is only calculated on the bins for which the ground truth is above
    the offset_level. Normalized to deal with varying magnitudes just like l2_normalized_loss
    """
    # set all the bins which are below offset_level in the ground truth to 0 in both the ground
    # truth and the rendered images
    mask = gt_images < offset_level
    gt_images = torch.where(mask, torch.zeros_like(gt_images), gt_images)
    rendered_images = torch.where(mask, torch.zeros_like(rendered_images), rendered_images)

    # normalize the histograms
    maxes = torch.max(gt_images, dim=2, keepdim=True)[0]
    rendered_images = rendered_images / maxes
    gt_images = gt_images / maxes

    return torch.sum((rendered_images - gt_images) ** 2) / rendered_images.shape[0]


def differentiable_method(
    hists,
    reference_hist,
    device,
    initial_est_fn=direct_method,
    fit_iters=100,
    samples_per_zone=256 * 9,
    loss_fn=l2_normalized_offset_loss,
):
    """
    Differentiable plane recovery where only the geometry is optimized
    """

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

    initial_a_est, initial_d_est, _ = initial_est_fn(hists)

    # convert to aoi, azimuth, z_dist
    initial_aoi_est = angle_between_vecs(initial_a_est, [0, 0, 1])
    initial_azimuth_est = np.arctan2(initial_a_est[1], initial_a_est[0])
    initial_z_dist_est = initial_d_est / initial_a_est[2]

    aoi_est = torch.scalar_tensor(initial_aoi_est, requires_grad=True)
    azimuth_est = torch.scalar_tensor(initial_azimuth_est, requires_grad=True)
    z_dist_est = torch.scalar_tensor(initial_z_dist_est, requires_grad=True)

    forward_params["surface_albedo"].requires_grad = True
    forward_params["specular_weight"].requires_grad = True
    forward_params["specular_exponent"].requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": aoi_est, "lr": 0.001},
            {"params": azimuth_est, "lr": 0.001},
            {"params": z_dist_est, "lr": 0.001},
            {"params": forward_params["surface_albedo"], "lr": 0.1},
            {"params": forward_params["specular_weight"], "lr": 0.01},
            {"params": forward_params["specular_exponent"], "lr": 0.01},
        ]
    )

    for fit_iter in tqdm(range(fit_iters), leave=False, desc="fitting plane"):
        optimizer.zero_grad()

        image = render(
            aoi=aoi_est,
            azimuth=azimuth_est,
            z_dist=z_dist_est,
            impulse_response=torch.tensor(reference_hist).to(torch.float64),
            zone_spec=ZONE_SPEC,
            samples_per_zone=samples_per_zone,
            device=device,
            **forward_params
        )

        loss = loss_fn(image.unsqueeze(0), torch.tensor(hists).unsqueeze(0))

        loss.backward()

        optimizer.step()

    a_est = np.array(
        [
            np.cos(azimuth_est.item()) * np.sin(aoi_est.item()),
            np.sin(azimuth_est.item()) * np.sin(aoi_est.item()),
            np.cos(aoi_est.item()),
        ]
    )
    d_est = a_est[2] * z_dist_est.item()

    return a_est, d_est, None

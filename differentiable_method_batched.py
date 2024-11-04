import numpy as np
import torch
from torch.func import vmap


def render_batched(
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
    device,
):

    batch_size = aoi.shape[0]

    # validate input shapes
    assert azimuth.shape == (batch_size,)
    assert z_dist.shape == (batch_size,)
    assert impulse_response.shape == (batch_size, 128)
    assert surface_albedo.shape == (batch_size,)
    assert edge_brightness.shape == (batch_size,)
    assert corner_brightness.shape == (batch_size,)
    assert bin_offset.shape == (batch_size,)
    assert bin_size.shape == (batch_size,)
    assert crosstalk_scale.shape == (batch_size,)
    assert specular_weight.shape == (batch_size,)
    assert specular_exponent.shape == (batch_size,)
    assert dc_offset.shape == (batch_size,)
    assert soft_hist_sigma.shape == (batch_size,)
    assert impulse_x_scale.shape == (batch_size,)
    assert impulse_y_offset.shape == (batch_size,)
    assert saturation_point.shape == (batch_size,)

    # Create the base vectors
    x_basis_vec = torch.tensor([1, 0, 0])
    y_basis_vec = torch.tensor([0, 1, 0])
    z_basis_vec = torch.tensor([0, 0, 1])

    # Expand the base vectors to match the batch size
    x_basis_vec = x_basis_vec.unsqueeze(0).expand(batch_size, -1)  # (batch_size, 3)
    y_basis_vec = y_basis_vec.unsqueeze(0).expand(batch_size, -1)  # (batch_size, 3)
    z_basis_vec = z_basis_vec.unsqueeze(0).expand(batch_size, -1)  # (batch_size, 3)

    # form the plane_a vector using the azimuth, aoi, and z_dist
    # (ax+d=0 form where d is positive)
    plane_a = (
        x_basis_vec * torch.cos(azimuth).unsqueeze(1) * torch.sin(aoi).unsqueeze(1)
        + y_basis_vec * torch.sin(azimuth).unsqueeze(1) * torch.sin(aoi).unsqueeze(1)
        + z_basis_vec * torch.cos(aoi).unsqueeze(1)
    )  # (batch_size, 3)
    plane_d = plane_a[:, 2] * z_dist  # (batch_size,)

    image = torch.zeros(batch_size, 9, 128)

    corner_zone_idxs = torch.tensor([1, 0, 1, 0, 0, 0, 1, 0, 1]).unsqueeze(0).expand(batch_size, -1)
    edge_zone_idxs = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0]).unsqueeze(0).expand(batch_size, -1)
    center_zone_idxs = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0]).unsqueeze(0).expand(batch_size, -1)

    gains = (
        corner_zone_idxs * corner_brightness[:, None] * surface_albedo[:, None]
        + edge_zone_idxs * edge_brightness[:, None] * surface_albedo[:, None]
        + center_zone_idxs * surface_albedo[:, None]
    )  # (batch_size, 9)

    # rescale the impulse response
    # first subtract the y offset and clamp any negative values
    impulse_response = impulse_response.to(device)  # (batch_size, 128)
    impulse_response = impulse_response - impulse_y_offset[:, None]  # (batch_size, 128)
    impulse_response = torch.clamp(impulse_response, min=0)  # (batch_size, 128)
    # then normalize it so that the max is 1
    impulse_response = (
        impulse_response / impulse_response.max(dim=1).values[:, None]
    )  # (batch_size, 128)
    # then scale it along the x axis by impulse_x_scale - this requires resampling and interpolating
    scaled_impulse = vmap(resample)(impulse_response, impulse_x_scale)  # (batch_size, 128)

    for i, single_zone_spec in enumerate(zone_spec):
        # sample rays from zone
        zone = Zone(
            single_zone_spec["center_px"],
            [single_zone_spec["width_px"], single_zone_spec["height_px"]],
        )
        rays = zone.sample_ray_directions(samples_per_zone)  # (samples_per_zone, 3)

        # render all the rays to get a distance and intensity measure for each ray
        # the same random rays are sampled for each batch (could be improved)
        dists, intensities = vmap(render_rays, in_dims=(0, 0, 0, 0, 0, 0, 0, None))(
            torch.from_numpy(rays)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(device),  # (batch_size, samples_per_zone, 3)
            plane_a,
            plane_d,
            specular_weight,
            specular_exponent,
            saturation_point,
            gains[:, i],
            device,
        ) # (batch_size, samples_per_zone), (batch_size, samples_per_zone)

        # apply some constant offset to the distances (this should be the same as offsetting the
        # final histogram by some amount?)
        dists = dists + (bin_offset[:, None] * bin_size[:, None])

        # create a histogram of the distances using differentiable soft hist function
        hist_max = bin_size * 128 # (batch_size,)
        sigma = soft_hist_sigma * bin_size # (batch_size,)
        hist = vmap(soft_hist, in_dims=(0, 0, 0, 0, None, None, None))(
            dists, hist_max, intensities, sigma, 0, 128, device
        ) # (batch_size, 128)

        # convolve the impulse response - pad the start but cut off the end
        num_bins = hist.shape[1]
        hist = torch.nn.functional.conv1d(
            hist.unsqueeze(1), # (13, 1, 128) - conv1d expects (batch_size, num_channels, length)
            torch.flip(scaled_impulse, dims=[1]).unsqueeze(1), # (1, 13, 128)
            padding=num_bins - 1,
            groups=1
        )
        # TODO not sure why taking the 0 index on axis 1 below is necessary
        hist = hist[:, 0, :num_bins] # (13, 128) 

        image[:, i] = hist

    # apply crosstalk between zones
    image = vmap(apply_crosstalk)(image, crosstalk_scale) # (batch_size, 9, 128)

    # add the dc offset - maybe this should override the function rather than add to it?
    # e.g. maybe it should be hist = max(dc_offset, hist)
    image = image + dc_offset[:, None, None] # (batch_size, 9, 128)

    return image


def render_rays(
    rays, plane_a, plane_d, specular_weight, specular_exponent, saturation_point, gain, device
):
    """
    The first dimension of each input tensor is the number of rays
    """
    num_rays = rays.shape[0]
    plane_as = plane_a.repeat(num_rays, 1).to(device)
    plane_ds = plane_d.repeat(num_rays, 1).to(device)

    pts = intersect_lines_planes(torch.zeros(num_rays, 1).to(device), rays, plane_as, plane_ds)

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

    r_hat = 2 * torch.sum(-rays * -plane_as, dim=1).unsqueeze(1) * -plane_as + rays
    specular_dotprod = torch.sum(-rays * r_hat, dim=1)
    # if the specular dot product is negative, set the specular term to zero
    specular_dotprod = torch.where(
        specular_dotprod < 0, torch.tensor(0.0).to(device), specular_dotprod
    )
    specular_term = specular_dotprod**specular_exponent * specular_weight

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


def soft_hist(d, vmax, weights, sigma, vmin, size, device):
    # For each data point, construct a gaussian centered at that data point, and sum all these
    weights = weights.flatten() if weights is not None else torch.ones_like(d)
    bin_size = (vmax - vmin) / size
    sigma = sigma if sigma is not None else bin_size / 2

    centers = torch.arange(0, size).to(device) * bin_size + vmin + bin_size / 2

    # d is now dimension (N, 1), centers is dimension (1, N)
    x = torch.unsqueeze(d, 0) - torch.unsqueeze(centers, 1)
    x = 1 / torch.sqrt(2 * np.pi * sigma**2) * torch.exp(-1 / 2 * (x / sigma) ** 2) * weights
    x = x.sum(dim=1) * bin_size
    return x


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
    """Zone with multiple SPAD pixels."""

    pixel_width = 16.8  # width (x) of a single SPAD pixel (unit: um)
    pixel_height = 38.8  # height (y) of a single SPAD pixel (unit: um)
    focal_distance = 400  # distance from optical center to lens (unit: um)

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
        S = g0 + g1 - k  # solid angle

        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.b0, self.b1 = b0, b1
        self.k = k
        self.S = S

    def get_xfov(self):
        """Return FoV (in radians) along x-axis."""
        return self.xfov

    def get_yfov(self):
        """Return FoV (in radians) along y-axis."""
        return self.yfov

    def _stratified_uv_sample(self, n):
        """Stratified sampling from unit square."""
        tics = np.linspace(0, n, n + 1)[:-1]
        grid_v, grid_u = np.meshgrid(tics, tics)
        grid = np.stack((grid_u, grid_v), axis=-1).reshape(-1, 2)
        uv = (grid + np.random.rand(*grid.shape)) / n
        return uv

    def sample_ray_directions(self, num_rays):
        n = int(np.sqrt(num_rays))
        assert n**2 == num_rays
        uv = self._stratified_uv_sample(n)  # (n**2, 2)
        u, v = uv[:, 0], uv[:, 1]

        au = u * self.S + self.k
        fu = (np.cos(au) * self.b0 - self.b1) / np.sin(au)
        cu = ((fu > 0) * 2 - 1) / np.sqrt(fu**2 + self.b0**2)
        cu = np.clip(cu, -1, 1)

        xu = -cu / np.sqrt(1 - cu**2)
        xu = np.clip(xu, self.x0, self.x1)

        d = np.sqrt(xu**2 + 1)
        h0 = self.y0 / np.sqrt(d**2 + self.y0**2)
        h1 = self.y1 / np.sqrt(d**2 + self.y1**2)
        hv = h0 + v * (h1 - h0)
        yv = (hv * d) / np.sqrt(1 - hv**2 + 1e-8)
        yv = np.clip(yv, self.y0, self.y1)

        r = np.stack((xu, yv, np.ones(n**2)), -1)  # (n**2, 3)
        r /= np.linalg.norm(r, axis=-1, keepdims=True)
        return r


def resample(x, scale_factor, sigma=torch.scalar_tensor(1.0)):
    """
    Resample a function along the x axis to "squish" or "stretch" it
    """

    impulse_bin_centers = (
        torch.arange(x.shape[0]).to(torch.float32).repeat(x.shape[0], 1).transpose(0, 1)
    )
    sample_bin_centers = (
        torch.arange(x.shape[0]).to(torch.float32).repeat(x.shape[0], 1) * scale_factor
    )

    gaussians = (
        1
        / torch.sqrt(2 * np.pi * sigma**2)
        * torch.exp(-(1 / 2) * ((sample_bin_centers - impulse_bin_centers) ** 2 / (sigma**2)))
        * x
    )

    # gaussians is a 128x128 array where [0,:] is a gaussian centered on bin 0, [1,:] is a gaussian centered on bin 1, etc.
    # sum along the rows (gaussians) to get the output signal
    return gaussians.sum(dim=1)

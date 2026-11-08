"""Batched GPU pixel rendering for brax MJX envs via the MuJoCo Warp ray tracer.

Physics stays in brax/MJX, bit-identical to the state-based runs; this module
only turns each step's qpos into observation frames. Rendering runs on a
"render twin" of the physics model (same kinematic tree, retuned appearance) and
never leaves the GPU: qpos comes in as a torch tensor, frames come out as a
torch tensor.
"""

from __future__ import annotations

import mujoco
import torch

RENDER_ASSETS = {
    "halfcheetah": "half_cheetah.xml",
    "walker2d": "walker2d.xml",
    "humanoid": "humanoid.xml",
}


def build_render_model(env_name: str) -> mujoco.MjModel:
    """Compile a appearance-retuned twin of the brax asset for 84x84 frames.

    Rendering-only changes (physics never reads them):
    - Floor checker at lower contrast and 24 repeats: the stock sub-metre
      cells alias into full-contrast moire under 1-ray-per-pixel ray tracing.
    - Light tilted toward the tracking camera: the stock straight-down light
      leaves camera-facing surfaces unlit (pure black silhouettes).
    - Dark gray robot and a bright skybox so the body reads against both the
      floor band and the sky in grayscale.
    """
    from etils import epath

    path = epath.resource_path("brax") / "envs/assets" / RENDER_ASSETS[env_name]
    spec = mujoco.MjSpec.from_string(path.read_text())
    for tex in spec.textures:
        if tex.name == "texplane":
            tex.rgb1[:] = [0.35, 0.35, 0.35]
            tex.rgb2[:] = [0.65, 0.65, 0.65]
        if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            tex.rgb1[:] = [0.9, 0.9, 0.9]
            tex.rgb2[:] = [0.6, 0.6, 0.6]
    for mat in spec.materials:
        if mat.name == "MatPlane":
            mat.texrepeat[:] = [24.0, 24.0]
            mat.reflectance = 0.0
    for geom in spec.geoms:
        if geom.name != "floor":
            geom.rgba[:] = [0.25, 0.25, 0.25, 1.0]
    for light in spec.lights:
        light.dir[:] = [0.0, 0.5, -1.0]
    return spec.compile()


class MJWarpBatchRenderer:
    """Grayscale frames for a batch of worlds from batched qpos, on GPU.

    Renders through the model's first camera (the envs' tracking camera) at
    ``image_size * supersample`` and average-pools down: supersampling is
    ~free (the ray tracer is launch-bound at these resolutions) and stands in
    for the mip filtering GL renderers get.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        *,
        nworld: int,
        image_size: int,
        supersample: int = 2,
        device: torch.device,
    ):
        if device.type != "cuda":
            raise ValueError(f"MJWarpBatchRenderer requires a CUDA device, got {device}")
        if mj_model.ncam < 1:
            raise ValueError("render model has no camera")
        import mujoco_warp as mjw
        import warp as wp

        self._wp = wp
        self._mjw = mjw
        self.nworld = nworld
        self.image_size = image_size
        self.supersample = supersample
        self.device = device

        mjd = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mjd)
        self._m = mjw.put_model(mj_model)
        self._d = mjw.put_data(mj_model, mjd, nworld=nworld)

        res = image_size * supersample
        self._rc = mjw.create_render_context(
            mj_model,
            nworld=nworld,
            cam_res=(res, res),
            render_rgb=True,
            render_depth=False,
            render_skybox=True,
        )
        # Zero-copy views shared between torch and warp.
        self._qpos_t = wp.to_torch(self._d.qpos)
        self._rgb_t = torch.zeros((nworld, res, res, 3), device=device)
        self._rgb_wp = wp.from_torch(self._rgb_t, dtype=wp.vec3)
        self._lum = torch.tensor([0.299, 0.587, 0.114], device=device)

    @torch.no_grad()
    def render_gray(self, qpos: torch.Tensor) -> torch.Tensor:
        """``qpos (nworld, nq)`` -> frames ``(nworld, H, W)``, float32 in [0, 255].

        Values are rounded to integers so a uint8 round-trip is lossless.
        """
        mjw = self._mjw
        self._qpos_t.copy_(qpos)
        torch.cuda.synchronize(self.device)  # qpos write must be visible to warp's stream
        mjw.kinematics(self._m, self._d)
        mjw.com_pos(self._m, self._d)  # trackcom cameras read subtree_com
        mjw.camlight(self._m, self._d)
        mjw.refit_bvh(self._m, self._d, self._rc)
        mjw.render(self._m, self._d, self._rc)
        mjw.get_rgb(self._rc, 0, self._rgb_wp)
        self._wp.synchronize_device()  # frame must be visible to torch's stream
        gray = self._rgb_t @ self._lum
        if self.supersample > 1:
            s, hw = self.supersample, self.image_size
            gray = gray.view(self.nworld, hw, s, hw, s).mean(dim=(2, 4))
        return torch.round(gray.clamp(0.0, 1.0) * 255.0)

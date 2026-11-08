"""Patch QDax v1 imports to use Brax v2 / MJX-compatible modules."""

from __future__ import annotations

import sys
import types

from qdax.tasks.brax.v2.envs import base_env as _v2_base_env

# QDax emitters import qdax.tasks.brax.v1.envs.base_env, which triggers v1/__init__.py
# and a hard dependency on deprecated brax.v1. Pre-register stub modules.
_v1_pkg = types.ModuleType("qdax.tasks.brax.v1")
_v1_envs = types.ModuleType("qdax.tasks.brax.v1.envs")
_v1_envs.base_env = _v2_base_env
_v1_pkg.envs = _v1_envs

sys.modules["qdax.tasks.brax.v1"] = _v1_pkg
sys.modules["qdax.tasks.brax.v1.envs"] = _v1_envs
sys.modules["qdax.tasks.brax.v1.envs.base_env"] = _v2_base_env

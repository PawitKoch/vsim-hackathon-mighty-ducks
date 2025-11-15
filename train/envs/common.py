import torch
import vlearn as v
from vlearn.constant import GLOBAL_ENV_DEF_HANDLE


def create_plane(gym, dynamic_friction=0.5, static_friction=0.5, restitution=0.0, damping=0.0):

    # Create rigid material
    material = v.RigidMaterial()
    material.dynamic_friction = dynamic_friction
    material.static_friction = static_friction
    material.restitution = restitution
    material.damping = damping

    global_env_def = gym.get_environment_def(GLOBAL_ENV_DEF_HANDLE)
    material_handle = global_env_def.create_rigid_material(material)

    # Rotate plane so that plane normal is up
    up_axis = gym.get_up_axis()
    rot = v.shortest_rotation(v.Vec3(0, 1, 0), up_axis)
    pos = v.Vec3(0)
    transform = v.Transform(rot, pos)

    gym.create_plane(transform, material_handle)


def create_box(gym, halfsize, boxPos, dynamic_friction=0.5, static_friction=0.5):

    global_env_def = gym.get_environment_def(GLOBAL_ENV_DEF_HANDLE)

    material = v.RigidMaterial()
    material.dynamic_friction = dynamic_friction
    material.static_friction = static_friction
    material.restitution = 0.
    material.damping = 0.

    material_handle = global_env_def.create_rigid_material(material)

    box_def_handle = global_env_def.create_box_def(halfsize, rigid_material_handle=material_handle)
    global_env_def.create_rigid_body(box_def_handle, v.Transform(v.Quat(0, 0, 0, 1), 2 *
                                                                 boxPos))


def create_heightfield(gym):
    material = v.RigidMaterial()
    material.dynamic_friction = 0.5
    material.static_friction = 0.5
    material.restitution = 0.
    material.damping = 0.

    materialHandle = gym.create_rigid_material(material)

    maxRigidBodyDef = 1
    maxRigidBodies = 1
    rigidBodySetHandle = gym.create_rigid_body_set(maxRigidBodyDef, maxRigidBodies)

    planePos = v.Vec3(-10., 0.8, -10.)
    planeQuat = v.Quat(0., 0., 0., 1.)
    localTransform = v.Transform(v.Vec3(0., 0., 0.))
    planeTransform = v.Transform(planeQuat, planePos)
    gym.create_heightfield(rigidBodySetHandle, localTransform, planeTransform, materialHandle)


def transform_tensor(transform: v.Transform, device: torch.device):
    return torch.tensor([transform.q.x, transform.q.y, transform.q.z,
                         transform.q.w, transform.p.x, transform.p.y, transform.p.z],
                        device=device, dtype=torch.float32)


def spatial_vector_tensor(vector: v.SpatialVector, device: torch.device):
    return torch.tensor([vector.bottom.x, vector.bottom.y, vector.bottom.z,
                         vector.top.x, vector.top.y, vector.top.z], device=device,
                        dtype=torch.float32)


def quat_tensor(quat: v.Quat, device: torch.device):
    return torch.tensor([quat.x, quat.y, quat.z, quat.w], device=device,
                        dtype=torch.float32)


@torch.jit.script
def reset_noise_helper(init_dof_val, noise_scale: float, scale1: float, scale2: float):
    return noise_scale * (torch.rand(init_dof_val.shape, dtype=torch.float32,
                          device=init_dof_val.device) * scale1 - scale2)

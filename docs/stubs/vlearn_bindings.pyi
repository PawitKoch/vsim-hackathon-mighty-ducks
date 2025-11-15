"""

        Pybind11 example plugin
        -----------------------

        .. currentmodule:: vlearn

        .. autosummary::
           :toctree: _generate

           math
           rigid
           gym
           render
    
"""
from __future__ import annotations
import typing
__all__: list[str] = ['A', 'ANCHOR1', 'ANCHOR2', 'ANTI_ALIASING', 'ARMATURE', 'ARTICULATION', 'Articulation', 'ArticulationControlType', 'ArticulationDef', 'ArticulationDefHandle', 'ArticulationFixedTendonHandle', 'ArticulationForceSensorHandle', 'ArticulationHandle', 'ArticulationJointDef', 'ArticulationJointDofDef', 'ArticulationJointSensorHandle', 'ArticulationKinematicSensorDef', 'ArticulationKinematicStateCommand', 'ArticulationKinematicStateCommandGpuArray', 'ArticulationLinkDef', 'ArticulationMotorDefHandle', 'ArticulationMotorHandle', 'ArticulationPIDDefHandle', 'ArticulationPIDHandle', 'ArticulationSpatialTendonHandle', 'B', 'BoolGpuArray', 'BoolGpuBufferWrapper', 'BroadPhaseType', 'CHILD_FRAME', 'COLOR', 'COM', 'Camera', 'CameraPostFilter', 'CameraType', 'ContactFilterCommand', 'ContactFilterCommandGpuArray', 'ContactFilterHandle', 'DAMPING', 'DYNAMIC_FRICTION', 'Deformable', 'DeformableDef', 'DeformableDefHandle', 'DeformableHandle', 'DeformableMaterial', 'DeformableMaterialHandle', 'DeformableMaterialProperty', 'DeformableMaterialPropertyCommand', 'DeformableMaterialPropertyCommandGpuArray', 'DeformableTransformCommand', 'DeformableTransformCommandGpuArray', 'DepthCamera', 'DepthCameraCommand', 'DepthCameraCommandGpuArray', 'DepthCameraDef', 'DepthCameraHandle', 'DepthCameraTransformCommand', 'DepthCameraTransformCommandGpuArray', 'ENVIRONMENT', 'ElasticMaterial', 'ElasticMaterialHandle', 'ElasticMaterialProperty', 'ElasticMaterialPropertyCommand', 'ElasticMaterialPropertyCommandGpuArray', 'Environment', 'EnvironmentDef', 'EnvironmentDefHandle', 'EnvironmentGroup', 'EnvironmentGroupHandle', 'EnvironmentHandle', 'EnvironmentSet', 'EnvironmentSetHandle', 'ErasePatchDepthCommand', 'ErasePatchDepthCommandGpuArray', 'ErasePatchRGBCommand', 'ErasePatchRGBCommandGpuArray', 'ErasePixelsDepthCommand', 'ErasePixelsDepthCommandGpuArray', 'ErasePixelsRGBCommand', 'ErasePixelsRGBCommandGpuArray', 'FORCE', 'FORCE_POSITION', 'FORCE_TORQUE', 'FRICTION', 'Finalizable', 'FixedTendonControlCommand', 'FixedTendonControlCommandGpuArray', 'FixedTendonDef', 'FixedTendonProperty', 'FixedTendonPropertyCommand', 'FixedTendonPropertyCommandGpuArray', 'Float32GpuArray', 'Float32GpuBufferWrapper', 'ForceSensorCommand', 'ForceSensorCommandGpuArray', 'ForceSensorDef', 'ForceType', 'FrameType', 'GEAR_RATIO', 'GLOBAL_ENV_DEF_HANDLE', 'GLOBAL_ENV_HANDLE', 'GaussianNoiseDepthCommand', 'GaussianNoiseDepthCommandGpuArray', 'GaussianNoiseRGBCommand', 'GaussianNoiseRGBCommandGpuArray', 'GravityCompensationCommand', 'GravityCompensationCommandGpuArray', 'Gym', 'GymRender', 'GymSingleton', 'HIGH_LIMIT', 'HillMaterial', 'HillMaterialHandle', 'HillMaterialProperty', 'HillMaterialPropertyCommand', 'HillMaterialPropertyCommandGpuArray', 'INV_MASS', 'InverseDynamicsCommand', 'InverseDynamicsCommandGpuArray', 'InverseKinematicsCommand', 'InverseKinematicsCommandGpuArray', 'JOINT_FRICTION', 'JacobianCommand', 'JacobianCommandGpuArray', 'JointActiveMaskCommand', 'JointActiveMaskCommandGpuArray', 'JointDofProperty', 'JointDofPropertyCommand', 'JointDofPropertyCommandGpuArray', 'JointForceSensorCommand', 'JointForceSensorCommandGpuArray', 'JointProperty', 'JointPropertyCommand', 'JointPropertyCommandGpuArray', 'JointStateCommand', 'JointStateCommandGpuArray', 'KinematicSensorHandle', 'KinematicSensorStateCommand', 'KinematicSensorStateCommandGpuArray', 'KinematicSensorTransformCommand', 'KinematicSensorTransformCommandGpuArray', 'KinematicSensorType', 'KinematicSensorVelocityCommand', 'KinematicSensorVelocityCommandGpuArray', 'LENGTH', 'LIMIT_DAMPING', 'LIMIT_STIFFNESS', 'LOCAL', 'LOW_LIMIT', 'LSE0', 'LUMINESCENE', 'LightHandle', 'LightTransformCommand', 'LightTransformCommandGpuArray', 'LinkExternalForceCommand', 'LinkExternalForceCommandGpuArray', 'LinkProperty', 'LinkPropertyCommand', 'LinkPropertyCommandGpuArray', 'LinkTransformCommand', 'LinkTransformCommandGpuArray', 'LinkVelocityCommand', 'LinkVelocityCommandGpuArray', 'MASS', 'MAX_FORCE', 'MAX_JOINT_VELOCITY', 'MODEL', 'MOTOR', 'MassMatrixCommand', 'MassMatrixCommandGpuArray', 'Model', 'ModelDef', 'MotorControlCommand', 'MotorControlCommandGpuArray', 'MotorDef', 'MotorProperty', 'MotorPropertyCommand', 'MotorPropertyCommandGpuArray', 'NONE', 'PIDControlCommand', 'PIDControlCommandGpuArray', 'PIDDef', 'PIDProperty', 'PIDPropertyCommand', 'PIDPropertyCommandGpuArray', 'PINHOLE', 'POISSON_RATIO', 'Quat', 'QueryMode', 'RADIAL', 'RESTITUTION', 'RGBCamera', 'RGBCameraCommand', 'RGBCameraCommandGpuArray', 'RGBCameraDef', 'RGBCameraDefHandle', 'RGBCameraHandle', 'RGBCameraSkybox', 'RGBCameraSkyboxCommand', 'RGBCameraSkyboxCommandGpuArray', 'RGBCameraTransformCommand', 'RGBCameraTransformCommandGpuArray', 'RGBMaterial', 'RGBMaterialHandle', 'RGBMaterialProperty', 'RGBMaterialPropertyCommand', 'RGBMaterialPropertyCommandGpuArray', 'RIGID_BODY', 'RaycastCommand', 'RaycastCommandGpuArray', 'RigidBody', 'RigidBodyDef', 'RigidBodyDefCommand', 'RigidBodyDefCommandGpuArray', 'RigidBodyDefHandle', 'RigidBodyExternalForceCommand', 'RigidBodyExternalForceCommandGpuArray', 'RigidBodyHandle', 'RigidBodyKinematicSensorDef', 'RigidBodyKinematicStateCommand', 'RigidBodyKinematicStateCommandGpuArray', 'RigidBodyProperty', 'RigidBodyPropertyCommand', 'RigidBodyPropertyCommandGpuArray', 'RigidBodyTransformCommand', 'RigidBodyTransformCommandGpuArray', 'RigidBodyVelocityCommand', 'RigidBodyVelocityCommandGpuArray', 'RigidDistanceJointHandle', 'RigidDistanceJointProperty', 'RigidDistanceJointPropertyCommand', 'RigidDistanceJointPropertyCommandGpuArray', 'RigidMaterial', 'RigidMaterialHandle', 'RigidMaterialProperty', 'RigidMaterialPropertyCommand', 'RigidMaterialPropertyCommandGpuArray', 'SCENE', 'SPECULAR_EXPONENT', 'SPECULAR_INTENSITY', 'STATIC_FRICTION', 'STIFFNESS', 'SegmentedDepthCameraCommand', 'SegmentedDepthCameraCommandGpuArray', 'SegmentedRGBCameraCommand', 'SegmentedRGBCameraCommandGpuArray', 'SpatialTendonControlCommand', 'SpatialTendonControlCommandGpuArray', 'SpatialTendonDef', 'SpatialTendonProperty', 'SpatialTendonPropertyCommand', 'SpatialTendonPropertyCommandGpuArray', 'SpatialTendonState', 'SpatialTendonStateCommand', 'SpatialTendonStateCommandGpuArray', 'SpatialVector', 'TEXTURE', 'TendonDef', 'TextureHandle', 'Transform', 'TransformCommand', 'TransformCommandGpuArray', 'TransformHandle', 'TransformHandleGpuBufferWrapper', 'TransformType', 'USE_COLLISIONS', 'USE_FILE', 'USE_NONE', 'USE_VISUALS', 'Uint16GpuArray', 'Uint16GpuBufferWrapper', 'Uint32GpuArray', 'Uint32GpuBufferWrapper', 'Uint8GpuArray', 'Uint8GpuBufferWrapper', 'UserCheckbox', 'UserCombo', 'UserLine', 'UserLineCube', 'UserLineShape', 'UserMenuItem', 'UserSlider', 'VELOCITY', 'Vec3', 'VsMaterialHandle', 'VsQueryGeometryHandle', 'VsRigidBodyHandle', 'VsRigidBodySetHandle', 'VsTransformHandle', 'WORLD', 'YOUNGS_MODULUS', 'axis_angle_from_quat', 'matrix_from_quat', 'quat_from_rpy', 'rotate_inertia_matrix', 'rpy_from_quat', 'shortest_rotation']
class Articulation(Model):
    """
    A class holding the data of an instantiated articulation.
    
    .. warning::
        Handles returned by :meth:`get_motor_handle`, :meth:`get_pid_handle`,
        :meth:`get_force_sensor_handle`,
        :meth:`get_joint_sensor_handle`, :meth:`get_kinematic_sensor_handle`
        only get initialized after :meth:`EnvironmentGroup.create_environment_set` is called!
        Calls to these method beforehand, will return an invalid handle.
    """
    def get_articulation_def(self) -> ArticulationDef:
        """
        Gets the articulation definition.
        """
    def get_articulation_def_handle(self) -> ArticulationDefHandle:
        """
        Gets the articulation definition handle.
        
        It can be used to query :meth:`EnvironmentDef.get_articulation_def`.
        """
    def get_fixed_tendon_handle(self, index: int) -> ArticulationFixedTendonHandle:
        """
        Gets the :class:`ArticulationFixedTendonHandle` of this articulation instance.
        """
    def get_force_sensor_handle(self, arg0: int) -> ArticulationForceSensorHandle:
        """
        Gets the :class:`ArticulationForceSensorHandle` of this articulation instance.
        """
    def get_joint_sensor_handle(self) -> ArticulationJointSensorHandle:
        """
        Gets the :class:`ArticulationJointSensorHandle` of this articulation instance.
        """
    def get_kinematic_sensor_handle(self, arg0: int) -> KinematicSensorHandle:
        """
        Gets the :class:`KinematicSensorHandle` of this articulation instance.
        """
    def get_motor_handle(self) -> ArticulationMotorHandle:
        """
        Gets the :class:`ArticulationMotorHandle` of this articulation instance.
        """
    def get_pid_handle(self) -> ArticulationPIDHandle:
        """
        Gets the :class:`ArticulationPIDHandle` of this articulation instance.
        """
    def get_segmentation(self, index: int) -> int:
        """
        Gets the segmentation value for link `index` on this articulation instance.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
    def get_spatial_tendon_handle(self, index: int) -> ArticulationSpatialTendonHandle:
        """
        Gets the :class:`ArticulationSpatialTendonHandle` of this articulation instance.
        """
    def get_transform_handle(self, index: int) -> TransformHandle:
        """
        Gets the :class:`TransformHandle` for link `index` on this articulation instance.
        """
    def set_segmentation(self, index: int, segmentation: int) -> None:
        """
        Sets the segmentation value for link `index` on this articulation instance.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
    def set_segmentations(self, segmentations: list[int]) -> None:
        """
        Gets the segmentation value for link `index` on this articulation instance.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
class ArticulationControlType:
    """
    A flag enum to determine what type of controls the articulation can have.
    
    Members:
    
      MOTOR : Enables motors on the articulation.
    """
    MOTOR: typing.ClassVar[ArticulationControlType]  # value = <ArticulationControlType.MOTOR: 1>
    __members__: typing.ClassVar[dict[str, ArticulationControlType]]  # value = {'MOTOR': <ArticulationControlType.MOTOR: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ArticulationDef(ModelDef):
    """
    A class that defines the whole structure of a single articulation.
    
    This includes all link, joint, dof, shape, controller and sensors information.
    
    .. warning::
        Handles returned by :meth:`get_motor_control_def_handle` and :meth:`get_pid_control_def_handle`
        only get initialized after :meth:`EnvironmentGroup.create_environment_set` is called!
        Calls to these method beforehand, will return an invalid handle.
    """
    def enable_control_type(self, type: ArticulationControlType, enable: bool) -> None:
        """
        Enables or disables ``type`` control based on ``enable`` value.
        """
    def get_fixed_tendon_def(self, index: int) -> FixedTendonDef:
        """
        Gets the fixed tendon definition at ``index``.
        
        :param index: index
        :type index: int
        :returns: fixed tendon definition
        :rtype: FixedTendonDef
        """
    def get_force_sensor_def(self, index: int) -> ForceSensorDef:
        """
        Gets the force sensor def at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_joint_def(self, index: int) -> ArticulationJointDef:
        """
        Gets the joint definition at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_joint_def_names(self) -> list[str]:
        """
        Returns a list of all joint definition names.
        """
    def get_joint_dof_def(self, index: int) -> ArticulationJointDofDef:
        """
        Gets the DOF definition at ``index``.
        """
    def get_joint_dof_def_by_name(self, name: str) -> ArticulationJointDofDef:
        """
        Gets the DOF definition for the supplied ``name``.
        
        Returns `None` if the ``name`` isn't found.
        """
    def get_joint_dof_def_index_by_name(self, name: str) -> int:
        """
        Gets the DOF index for the supplied ``name``.
        
        Returns -1 if the ``name`` isn't found.
        """
    def get_joint_dof_def_name(self, index: int) -> str:
        """
        Gets the name of the DOF definition at ``index``.
        """
    def get_joint_dof_def_names(self) -> list[str]:
        """
        Gets the array of DOF definition names.
        """
    def get_joint_dof_index(self, link_index: int, local_dof_index: int) -> int:
        """
        Gets the global joint index for a local DOF at ``local_dof_index`` belonging to link at ``link_index``
        
        .. code-block:: python
        
            for i in range(artDef.get_num_pid_defs()):
                pid_def = artDef.get_pid_def(i)
                joint_index = artDef.get_joint_dof_index(pid_def.link_index, pid_def.local_dof_index)
                pid_indices.append(joint_index)
        """
    def get_kinematic_sensor_def(self, index: int) -> ArticulationKinematicSensorDef:
        """
        Gets the kinematic sensor def at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_link_def(self, index: int) -> ArticulationLinkDef:
        """
        Gets the link definition at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_link_def_names(self) -> list[str]:
        """
        Returns a list of all link definition names.
        """
    def get_link_index(self, dof_index: int) -> int:
        """
        Gets the link index for a given ``dof_index``
        """
    def get_motor_control_def_handle(self) -> ArticulationMotorDefHandle:
        """
        Gets the :class:`ArticulationMotorDefHandle` of this articulation instance.
        """
    def get_motor_def(self, index: int) -> MotorDef:
        """
        Gets the motor definition at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_motor_def_name(self, index: int) -> str:
        """
        Gets the motor definition name at ``index``.
        """
    def get_num_fixed_tendon_defs(self) -> int:
        """
        Gets the number of fixed tendon definitions contained in the articulation definition.
        """
    def get_num_force_sensor_defs(self) -> int:
        """
        Gets the number of force sensor defs contained in the articulation definition
        """
    def get_num_joint_defs(self) -> int:
        """
        Gets the number of joint definitions contained in the articulation definition.
        """
    def get_num_joint_dof_defs(self) -> int:
        """
        Returns the total number of degrees of freedom that the articulation definition has.
        """
    def get_num_kinematic_sensor_defs(self) -> int:
        """
        Gets the number of kinematic sensor defs contained in the articulation definition
        """
    def get_num_link_defs(self) -> int:
        """
        Gets the number of link definitions contained in the articulation definition.
        """
    def get_num_motor_defs(self) -> int:
        """
        Gets the number of motors contained in the articulation definition.
        """
    def get_num_pid_defs(self) -> int:
        """
        Gets the number of PID controller definitions contained in the articulation definition.
        """
    def get_num_spatial_tendon_defs(self) -> int:
        """
        Gets the number of spatial tendon definitions contained in the articulation definition.
        """
    def get_pid_control_def_handle(self) -> ArticulationPIDDefHandle:
        """
        Gets the :class:`ArticulationPIDDefHandle` of this articulation instance.
        """
    def get_pid_def(self, index: int) -> PIDDef:
        """
        Gets the PID definition at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_pid_def_name(self, index: int) -> str:
        """
        Gets the PID definition name at ``index``.
        """
    def get_pose(self, pose_name: str) -> list[float]:
        """
        Gets the sorted joint position array for a pose named `pose_name`.
        """
    def get_spatial_tendon_def(self, index: int) -> SpatialTendonDef:
        """
        Gets the spatial tendon definition at ``index``.
        
        :param index: index
        :type index: int
        :returns: spatial tendon definition
        :rtype: SpatialTendonDef
        """
    def has_control_type(self, type: ArticulationControlType) -> bool:
        """
        Returns whether ``type`` control is enabled **and** has any controls associated with it.
        """
    def has_joint_force_sensors(self) -> bool:
        """
        Returns whether articulation definition has joint force sensors
        """
    @property
    def has_self_collisions(self) -> bool:
        """
        Indicates whether self collisions are enabled for the articulation.
        
        Parent-child links are not going to collide regardless of this setting.
        """
    @has_self_collisions.setter
    def has_self_collisions(self, arg1: bool) -> None:
        ...
class ArticulationDefHandle:
    """
    A class wrapping a handle for an articulation definition.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationDefHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the articulation definition.
        """
class ArticulationFixedTendonHandle:
    """
    A class wrapping a handle for a fixed tendon.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationFixedTendonHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the fixed tendon
        """
class ArticulationForceSensorHandle:
    """
    A class wrapping a handle for an articulation force sensor.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationForceSensorHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the articulation force sensors.
        """
class ArticulationHandle:
    """
    A class wrapping a handle for an articulation instance.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the articulation instance.
        """
class ArticulationJointDef:
    """
    A class holding the data for an articulation joint definition.
    """
    @property
    def dof(self) -> int:
        """
        Dof of the joint.
        """
    @property
    def dof_start_index(self) -> int:
        """
        Dof start index of the joint.
        """
    @property
    def joint_friction(self) -> float:
        """
        The friction value of the joint.
        """
    @joint_friction.setter
    def joint_friction(self, arg1: float) -> None:
        ...
    @property
    def joint_friction_loss(self) -> float:
        """
        The friction loss value of the joint.
        """
    @joint_friction_loss.setter
    def joint_friction_loss(self, arg1: float) -> None:
        ...
    @property
    def max_joint_velocity(self) -> float:
        """
        The maximum joint velocity of the joint.
        """
    @max_joint_velocity.setter
    def max_joint_velocity(self, arg1: float) -> None:
        ...
    @property
    def name(self) -> str:
        """
        The name of the joint.
        """
class ArticulationJointDofDef:
    """
    A class holding the data for an articulation joint DOF definition.
    """
    @property
    def armature(self) -> float:
        """
        The armature value of the DOF.
        """
    @armature.setter
    def armature(self, arg1: float) -> None:
        ...
    @property
    def high_limit(self) -> float:
        """
        The high limit of the DOF. In meters if the joint is prismatic, in radians otherwise.
        If ``low_limit > high_limit``, then the DOF has no limits.
        """
    @high_limit.setter
    def high_limit(self, arg1: float) -> None:
        ...
    @property
    def low_limit(self) -> float:
        """
        The lower limit of the DOF. In meters if the joint is prismatic, in radians otherwise.
        """
    @low_limit.setter
    def low_limit(self, arg1: float) -> None:
        ...
    @property
    def name(self) -> str:
        """
        The name of the DOF.
        """
class ArticulationJointSensorHandle:
    """
    A class wrapping a handle for an articulation joint sensor.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationJointSensorHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the articulation joint sensor.
        """
class ArticulationKinematicSensorDef:
    """
    A class holding the data for a kinematic sensor definition on an articulation.
    """
    @property
    def flags(self) -> list[str]:
        """
        The flags of this kinematic sensor def.
        """
    @property
    def link_index(self) -> int:
        """
        The index of the link that the kinematic sensor def is attached to.
        """
    @property
    def link_name(self) -> str:
        """
        The name of the link that the kinematic sensor def is attached to.
        """
    @property
    def name(self) -> str:
        """
        The name of the kinematic sensor def.
        """
    @property
    def offset(self) -> Vec3:
        """
        The offset of the kinematic sensor def relative to the link COM.
        """
    @offset.setter
    def offset(self, arg1: Vec3) -> None:
        ...
class ArticulationKinematicStateCommand:
    """
    A command to access kinematic state information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_articulation_kinematic_state_command` instead to create these.
    
    The kinematic state consists of link transforms, link velocities, joint positions, and joint
    velocities.
    
    .. list-table:: Overview of GPU buffers in command
       :header-rows: 1
    
       * - GPU buffer
         - Type
         - Dimensions
       * - joint_positions_buffer
         - float32
         - num_envs x num_dofs
       * - joint_velocities_buffer
         - float32
         - num_envs x num_dofs
       * - link_transforms_buffer
         - float32
         - num_envs x num_links x 7
       * - link_velocities_buffer
         - float32
         - num_envs x num_links x 6
    
    This command is the same as using
    
    - :class:`JointStateCommand` command to access joint positions
    - :class:`JointStateCommand` command to access joint velocities
    - :class:`LinkTransformCommand` command to access link transforms
    - :class:`LinkVelocityCommand` command to access link velocities
    
    Combining these in one command is much faster and is the preferred method if more than one of the
    above commands is needed.
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def frame_type(self) -> FrameType:
        """
        Frame in which velocities are defined
        """
    @property
    def joint_end_index(self) -> int:
        """
        Exclusive last index of the joint to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` joints at indices 2, 3, and 4 will be accessed.
        """
    @property
    def joint_positions_data_ptr(self) -> int:
        """
        Gets the joint position data pointer of type `float32`.
        """
    @property
    def joint_start_index(self) -> int:
        """
        First index of the joint to be accessed.
        """
    @property
    def joint_velocities_data_ptr(self) -> int:
        """
        Gets the joint velocity data pointer of type `float32`.
        """
    @property
    def link_end_index(self) -> int:
        """
        Exclusive last index of the link to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``link_start_index = 2`` and ``link_end_index = 5`` links at indices 2, 3, and 4 will be accessed.
        """
    @property
    def link_start_index(self) -> int:
        """
        First index of the link to be accessed.
        """
    @property
    def link_transforms_data_ptr(self) -> int:
        """
        Gets the link transform data pointer of type ``Transform``.
        """
    @property
    def link_velocities_data_ptr(self) -> int:
        """
        Gets the link velocity data pointer of type ``SpatialVector``.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def transform_type(self) -> TransformType:
        """
        Type of transform (only for getting transforms)
        """
class ArticulationKinematicStateCommandGpuArray:
    """
    A GPU array of :class:`ArticulationKinematicStateCommand`.
    
    No constructor defined. Use :meth:`Gym.create_articulation_kinematic_state_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`ArticulationKinematicStateCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[ArticulationKinematicStateCommand]:
        """
        TODO: document
        """
class ArticulationLinkDef:
    """
    A class holding the data for an articulation Link definition.
    """
    @property
    def body_to_model(self) -> Transform:
        """
        Transform from body frame to model frame.
        
        This transform defines the offset to the COM of the body and additionally the rotational frame of
        the diagonalized inverse inertia matrix
        """
    @property
    def diag_inertia(self) -> Vec3:
        """
        Diagonalized inertia of link
        """
    @property
    def mass(self) -> float:
        """
        Mass of link
        
        Setting the mass also scales the diagonalized inertia tensor by the same factor.
        """
    @mass.setter
    def mass(self, arg1: float) -> None:
        ...
    @property
    def name(self) -> str:
        """
        The name of the link.
        """
class ArticulationMotorDefHandle:
    """
    A class wrapping a handle for an articulation motor definition.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationMotorDefHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the articulation motor control def.
        """
class ArticulationMotorHandle:
    """
    A class wrapping a handle for an articulation motor.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationMotorHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the articulation motor.
        """
class ArticulationPIDDefHandle:
    """
    A class wrapping a handle for an articulation motor definition.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationPIDDefHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the articulation PID control def.
        """
class ArticulationPIDHandle:
    """
    A class wrapping a handle for an articulation PID controller.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationPIDHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the PID controller.
        """
class ArticulationSpatialTendonHandle:
    """
    A class wrapping a handle for a spatial tendon.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ArticulationSpatialTendonHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the spatial tendon.
        """
class BoolGpuArray:
    """
    A GPU array of `bool`.
    
    No constructor defined. Use :meth:`Gym.create_bool_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of `bool` elements in the GPU array.
        """
    def data_ptr(self, offset: int = 0) -> int:
        """
        Gets the underlying GPU pointer cast to an integer type big enough to hold the pointer.
        
        You can specify an offset in the array using the optional ``offset`` argument.
        """
    def get(self) -> list[bool]:
        """
        Returns the GPU array data as a vector of `bool` elements on the CPU.
        """
    def set(self, host_array: list[bool]) -> None:
        """
        Fills the GPU array with data from the supplied ``host_array`` vector of `bool` elements on
        the CPU.
        """
class BoolGpuBufferWrapper:
    """
    Wrapper class for user-provided GPU buffer of 8-bit bools
    """
    def __init__(self, data_ptr: int, size: int) -> None:
        """
        Wraps a GPU buffer using its data pointer and size in bytes
        """
    @property
    def data_ptr(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
class BroadPhaseType:
    """
    The type of broad phase to be used.
    
    Members:
    
      ENVIRONMENT : Only objects within the same environment are checked for collisions.
    
      SCENE : All objects belonging to the same scene are checked for collisions.
    """
    ENVIRONMENT: typing.ClassVar[BroadPhaseType]  # value = <BroadPhaseType.ENVIRONMENT: 0>
    SCENE: typing.ClassVar[BroadPhaseType]  # value = <BroadPhaseType.SCENE: 1>
    __members__: typing.ClassVar[dict[str, BroadPhaseType]]  # value = {'ENVIRONMENT': <BroadPhaseType.ENVIRONMENT: 0>, 'SCENE': <BroadPhaseType.SCENE: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Camera:
    """
    A class holding the data for a camera instance.
    """
    @property
    def attach_camera_render_to_body(self) -> bool:
        """
        Whether to attach camera render to the body instead of the environment.
        """
    @attach_camera_render_to_body.setter
    def attach_camera_render_to_body(self, arg1: bool) -> None:
        ...
    @property
    def camera_def_index(self) -> int:
        """
        Index of the camera def of this camera.
        """
    @property
    def camera_def_name(self) -> str:
        """
        Name of the camera def of this camera.
        """
    @property
    def link_index(self) -> int:
        """
        Index of the link this camera is attached to.
        """
    @property
    def link_name(self) -> str:
        """
        Name of the link this camera is attached to.
        """
    @property
    def name(self) -> str:
        """
        Name of camera.
        """
    @property
    def relative_transform(self) -> Transform:
        """
        Transform of camera relative to link at :attr:`link_index`.
        """
    @relative_transform.setter
    def relative_transform(self, arg1: Transform) -> None:
        ...
    @property
    def render_height(self) -> float:
        """
        Height of camera render.
        """
    @render_height.setter
    def render_height(self, arg1: float) -> None:
        ...
    @property
    def render_relative_transform(self) -> Transform:
        """
        Transform of camera render relative to environment frame.
        """
    @render_relative_transform.setter
    def render_relative_transform(self, arg1: Transform) -> None:
        ...
    @property
    def render_width(self) -> float:
        """
        Width of camera render.
        """
    @render_width.setter
    def render_width(self, arg1: float) -> None:
        ...
class CameraPostFilter:
    """
    Members:
    
      ANTI_ALIASING
    
      NONE
    """
    ANTI_ALIASING: typing.ClassVar[CameraPostFilter]  # value = <CameraPostFilter.ANTI_ALIASING: 0>
    NONE: typing.ClassVar[CameraPostFilter]  # value = <CameraPostFilter.NONE: 1>
    __members__: typing.ClassVar[dict[str, CameraPostFilter]]  # value = {'ANTI_ALIASING': <CameraPostFilter.ANTI_ALIASING: 0>, 'NONE': <CameraPostFilter.NONE: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CameraType:
    """
    A flag enum to determine what type of controls the articulation can have.
    
    Members:
    
      PINHOLE : Pinhole camera type.
    
      RADIAL : Radial camera type.
    """
    PINHOLE: typing.ClassVar[CameraType]  # value = <CameraType.PINHOLE: 0>
    RADIAL: typing.ClassVar[CameraType]  # value = <CameraType.RADIAL: 1>
    __members__: typing.ClassVar[dict[str, CameraType]]  # value = {'PINHOLE': <CameraType.PINHOLE: 0>, 'RADIAL': <CameraType.RADIAL: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ContactFilterCommand:
    """
    A command for setting contact filters on force sensors. No constructor defined. Use :meth:`EnvironmentGroup.create_contact_filter_command` instead to create these.
    """
    @property
    def data_ptr(self) -> int:
        ...
    @property
    def sensor_handle(self) -> ArticulationForceSensorHandle:
        """
        :class:`ArticulationForceSensorHandle` of the force sensor being accessed.
        """
class ContactFilterCommandGpuArray:
    """
    A GPU array of :class:`ContactFilterCommand`.
    
    No constructor defined. Use :meth:`Gym.create_contact_filter_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        ...
class ContactFilterHandle:
    """
    A class wrapping a handle for a contact filter.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ContactFilterHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the contat filter.
        """
class Deformable(Model):
    """
    A class holding the data of an instantiated deformable.
    """
    def get_deformable_def(self, definition_index: int = 0) -> DeformableDef:
        """
        Gets the deformable def.
        """
    def get_deformable_def_handle(self, definition_index: int = 0) -> DeformableDefHandle:
        """
        Gets the deformable definition handle.
        
        It can be used to query :meth:`EnvironmentDef.get_deformable_def`.
        """
    def get_num_definitions(self) -> int:
        """
        Gets the number of rigid body definitions associated with this :class:`RigidBody`.
        """
    def get_transform_handle(self) -> TransformHandle:
        """
        Gets the transform handle of the rigid body.
        """
class DeformableDef(ModelDef):
    """
    A class that defines the whole structure of a single deformable.
    """
class DeformableDefHandle:
    """
    A class wrapping a handle for a deformable definition.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: DeformableDefHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the deformable definition.
        """
class DeformableHandle:
    """
    A class wrapping a handle for a rigid body instance.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: DeformableHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the rigid body instance
        """
    def is_valid(self) -> bool:
        """
        Returns ``True`` if the handle is valid and ``False`` otherwise.
        """
class DeformableMaterial:
    damping: float
    friction: float
    poisson_ratio: float
    youngs_modulus: float
    def __init__(self) -> None:
        """
        Default constructor
        """
class DeformableMaterialHandle:
    """
    A class wrapping a handle for an DeformableMaterial.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: DeformableMaterialHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the DeformableMaterial.
        """
class DeformableMaterialProperty:
    """
    Members:
    
      YOUNGS_MODULUS
    
      POISSON_RATIO
    
      FRICTION
    
      DAMPING
    """
    DAMPING: typing.ClassVar[DeformableMaterialProperty]  # value = <DeformableMaterialProperty.DAMPING: 2>
    FRICTION: typing.ClassVar[DeformableMaterialProperty]  # value = <DeformableMaterialProperty.FRICTION: 3>
    POISSON_RATIO: typing.ClassVar[DeformableMaterialProperty]  # value = <DeformableMaterialProperty.POISSON_RATIO: 1>
    YOUNGS_MODULUS: typing.ClassVar[DeformableMaterialProperty]  # value = <DeformableMaterialProperty.YOUNGS_MODULUS: 0>
    __members__: typing.ClassVar[dict[str, DeformableMaterialProperty]]  # value = {'YOUNGS_MODULUS': <DeformableMaterialProperty.YOUNGS_MODULUS: 0>, 'POISSON_RATIO': <DeformableMaterialProperty.POISSON_RATIO: 1>, 'FRICTION': <DeformableMaterialProperty.FRICTION: 3>, 'DAMPING': <DeformableMaterialProperty.DAMPING: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DeformableMaterialPropertyCommand:
    """
    A command to access deformable material property information. No constructor defined. Use :meth:`EnvironmentGroup.create_deformable_material_property_command` instead to create these.
    """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def deformable_material_handle(self) -> DeformableMaterialHandle:
        """
        :class:`DeformableMaterialHandle` of the deformable material being accessed.
        """
    @property
    def property(self) -> DeformableMaterialProperty:
        """
        Selects the property to be accessed.  See :class:`DeformableMaterialProperty` for options
        """
class DeformableMaterialPropertyCommandGpuArray:
    """
    A GPU array of :class:`DeformableMaterialPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_deformable_material_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`DeformableMaterialPropertyCommand` elements in the GPU array.
        """
class DeformableTransformCommand:
    """
    A command to access transform information of a rigid body. No constructor defined. Use :meth:`EnvironmentGroup.create_deformable_transform_command` instead to create these.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    """
    @property
    def data_ptr(self) -> int:
        """
        Pointer to the data of this command. It will either be used:
        
        - as a source to write data into rigid bodies in setter methods. The required memory size is then equal to ``sizeof(Transform) * num_environments``.
        - as a destination to store read rigid body data in getter methods. The required memory size is then equal to ``sizeof(Transform) * num_environments``.
        
        ``num_environments`` is equal to the total number of environments in the :class:`EnvironmentGroup`.
        """
    @property
    def frame_type(self) -> FrameType:
        """
        Frame of transform (only for getting transforms)
        """
    @property
    def indices_data_ptr(self) -> int:
        ...
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rigid_body_handle(self) -> DeformableHandle:
        """
        :class:`RigidBodyHandle` of the rigid body being accessed.
        """
    @property
    def rigid_body_handle_list(self) -> list[DeformableHandle]:
        ...
    @property
    def transform_type(self) -> TransformType:
        """
        Type of transform (only for getting transforms)
        """
class DeformableTransformCommandGpuArray:
    """
    A GPU array of :class:`DeformableTransformCommand`.
    
    No constructor defined. Use :meth:`Gym.create_deformable_transform_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`DeformableTransformCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[DeformableTransformCommand]:
        """
        TODO: document
        """
class DepthCamera(Camera):
    """
    A class holding the data for a depth camera instance.
    """
    @property
    def base_distance(self) -> float:
        """
        Distance reported for no-hits.
        """
    @base_distance.setter
    def base_distance(self, arg1: float) -> None:
        ...
    @property
    def render_max_depth(self) -> float:
        """
        Max depth of camera render range.
        """
    @render_max_depth.setter
    def render_max_depth(self, arg1: float) -> None:
        ...
    @property
    def render_min_depth(self) -> float:
        """
        Min depth of camera render range.
        """
    @render_min_depth.setter
    def render_min_depth(self, arg1: float) -> None:
        ...
class DepthCameraCommand:
    """
    A command for depth cameras.
    
    The ``data`` buffer must have size num_envs * resolution_y * resolution_x * 4 (float32) bytes.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def depth_camera_handle(self) -> DepthCameraHandle:
        """
        :class:`DepthCameraHandle` of the depth camera being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class DepthCameraCommandGpuArray:
    """
    A GPU array of :class:`DepthCameraCommand`.
    
    No constructor defined. Use :meth:`Gym.create_depth_camera_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`DepthCameraCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[DepthCameraCommand]:
        """
        TODO: document
        """
class DepthCameraDef:
    """
    A class holding the data for a camera definition.
    """
    @property
    def far_clip(self) -> float:
        """
        Far clip of camera def.
        """
    @far_clip.setter
    def far_clip(self, arg1: float) -> None:
        ...
    @property
    def field_of_view_x(self) -> float:
        """
        Field of view of camera def along x-axis in radians.
        """
    @field_of_view_x.setter
    def field_of_view_x(self, arg1: float) -> None:
        ...
    @property
    def field_of_view_y(self) -> float:
        """
        Field of view of camera def along y-axis in radians.
        """
    @field_of_view_y.setter
    def field_of_view_y(self, arg1: float) -> None:
        ...
    @property
    def has_stereo_filter(self) -> bool:
        """
        Whether camera def has stereo filter.
        """
    @property
    def name(self) -> str:
        """
        Name of camera def.
        """
    @property
    def post_filter(self) -> CameraPostFilter:
        """
        Post filter for camera rendering.
        """
    @post_filter.setter
    def post_filter(self, arg1: CameraPostFilter) -> None:
        ...
    @property
    def resolution_x(self) -> int:
        """
        Resolution of camera def along x-axis.
        """
    @resolution_x.setter
    def resolution_x(self, arg1: int) -> None:
        ...
    @property
    def resolution_y(self) -> int:
        """
        Resolution of camera def along y-axis.
        """
    @resolution_y.setter
    def resolution_y(self, arg1: int) -> None:
        ...
    @property
    def stereo_filter_spacing(self) -> float:
        """
        Spacing of stereoscopic filter.
        """
    @stereo_filter_spacing.setter
    def stereo_filter_spacing(self, arg1: float) -> None:
        ...
class DepthCameraHandle:
    """
    A class wrapping a handle for a depth camera.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: DepthCameraHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the depth camera.
        """
class DepthCameraTransformCommand:
    """
    A command to access depth camera transform information. No constructor defined. Use :meth:`EnvironmentGroup.create_depth_camera_transform_command` instead to create these.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def depth_camera_handle(self) -> DepthCameraHandle:
        """
        :class:`DepthCameraHandle` of the depth camera being accessed.
        """
class DepthCameraTransformCommandGpuArray:
    """
    A GPU array of :class:`DepthCameraTransformCommand`.
    
    No constructor defined. Use :meth:`Gym.create_depth_camera_transform_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`DepthCameraTransformCommand` elements in the GPU array.
        """
class ElasticMaterial:
    damping: float
    limit_damping: float
    limit_stiffness: float
    stiffness: float
    def __init__(self) -> None:
        """
        Default constructor
        """
class ElasticMaterialHandle:
    """
    A class wrapping a handle for an ElasticMaterial.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: ElasticMaterialHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the ElasticMaterial.
        """
class ElasticMaterialProperty:
    """
    Members:
    
      STIFFNESS
    
      DAMPING
    
      LIMIT_STIFFNESS
    
      LIMIT_DAMPING
    """
    DAMPING: typing.ClassVar[ElasticMaterialProperty]  # value = <ElasticMaterialProperty.DAMPING: 1>
    LIMIT_DAMPING: typing.ClassVar[ElasticMaterialProperty]  # value = <ElasticMaterialProperty.LIMIT_DAMPING: 3>
    LIMIT_STIFFNESS: typing.ClassVar[ElasticMaterialProperty]  # value = <ElasticMaterialProperty.LIMIT_STIFFNESS: 2>
    STIFFNESS: typing.ClassVar[ElasticMaterialProperty]  # value = <ElasticMaterialProperty.STIFFNESS: 0>
    __members__: typing.ClassVar[dict[str, ElasticMaterialProperty]]  # value = {'STIFFNESS': <ElasticMaterialProperty.STIFFNESS: 0>, 'DAMPING': <ElasticMaterialProperty.DAMPING: 1>, 'LIMIT_STIFFNESS': <ElasticMaterialProperty.LIMIT_STIFFNESS: 2>, 'LIMIT_DAMPING': <ElasticMaterialProperty.LIMIT_DAMPING: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ElasticMaterialPropertyCommand:
    """
    A command to access elastic material property information. No constructor defined. Use :meth:`EnvironmentGroup.create_elastic_material_property_command` instead to create these.
    """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def elastic_material_handle(self) -> ElasticMaterialHandle:
        """
        :class:`ElasticMaterialHandle` of the elastic material being accessed.
        """
    @property
    def property(self) -> ElasticMaterialProperty:
        """
        Selects the property to be accessed.  See :class:`ElasticMaterialProperty` for options
        """
class ElasticMaterialPropertyCommandGpuArray:
    """
    A GPU array of :class:`ElasticMaterialPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_elastic_material_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`ElasticMaterialPropertyCommand` elements in the GPU array.
        """
class Environment:
    """
    An Environment instance.
    """
    def get_environment_set(self) -> EnvironmentSet:
        """
        Gets the set that the environment belongs to.
        """
    def get_handle(self) -> EnvironmentHandle:
        """
        Gets the handle of this environment.
        """
    def get_transform(self) -> Transform:
        """
        Gets the :class:`Transform` offset of the specified environment.
        """
    def set_transform(self, transform: Transform) -> None:
        """
        Sets the :class:`Transform` offset of this environment.
        
        All rigid bodies and articulation current in the environment are offset as well.
        
        Note that the environment transform is normalized before it is applied to the environment.
        """
class EnvironmentDef(Finalizable):
    """
    Provides read-only access to articulation definitions, articulations, rigid body
    definitions, rigid bodies.
    """
    def assign_articulation_material(self, arti_model_def_handle: ArticulationDefHandle, rigid_material_handle: RigidMaterialHandle, link_index: int) -> None:
        ...
    def assign_deformable_material_to_deformable(self, deformable_model_def_handle: DeformableDefHandle, deformable_material_handle: DeformableMaterialHandle) -> None:
        ...
    def assign_rgb_material_to_articulation_link(self, arti_model_def_handle: ArticulationDefHandle, rgb_material_handle: RGBMaterialHandle, link_index: int) -> None:
        ...
    def assign_rgb_material_to_rigid_body(self, rigid_model_def_handle: RigidBodyDefHandle, rgb_material_handle: RGBMaterialHandle) -> None:
        ...
    def assign_rigid_body_material(self, rigid_body_def_handle: RigidBodyDefHandle, rigid_material_handle: RigidMaterialHandle) -> None:
        ...
    def assign_rigid_material_to_articulation_link(self, arti_model_def_handle: ArticulationDefHandle, rigid_material_handle: RigidMaterialHandle, link_index: int) -> None:
        ...
    def assign_rigid_material_to_rigid_body(self, rigid_model_def_handle: RigidBodyDefHandle, rigid_material_handle: RigidMaterialHandle) -> None:
        ...
    def create_articulation(self, arti_def_handle: ArticulationDefHandle, local_transform: Transform = ..., name: str = 'default') -> ArticulationHandle:
        """
        Creates an :class:`Articulation` instance from an :class:`ArticulationDef`.
        
        :param arti_def_handle: handle of :class:`ArticulationDef`
        :type arti_def_handle: ArticulationDefHandle
        
        :param local_transform: transform of articulation in environment space (default: identity transform)
        :type local_transform: Transform
        
        :param name: name of articulation (default: "default")
        :type name: str
        
        :return: handle of created :class:`Articulation`
        :rtype: ArticulationHandle
        """
    def create_box_def(self, half_size: Vec3 = ..., density: float = 800.0, fixed: bool = False, name: str = 'box', rigid_material_handle: RigidMaterialHandle = None, rgb_material_handle: RGBMaterialHandle = None) -> RigidBodyDefHandle:
        """
        Creates a rigid body definition of a box.
        
        :param half_size: vector defining box size along x, y, and z dimensions, respectively (default
            Vec3(0.5, 0.5, 0.5))
        :type half_size: Vec3
        :param density: density used in mass and inertia computation (default: 800.0)
        :type density: float
        :param fixed: is box fixed (default False)
        :type fixed: bool
        :param name: name of box (default: 'box')
        :type name: str
        :param rigid_material_handle: handle for rigid material.  If not given, the default rigid material
            will be used.
        :type rigid_material_handle: RigidMaterialHandle
        :param rgb_material_handle: handle for RGB material.  If not given, the default RGB material will be
            used
        :type rgb_material_handle: RGBMaterialHandle
        :return: rigid body def handle of box
        :rtype: RigidBodyDefHandle
        """
    def create_contact_filter(self, transform_handles: list[TransformHandle]) -> ContactFilterHandle:
        """
        Creates a contact filter from a set of transform handles.
        
        Use this when creating a contact filter command.
        
        :param transform_handles: list of transform handles
        :type transform_handles: list[TransformHandle]
        
        :return: handle of the contact filter
        :rtype: ContactFilterHandle
        """
    def create_deformable(self, deformable_def_handle: DeformableDefHandle, local_transform: Transform = ..., name: str = 'default') -> DeformableHandle:
        ...
    def create_deformable_material(self, deformable_material: DeformableMaterial) -> DeformableMaterialHandle:
        """
        Creates a deformable material in the :class:`EnvironmentDef`
        
        :param deformable_material: deformable material to be created
        :type deformable_material: DeformableMaterial
        :return: handle of deformable material
        :rtype: DeformableMaterialHandle
        """
    def create_elastic_material(self, elastic_material: ElasticMaterial) -> ElasticMaterialHandle:
        """
        Creates an elastic material in the :class:`EnvironmentDef`
        
        :param elastic_material: elastic material to be created
        :type elastic_material: ElasticMaterial
        :return: handle of elastic material
        :rtype: ElasticMaterialHandle
        """
    def create_hill_material(self, hill_material: HillMaterial) -> HillMaterialHandle:
        """
        Creates a Hill material in the :class:`EnvironmentDef`
        
        :param hill_material: Hill material to be created
        :type hill_material: HillMaterial
        :return: handle of Hill material
        :rtype: HillMaterialHandle
        """
    def create_plane_def(self, name: str = 'plane', rigid_material_handle: RigidMaterialHandle = None, rgb_material_handle: RGBMaterialHandle = None) -> RigidBodyDefHandle:
        """
        Creates a rigid body definition of a plane.
        
        Note that planes are fixed.
        
        :param name: name of plane (default: 'plane')
        :type name: str
        :param rigid_material_handle: handle for rigid material.  If not given, the default rigid material
            will be used.
        :type rigid_material_handle: RigidMaterialHandle
        :param rgb_material_handle: handle for RGB material.  If not given, the default RGB material will be
            used
        :type rgb_material_handle: RGBMaterialHandle
        :return: rigid body def handle of plane
        :rtype: RigidBodyDefHandle
        
        **Note**: the default orientation of the plane is facing the positive y-axis.  Rotate as needed when
        creating the rigid body.
        
        **Example for z-axis up**:
        
        ..  code-block:: python
        
            up_axis = Vec3(0, 0, 1)
            gym = create_gym(up_axis=up_axis)
        
            ...
        
            plane_def_handle = env_def.create_plane_def()
        
            rot = shortest_rotation(Vec3(0, 1, 0), up_axis)
            pos = Vec3(0)
            plane_transform = Transform(rot, pos)
        
            plane_handle = env_def.create_rigid_body(plane_def_handle, plane_transform)
        """
    def create_rgb_material(self, rgb_material: RGBMaterial) -> RGBMaterialHandle:
        """
        Creates an RGB material in the :class:`EnvironmentDef`
        
        :param rgb_material: RGB material to be created
        :type rgb_material: RGBMaterial
        :return: handle of RGB material
        :rtype: RGBMaterialHandle
        """
    @typing.overload
    def create_rigid_body(self, rigid_def_handle: RigidBodyDefHandle, local_transform: Transform = ..., name: str = 'default') -> RigidBodyHandle:
        """
        Creates a :class:`RigidBody` instance from a :class:`RigidBodyDef`.
        
        :param rigid_def_handle: handle of :class:`RigidBodyDef`
        :type rigid_def_handle: RigidBodyDefHandle
        
        :param local_transform: transform of rigid body in environment space (default: identity transform)
        :type local_transform: Transform
        
        :param name: name of rigid body (default: "default")
        :type name: str
        
        :return: handle of created :class:`RigidBody`
        :rtype: RigidBodyHandle
        """
    @typing.overload
    def create_rigid_body(self, rigid_def_handles: list[RigidBodyDefHandle], local_transform: Transform = ..., name: str = 'default') -> RigidBodyHandle:
        """
        Creates a swappable :class:`RigidBody` instance from a list of :class:`RigidBodyDef`.  Use the
        :class:`RigidBodyDefCommand` to swap definitions.
        
        .. note::
        
           Distance joints are currently not supported in swappable rigid body instances.
        
        .. note::
        
           If scene query is enabled using the ``enable_scene_query`` flag in :meth:`vlearn.create_gym()`,
           then the scene query geometries of the rigid body definitions need to be compatible.  Two
           definitions are compatible if they contain the same total number of query geometries, and the
           same counts of each geometry type.
        
        :param rigid_def_handles: list of handles of :class:`RigidBodyDef`
        :type rigid_def_handles: list[RigidBodyDefHandle]
        
        :param local_transform: transform of rigid body in environment space (default: identity transform)
        :type local_transform: Transform
        
        :param name: name of rigid body (default: "default")
        :type name: str
        
        :return: handle of created :class:`RigidBody`
        :rtype: RigidBodyHandle
        """
    def create_rigid_material(self, rigid_material: RigidMaterial) -> RigidMaterialHandle:
        """
        Creates a rigid material in the :class:`EnvironmentDef`
        
        :param rigid_material: rigid material to be created
        :type rigid_material: RigidMaterial
        :return: handle of rigid material
        :rtype: RigidMaterialHandle
        """
    def export_environment(self, file_path: str) -> bool:
        """
        Exports this environment definition into a vsim environment file.
        
        :param file_path: path to the desired output file, preferably with a .venv extension
        :type file_path: str
        
        :return: True if the export was successful, False otherwise
        :rtype: bool
        """
    def finalize(self) -> None:
        """
        Finalize so that the EnvironmentDef cannot be edited anymore.
        
        Must be called before creating an environment group from environment def.
        """
    def get_articulation(self, arti_handle: ArticulationHandle) -> Articulation:
        """
        Gets the :class:`Articulation` articulation instance for the specified
        ``arti_handle``.
        """
    def get_articulation_def(self, arti_model_def_handle: ArticulationDefHandle) -> ArticulationDef:
        """
        Gets the :class:`ArticulationDef` for the supplied ``arti_model_def_handle``.
        """
    def get_articulation_def_handle(self, articulation_def_index: int) -> ArticulationDefHandle:
        """
        Gets the handle of the :class:`ArticulationDef` at index
        ``articulation_def_index``.
        """
    def get_articulation_def_handle_by_name(self, model_name: str) -> ArticulationDefHandle:
        """
        Gets :class:`ArticulationDefHandle` by the name of the articulation definition.
        
        This is the preferred way of getting the handle of the articulation definition you just imported.
        E.g.
        
        .. code-block:: python
        
            num_arti, num_rigid = env_def.import_definitions(filename, fixed)
            assert num_arti == 1
            def_handle = env_def.get_articulation_def_handle_by_name("my_model")
        
        :param model_name: model name
        :type model_name: str
        
        :return: handle of articulation def
        :rtype: ArticulationDefHandle
        """
    def get_articulation_handle(self, articulation_index: int) -> ArticulationHandle:
        """
        Gets the handle of the :class:`Articulation` at index ``articulation_index``.
        """
    def get_articulation_handle_by_name(self, arti_name: str, silent: bool = False) -> ArticulationHandle:
        """
        Gets :class:`ArticulationHandle` by the name of the articulation.
        """
    def get_deformable(self, deformable_handle: DeformableHandle) -> Deformable:
        """
        Gets the :class:`Deformable` deformable instance for the specified ``deformable_handle``.
        """
    def get_deformable_def(self, deformable_def_handle: DeformableDefHandle) -> DeformableDef:
        """
        Gets the :class:`DeformableDef` for the supplied ``deformable_def_handle``.
        """
    def get_deformable_def_handle_by_name(self, model_name: str) -> DeformableDefHandle:
        """
        Gets :class:`DeformableDefHandle` by the name of the deformable.
        
        This is the preferred way of getting the handle of the model you just imported. E.g.
        
        .. code-block:: python
        
            num_arti, num_rigid = env_def.import_definitions(filename, fixed)
            assert num_rigid == 1
            def_handle = env_def.get_rigid_body_def_handle_by_name("my_model")
        """
    def get_deformable_material(self, deformable_material_handle: DeformableMaterialHandle) -> DeformableMaterial:
        """
        Gets the deformable material in the :class:`EnvironmentDef` for the given handle
        
        :param deformable_material_handle: handle of deformable material
        :type deformable_material_handle: DeformableMaterialHandle
        :return: deformable material
        :rtype: DeforambleMaterial
        """
    def get_elastic_material(self, elastic_material_handle: ElasticMaterialHandle) -> ElasticMaterial:
        """
        Gets the elastic material in the :class:`EnvironmentDef` for the given handle
        
        :param elastic_material_handle: handle of elastic material
        :type elastic_material_handle: ElasticMaterialHandle
        :return: elastic material
        :rtype: ElasticMaterial
        """
    def get_hill_material(self, hill_material_handle: HillMaterialHandle) -> HillMaterial:
        """
        Gets the Hill material in the :class:`EnvironmentDef` for the given handle
        
        :param hill_material_handle: handle of Hill material
        :type hill_material_handle: HillMaterialHandle
        :return: Hill material
        :rtype: HillMaterial
        """
    def get_name(self) -> str:
        """
        Gets the name of this environment definition.
        """
    def get_num_articulation_defs(self) -> int:
        """
        Gets the number of :class:`ArticulationDef` in the environment.
        """
    def get_num_articulations(self) -> int:
        """
        Gets the number of :class:`Articulation` in the environment.
        """
    def get_num_deformable_defs(self) -> int:
        """
        Gets the number of :class:`DeformableDef` in the environment.
        """
    def get_num_deformable_materials(self) -> int:
        """
        Gets the number of deformable materials defined in the :class:`EnvironmentDef`
        
        :return: number of deformable materials
        :rtype: int
        """
    def get_num_deformables(self) -> int:
        """
        Gets the number of :class:`Deformable` in the environment.
        """
    def get_num_elastic_materials(self) -> int:
        """
        Gets the number of elastic materials defined in the :class:`EnvironmentDef`
        
        :return: number of elastic materials
        :rtype: int
        """
    def get_num_hill_materials(self) -> int:
        """
        Gets the number of Hill materials defined in the :class:`EnvironmentDef`
        
        :return: number of Hill materials
        :rtype: int
        """
    def get_num_rgb_materials(self) -> int:
        """
        Gets the number of RGB materials defined in the :class:`EnvironmentDef`
        
        :return: number of RGB materials
        :rtype: int
        """
    def get_num_rigid_bodies(self) -> int:
        """
        Gets the number of :class:`RigidBody` in the environment.
        """
    def get_num_rigid_body_defs(self) -> int:
        """
        Gets the number of :class:`RigidBodyDef` in the environment.
        """
    def get_num_rigid_materials(self) -> int:
        """
        Gets the number of rigid materials defined in the :class:`EnvironmentDef`
        
        :return: number of rigid materials
        :rtype: int
        """
    def get_rgb_material(self, rgb_material_handle: RGBMaterialHandle) -> RGBMaterial:
        """
        Gets the RGB material in the :class:`EnvironmentDef` for the given handle
        
        :param rgb_material_handle: handle of RGB material
        :type rgb_material_handle: RGBMaterialHandle
        :return: RGB material
        :rtype: RGBMaterial
        """
    def get_rigid_body(self, rigid_handle: RigidBodyHandle) -> RigidBody:
        """
        Gets the :class:`RigidBody` rigid body instance for the specified ``rigid_handle``.
        """
    def get_rigid_body_def(self, rigid_body_def_handle: RigidBodyDefHandle) -> RigidBodyDef:
        """
        Gets the :class:`RigidBodyDef` for the supplied ``rigid_body_def_handle``.
        """
    def get_rigid_body_def_handle(self, rigid_body_def_index: int) -> RigidBodyDefHandle:
        """
        Gets the handle of the :class:`RigidBodyDef` at index ``rigid_body_def_index``.
        """
    def get_rigid_body_def_handle_by_name(self, model_name: str) -> RigidBodyDefHandle:
        """
        Gets :class:`RigidBodyDefHandle` by the name of the rigid body.
        
        This is the preferred way of getting the handle of the model you just imported. E.g.
        
        .. code-block:: python
        
            num_arti, num_rigid = env_def.import_definitions(filename, fixed)
            assert num_rigid == 1
            def_handle = env_def.get_rigid_body_def_handle_by_name("my_model")
        """
    def get_rigid_body_handle(self, rigid_body_index: int) -> RigidBodyHandle:
        """
        Gets the handle of the :class:`RigidBody` at index ``rigid_body_index``.
        """
    @typing.overload
    def get_rigid_body_handle_by_name(self, rigid_name: str, silent: bool = False) -> RigidBodyHandle:
        """
        Gets :class:`RigidBodyHandle` by the name of the articulation.
        """
    @typing.overload
    def get_rigid_body_handle_by_name(self, deformable_name: str, silent: bool = False) -> DeformableHandle:
        """
        Gets :class:`DeformableHandle` by the name of the deformable.
        """
    def get_rigid_distance_joint_handle(self, id: int) -> RigidDistanceJointHandle:
        """
        Get the distance joint given the id of all rigid body distance joints.
        """
    def get_rigid_material(self, rigid_material_handle: RigidMaterialHandle) -> RigidMaterial:
        """
        Gets the rigid material in the :class:`EnvironmentDef` for the given handle
        
        :param rigid_material_handle: handle of rigid material
        :type rigid_material_handle: RigidMaterialHandle
        :return: rigid material
        :rtype: RigidMaterial
        """
    def import_assembly(self, file_path: str, merge_meshes_inside_files: bool = True, use_visual_meshes: bool = True, force_inertia_computation: bool = False, force_mass_computation: bool = False) -> bool:
        """
        Imports contents of a Vsim environment file into this environment definition.
        
        :param file_path: path to the environment file.
        :type file_path: str
        
        :param merge_meshes_inside_files: merge all meshes in a geometry file into one (default: True)
        :type merge_meshes_inside_files: bool
        
        :param use_visual_meshes: whether to use visual meshes or collision meshes (default: True)
        :type use_visual_meshes: bool
        
        :return: True if the import was successful, False otherwise
        :rtype: bool
        """
    @typing.overload
    def import_definitions(self, filename: str, fixed: bool = False, use_visual_mesh: bool = False, merge_fixed_joints: bool = True, merge_meshes_inside_files: bool = True, force_mass_computation: bool = False, force_inertia_computation: bool = False, format: str = 'auto', alias: str = '', original_filename: str = '', scale: float = 1.0, density: float = 800.0, query_mode: QueryMode = ..., import_extra_mesh_data: bool = False) -> tuple[int, int]:
        """
        Import definitions from file.
        
        Definitions can be either articulations or rigid bodies.  Articulation definitions with a single
        link are automatically converted to rigid body definitions.
        
        :param filename: path to file.
        :type filename: str
        :param fixed: is articulation or rigid body fixed (default: False)
        :type fixed: bool
        :param use_visual_mesh: use visual mesh instead of collision mesh for render (default: False)
        :type use_visual_mesh: bool
        :param merge_fixed_joints: merge links connected by fixed joint (default: True)
        :type merge_fixed_joints: bool
        :param merge_meshes_inside_files: merge all meshes in a geometry file into one (default: True)
        :type merge_meshes_inside_files: bool
        :param force_mass_computation: force masses to be computed (default: False)
        :type force_mass_computation: bool
        :param force_inertia_computation: force inertia to be computed (default: False)
        :type force_inertia_computation: bool
        :param format: file format (default: "auto")
        :type format: str
        :param alias: prefix for definition name (default: '')
        :type alias: bool
        :param scale: scale dimensions by this value (default: 1)
        :type scale: float
        :param density: density used in mass and inertia computation (default: 800.0)
        :type density: float
        :param query_mode: which query import mode should be used (default: False)
        :type query_mode: QueryMode
        :param import_extra_mesh_data: whether to import extra non-physics data (e.g. smooth normals, UV
            coordinates) for meshes (default: False)
        :type density: float
        :return: A tuple containing:
            - (`int`) number of imported articulation definitions
            - (`int`) number of imported rigid body definitions
        :rtype: tuple[int, int]
        
        :valid format values:
            - `"urdf"`: URDF format
            - `"mjcf"`: MJCF format
            - `"vsim"`: VSIM format
            - `"auto"`: detect format based on filename extension
        """
    @typing.overload
    def import_definitions(self, filename: str, fixed: bool = False, use_visual_mesh: bool = False, merge_fixed_joints: bool = True, merge_meshes_inside_files: bool = True, force_mass_computation: bool = False, force_inertia_computation: bool = False, format: str = 'auto', alias: str = '', original_filename: str = '', scale: float = 1.0, density: float = 800.0, use_visual_for_query: bool = False, import_extra_mesh_data: bool = False) -> tuple[int, int]:
        """
        .. deprecated:: from version 0.2.6. Please use the overload with query_mode enum instead of use_visual_for_query boolean.
        
        Import definitions from file.
        
        Definitions can be either articulations or rigid bodies.  Articulation definitions with a single
        link are automatically converted to rigid body definitions.
        
        :param filename: path to file.
        :type filename: str
        :param fixed: is articulation or rigid body fixed (default: False)
        :type fixed: bool
        :param use_visual_mesh: use visual mesh instead of collision mesh for render (default: False)
        :type use_visual_mesh: bool
        :param merge_fixed_joints: merge links connected by fixed joint (default: True)
        :type merge_fixed_joints: bool
        :param merge_meshes_inside_files: merge all meshes in a geometry file into one (default: True)
        :type merge_meshes_inside_files: bool
        :param force_mass_computation: force masses to be computed (default: False)
        :type force_mass_computation: bool
        :param force_inertia_computation: force inertia to be computed (default: False)
        :type force_inertia_computation: bool
        :param format: file format (default: "auto")
        :type format: str
        :param alias: prefix for definition name (default: '')
        :type alias: bool
        :param scale: scale dimensions by this value (default: 1)
        :type scale: float
        :param density: density used in mass and inertia computation (default: 800.0)
        :type density: float
        :param use_visual_for_query: whether to use visual mesh for queries (default: False)
        :type density: bool
        :param import_extra_mesh_data: whether to import extra non-physics data (e.g. smooth normals, UV
            coordinates) for meshes (default: False)
        :type density: float
        :return: A tuple containing:
            - (`int`) number of imported articulation definitions
            - (`int`) number of imported rigid body definitions
        :rtype: tuple[int, int]
        
        :valid format values:
            - `"urdf"`: URDF format
            - `"mjcf"`: MJCF format
            - `"vsim"`: VSIM format
            - `"auto"`: detect format based on filename extension
        """
    def import_height_field_def(self, filename: str, min_height: float, max_height: float, scale: Vec3 = ..., name: str = 'height_field', rigid_material_handle: RigidMaterialHandle = None, rgb_material_handle: RGBMaterialHandle = None) -> RigidBodyDefHandle:
        """
        Import a height field def from PNG file.
        
        Note that height fields are always fixed.
        
        The origin of the height field will be at the top-left corner of the PNG image and zero height.
        
        **Note**: the default orientation of the height field is facing the positive y-axis.  Rotate as
        needed when creating the rigid body.
        
        :param filename: path to PNG file encoding height field
        :type filename: str
        :param min_height: minimum height
        :type min_height: float
        :param max_height: maximum height
        :type max_height: float
        :param scale: scale factors for each spatial dimension (default: no scaling)
        :type scale: Vec3
        :param name: name of height field (default: 'height_field')
        :type name: str
        :param rigid_material_handle: handle for rigid material.  If not given, the default rigid material
            will be used.
        :type rigid_material_handle: RigidMaterialHandle
        :param rgb_material_handle: handle for RGB material.  If not given, the default RGB material will be
            used
        :type rgb_material_handle: RGBMaterialHandle
        :return: rigid body def handle of triangle mesh
        :rtype: RigidBodyDefHandle
        """
    def import_triangle_mesh_def(self, filename: str, scale: Vec3 = ..., name: str = 'triangle_mesh', rigid_material_handle: RigidMaterialHandle = None, rgb_material_handle: RGBMaterialHandle = None) -> RigidBodyDefHandle:
        """
        Import a triangle mesh def from file
        
        Note that triangle meshes are always fixed.
        
        :param filename: path to file containing triangle mesh
        :type filename: str
        :param scale: scale factors for each spatial dimension
        :type scale: Vec3
        :param name: name of triangle mesh (default: 'triangle_mesh')
        :type name: str
        :param rigid_material_handle: handle for rigid material.  If not given, the default rigid material
            will be used.
        :type rigid_material_handle: RigidMaterialHandle
        :param rgb_material_handle: handle for RGB material.  If not given, the default RGB material will be
            used
        :type rgb_material_handle: RGBMaterialHandle
        :return: rigid body def handle of triangle mesh
        :rtype: RigidBodyDefHandle
        """
    def load_texture(self, path: str) -> TextureHandle:
        """
        Loads a texture file from the specified path and returns the handle that can be used to retrieve it.
        """
class EnvironmentDefHandle:
    """
    A class wrapping a handle for an environment definition.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: EnvironmentDefHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the environment definition.
        """
class EnvironmentGroup(Finalizable):
    """
    A collection owning a group of environment sets.
    """
    def __len__(self) -> int:
        """
        Gets the number of environment sets owned by the environment group.
        """
    def create_articulation_kinematic_state_command(self, joint_positions_buffer: Float32GpuBufferWrapper, joint_velocities_buffer: Float32GpuBufferWrapper, link_transforms_buffer: Float32GpuBufferWrapper, link_velocities_buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, dof_index_range: tuple[int, int] = ..., link_index_range: tuple[int, int] = ..., transform_type: TransformType = ..., frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None) -> ArticulationKinematicStateCommand:
        """
        Creates an :class:`ArticulationKinematicStateCommand` that acts on the environment group.
        
        Create an :class:`ArticulationKinematicStateCommandGpuArray` with
        :meth:`Gym.create_articulation_kinematic_state_command_gpu_array()`.
        
        :param joint_positions_buffer: buffer for reading or writing joint positions. Buffer size must be
            equal to ``num_envs * (dof_index_range[1] - dof_index_range[0])``
        :type joint_positions_buffer: Float32GpuBufferWrapper
        
        :param joint_velocities_buffer: buffer for reading or writing joint velocities. Buffer size must be
            equal to ``num_envs * (dof_index_range[1] - dof_index_range[0])``
        :type joint_velocities_buffer: Float32GpuBufferWrapper
        
        :param link_transforms_buffer: buffer for reading or writing link transforms. Buffer size must be
            equal to ``num_envs * (link_index_range[1] - link_index_range[0]) * 7``
        :type link_transforms_buffer: Float32GpuBufferWrapper
        
        :param link_velocities_buffer: buffer for reading or writing link velocities.  Buffer size must be
            equal to ``num_envs * (link_index_range[1] - link_index_range[0]) * 7``
        :type link_velocities_buffer: Float32GpuBufferWrapper
        
        :param articulation_handle: handle of articulation to act on
        :type articulation_handle: ArticulationHandle
        
        :param dof_index_range: the range of DOF indices to act on (default: ``[0, num_dofs)`` where
            ``num_dofs`` is the total number of DOFs)
        :type dof_index_range: Tuple[int, int]
        
        :param link_index_range: the range of link indices to act on (default: ``[0, num_links)`` where
            ``num_links`` is the total number of links)
        :type link_index_range: Tuple[int, int]
        
        :param transform_type: the type of transform to use (default: ``MODEL``).  Note: this is
            only used for getting transforms; setting transforms always assumes ``MODEL``.
        :type transform_type: TransformType
        
        :param frame_type: the frame used to read or write transforms and velocities (default:
            ``ENVIRONMENT``).
        :type frame_type: FrameType
        
        :param masks_buffer: masks buffer to control which environments to act on (default all active).
            Masks buffer size must be equal to ``num_envs``
        :type masks_buffer: BoolGpuBufferWrapper
        
        :return: command that can be used in an :class:`ArticulationKinematicStateCommandGpuArray` with
            :meth:`Gym.get_articulation_kinematic_states()` and
            :meth:`Gym.set_articulation_kinematic_states()`
        :rtype: ArticulationKinematicStateCommand
        """
    def create_contact_filter_command(self, buffer: Uint32GpuBufferWrapper, sensor_handle: ArticulationForceSensorHandle, contact_filter_handle_list: list[ContactFilterHandle]) -> ContactFilterCommand:
        """
        Creates a :class:`ContactFilterCommand` object.
        """
    def create_deformable_material_property_command(self, property: DeformableMaterialProperty, buffer: Float32GpuBufferWrapper, deformable_material_handle: DeformableMaterialHandle, masks_buffer: BoolGpuBufferWrapper = None) -> DeformableMaterialPropertyCommand:
        """
        Creates a :class:`DeformableMaterialPropertyCommand` object.
        """
    def create_deformable_transform_command(self, buffer: Float32GpuBufferWrapper, deformable_handle: DeformableHandle = None, transform_type: TransformType = ..., frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None, deformable_handle_list: list[DeformableHandle] = [], indices_buffer: Uint32GpuBufferWrapper = None) -> DeformableTransformCommand:
        """
        Creates a :class:`DeformableTransformCommand` object.
        """
    def create_depth_camera_command(self, buffer: Float32GpuBufferWrapper, depth_camera_handle: DepthCameraHandle, camera_type: CameraType = ...) -> DepthCameraCommand:
        """
        Creates a :class:`DepthCameraCommand` object using a `float32` GPU buffer.
        """
    def create_depth_camera_transform_command(self, buffer: Float32GpuBufferWrapper, depth_camera_handle: DepthCameraHandle, masks_buffer: BoolGpuBufferWrapper = None, camera_type: CameraType = ...) -> DepthCameraTransformCommand:
        """
        Creates a :class:`DepthCameraTransformCommand` object.
        """
    @typing.overload
    def create_depth_to_rgb_camera_command(self, buffer: Float32GpuBufferWrapper, depth_camera_handle: DepthCameraHandle, camera_type: CameraType = ...) -> DepthCameraCommand:
        """
        Creates a :class:`DepthCameraCommand` object using a `float32` GPU buffer.
        """
    @typing.overload
    def create_depth_to_rgb_camera_command(self, buffer: Uint8GpuBufferWrapper, depth_camera_handle: DepthCameraHandle, camera_type: CameraType = ...) -> DepthCameraCommand:
        """
        Creates a :class:`DepthCameraCommand` object using a `uint8` GPU buffer.
        """
    def create_elastic_material_property_command(self, property: ElasticMaterialProperty, buffer: Float32GpuBufferWrapper, elastic_material_handle: ElasticMaterialHandle, masks_buffer: BoolGpuBufferWrapper = None) -> ElasticMaterialPropertyCommand:
        """
        Creates a :class:`ElasticMaterialPropertyCommand` object.
        """
    def create_environment_set(self, num_envs: int, name: str = 'default_env_set') -> EnvironmentSetHandle:
        """
        Creates an environment set with `num_envs` environments in the environment group.
        
        This function cannot be called on finalized environment groups
        """
    def create_erase_patch_depth_command(self, depth_camera_handle: DepthCameraHandle, size_x: float, size_y: float) -> ErasePatchDepthCommand:
        """
        Creates a :class:`ErasePatchDepthCommand` that does not require an input buffer.
        """
    def create_erase_patch_rgb_command(self, rgb_camera_handle: RGBCameraHandle, size_x: float, size_y: float) -> ErasePatchRGBCommand:
        """
        Creates a :class:`ErasePatchRGBCommand` that does not require an input buffer.
        """
    def create_erase_pixels_depth_command(self, depth_camera_handle: DepthCameraHandle, percent: float) -> ErasePixelsDepthCommand:
        """
        Creates a :class:`ErasePixelsDepthCommand` that does not require an input buffer.
        """
    def create_erase_pixels_rgb_command(self, rgb_camera_handle: RGBCameraHandle, percent: float) -> ErasePixelsRGBCommand:
        """
        Creates a :class:`ErasePixelsRGBCommand` that does not require an input buffer.
        """
    def create_fixed_tendon_control_command(self, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> FixedTendonControlCommand:
        """
        Creates a :class:`FixedTendonControlCommand` object.
        
        Create a :class:`FixedTendonControlCommandGpuArray` with
        :meth:`Gym.create_fixed_tendon_control_command_gpu_array()`.
        
        :param buffer: buffer for writing tendon offsets or forces.  Buffer size must be
            equal to ``num_envs * (index_range[1] - index_range[0])``
        :type buffer: Float32GpuBufferWrapper
        
        :param articulation_handle: handle of articulation to act on
        :type articulation_handle: ArticulationHandle
        
        :param index_range: the range of fixed tendons to act on (default: ``[0,
            num_fixed_tendons)`` where ``num_fixed_tendons`` is the total number of fixed tendons)
        :type index_range: Tuple[int, int]
        
        :return: command that can be used in an :class:`FixedTendonControlCommandGpuArray` with
            :meth:`Gym.set_fixed_tendon_offsets()` and
            :meth:`Gym.set_fixed_tendon_forces()`.
        :rtype: FixedTendonControlCommand
        """
    def create_force_sensor_command(self, buffer: Float32GpuBufferWrapper, sensor_handle: ArticulationForceSensorHandle, masks_buffer: BoolGpuBufferWrapper = None) -> ForceSensorCommand:
        """
        Creates a :class:`ForceSensorCommand` object.
        """
    def create_gaussian_noise_depth_command(self, depth_camera_handle: DepthCameraHandle, mean: float, std: float) -> GaussianNoiseDepthCommand:
        """
        Creates a :class:`GaussianNoiseDepthCommand` that does not require an input buffer.
        """
    def create_gaussian_noise_rgb_command(self, rgb_camera_handle: RGBCameraHandle, mean: float, std: float) -> GaussianNoiseRGBCommand:
        """
        Creates a :class:`GaussianNoiseRGBCommand` that does not require an input buffer.
        """
    def create_gravity_compensation_command(self, joint_forces: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, gravity: Vec3) -> GravityCompensationCommand:
        """
        Creates a :class:`GravityCompensationCommand` object.
        
        :param joint_forces: output buffer containing computed joint forces for all dofs per
            environment. Size: `num_envs * num_dofs`.
        :type joint_forces: Float32GpuBufferWrapper
        :param articulation_handle: articulation handle
        :type articulation_handle: ArticulationHandle
        :param gravity: gravity vector
        :type gravity: Vec3
        """
    def create_inverse_dynamics_command(self, target_joint_accelerations: Float32GpuBufferWrapper, joint_forces: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, gravity: Vec3) -> InverseDynamicsCommand:
        """
        Creates a :class:`InverseDynamicsCommand` object.
        
        :param target_joint_accelerations: input buffer containing target joint accelerations for all dofs
            per environment. Size: `num_envs * num_dofs`.
        :type target_joint_accelerations: Float32GpuBufferWrapper
        :param joint_forces: output buffer containing computed joint forces for all dofs per environment.
            Size: `num_envs * num_dofs`.
        :type joint_forces: Float32GpuBufferWrapper
        :param articulation_handle: articulation handle
        :type articulation_handle: ArticulationHandle
        :param gravity: gravity vector
        :type gravity: Vec3
        """
    def create_inverse_kinematics_command(self, target_transforms: Float32GpuBufferWrapper, joint_positions: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, end_effector_indices: list[int], local_offsets: list[Vec3], max_num_iter: int = 50, is_orientation_dofs: list[bool] = []) -> InverseKinematicsCommand:
        """
        Creates a :class:`InverseKinematicsCommand` object.
        
        :param target_transforms: input buffer containing target transforms given as 7 floating point values
            per end effector per environment.  Size: ``num_envs * num_end_effectors * 7``.
        :type target_transforms: Float32GpuBufferWrapper
        :param joint_positions: output buffer containing computed joint positions for all dofs per
            environment. Size: ``num_envs * num_dofs``.
        :type joint_positions: Float32GpuBufferWrapper
        :param articulation_handle: articulation handle
        :type articulation_handle: ArticulationHandle
        :param end_effector_indices: list of end effector indices (currently only one end effector is
            supported)
        :type end_effector_indices: list[int]
        :param local_offsets: list of local offsets, one per end effector
        :type local_offsets: list[Vec3]
        :param max_num_iter: maximum number of iterations for IK solver (default 50)
        :type max_num_iter: int
        :param is_orientation_dofs: mask for orientation DOFs, must be either empty list (no orientation
            DOFs used) or list of size ``num_dofs`` (default empty list)
        :type is_orientation_dofs: list[bool]
        """
    def create_jacobian_command(self, jacobian_matrices: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, frame_type: FrameType = ...) -> JacobianCommand:
        """
        Creates a :class:`JacobianCommand` object.
        
        Fixed-base articulations: rows correspond to angular and linear velocity components of all links
        excluding the base link.  Columns correspond to joint DOFs.
        
        Free-base articulations: rows correspond to angular and linear velocity components of all links
        including the base link.  Columns correspond to angular and linear velocity components of base link
        followed by joint DOFs.
        
        :param jacobian_matrices: output buffer containing the computed Jacobian matrices per environment.
            Size for fixed-base articulations: `num_envs * (6 * (num_links - 1)) * num_dofs`.  Size for
            free-base articulations: `num_envs * (6 * num_links) * (6 + num_dofs)`
        :type jacobian_matrices: Float32GpuBufferWrapper
        :param articulation_handle: articulation handle
        :type articulation_handle: ArticulationHandle
        :param frame_type: select the rotational frame to represent the velocities in.  Valid values are
            `FrameType.ENVIRONMENT` (default) and `FrameType.WORLD`.
        :type frame_type: FrameType
        """
    def create_joint_active_mask_command(self, buffer: BoolGpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> JointActiveMaskCommand:
        """
        Creates a :class:`JointActiveMaskCommand` object.
        """
    def create_joint_dof_property_command(self, property: JointDofProperty, buffer: Float32GpuBufferWrapper, articulation_def_handle: ArticulationDefHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> JointDofPropertyCommand:
        """
        Creates a :class:`JointDofPropertyCommand` object.
        
        Args:
            property: choose which property to access
            buffer: GPU buffer used to read or write property values
            articulation_def_handle: handle of articulation def to be accessed
            index_range (optional): half-open range of dofs to access, default is [0, num_dofs)
            masks (optional): Boolean mask to control read/write access per environment
        """
    def create_joint_force_sensor_command(self, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> JointForceSensorCommand:
        """
        Creates a :class:`JointForceSensorCommand` object.
        """
    def create_joint_property_command(self, property: JointProperty, buffer: Float32GpuBufferWrapper, articulation_def_handle: ArticulationDefHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> JointPropertyCommand:
        """
        Creates a :class:`JointPropertyCommand` object.
        """
    def create_joint_state_command(self, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> JointStateCommand:
        """
        Creates a :class:`JointStateCommand` object.
        """
    def create_kinematic_sensor_state_command(self, transforms: Float32GpuBufferWrapper, velocities: Float32GpuBufferWrapper, sensor_handle: KinematicSensorHandle, frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None) -> KinematicSensorStateCommand:
        """
        Creates a :class:`KinematicSensorStateCommand` object.
        """
    def create_kinematic_sensor_transform_command(self, buffer: Float32GpuBufferWrapper, sensor_handle: KinematicSensorHandle, frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None) -> KinematicSensorTransformCommand:
        """
        Creates a :class:`KinematicSensorTransformCommand` object.
        """
    def create_kinematic_sensor_velocity_command(self, buffer: Float32GpuBufferWrapper, sensor_handle: KinematicSensorHandle, frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None) -> KinematicSensorVelocityCommand:
        """
        Creates a :class:`KinematicSensorVelocityCommand` object.
        """
    def create_light_transform_command(self, buffer: Float32GpuBufferWrapper, light_handle: LightHandle, masks_buffer: BoolGpuBufferWrapper = None) -> LightTransformCommand:
        """
        Creates a :class:`LightTransformCommand` object.
        """
    def create_link_external_force_command(self, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., frame_type: FrameType = ..., force_type: ForceType = ..., masks_buffer: BoolGpuBufferWrapper = None) -> LinkExternalForceCommand:
        """
        Creates a :class:`LinkExternalForceCommand` object.
        
        :param buffer: buffer for writing external forces. Buffer size must be great than or equal to
            ``num_envs * (index_range[1] - index_range[0]) * 6``
        :type buffer: Float32GpuBufferWrapper
        
        :param articulation_handle: handle of articulation
        :type articulation_handle: ArticulationHandle
        
        :param index_range: the range of link indices to act on (default: ``[0, num_links)`` where
            ``num_links`` is the total number of links
        :type index_range: Tuple[int, int]
        
        :param frame_type: the frame used to specify forces in (default: ``ENVIRONMENT``).
        :type frame_type: FrameType
        
        :param force_type: the format used to specify forces with (default: ``FORCE_TORQUE``).
        :type force_type: ForceType
        
        :param masks_buffer: masks buffer to control which environments to act on (default all active).
            Masks buffer size must be equal to ``num_envs``
        :type masks_buffer: BoolGpuBufferWrapper
        
        :return: command that can be used in an :class:`LinkExternalForceCommandGpuArray` with
            :meth:`Gym.set_link_external_forces()``
        :rtype: LinkExternalForceCommand
        """
    def create_link_property_command(self, property: LinkProperty, buffer: Float32GpuBufferWrapper, articulation_def_handle: ArticulationDefHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> LinkPropertyCommand:
        """
        Creates a :class:`LinkPropertyCommand` object.
        """
    def create_link_transform_command(self, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., transform_type: TransformType = ..., frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None) -> LinkTransformCommand:
        """
        Creates a :class:`LinkTransformCommand` object.
        """
    def create_link_velocity_command(self, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None) -> LinkVelocityCommand:
        """
        Creates a :class:`LinkVelocityCommand` object.
        """
    def create_mass_matrix_command(self, mass_matrices: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle) -> MassMatrixCommand:
        """
        Creates a :class:`MassMatrixCommand` object.
        
        :param mass_matrices: output buffer containing the computed generalized mass matrices per
            environment. Size: `num_envs * num_dofs * num_dofs`.
        :type mass_matrices: Float32GpuBufferWrapper
        :param articulation_handle: articulation handle
        :type articulation_handle: ArticulationHandle
        """
    def create_motor_control_command(self, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> MotorControlCommand:
        """
        Creates a :class:`MotorControlCommand` object.
        """
    def create_motor_property_command(self, property: MotorProperty, buffer: Float32GpuBufferWrapper, motor_control_def_handle: ArticulationMotorDefHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> MotorPropertyCommand:
        """
        Creates a :class:`MotorPropertyCommand` object.
        """
    def create_pid_control_command(self, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> PIDControlCommand:
        """
        Creates a :class:`PIDControlCommand` object.
        """
    def create_pid_property_command(self, property: PIDProperty, buffer: Float32GpuBufferWrapper, pid_control_def_handle: ArticulationPIDDefHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> PIDPropertyCommand:
        """
        Creates a :class:`PIDPropertyCommand` object.
        """
    def create_raycast_command(self, num_rays: int, origins_buffer: Float32GpuBufferWrapper, directions_buffer: Float32GpuBufferWrapper, is_hits_buffer: BoolGpuBufferWrapper, normals_buffer: Float32GpuBufferWrapper, distances_buffer: Float32GpuBufferWrapper, query_geometry_handle: VsQueryGeometryHandle = None) -> RaycastCommand:
        """
        Creates a :class:`RaycastCommand` object.
        """
    @typing.overload
    def create_rgb_camera_command(self, buffer: Float32GpuBufferWrapper, rgb_camera_handle: RGBCameraHandle, camera_type: CameraType = ...) -> RGBCameraCommand:
        """
        Creates a :class:`RGBCameraCommand` object using a `float32` GPU buffer.
        """
    @typing.overload
    def create_rgb_camera_command(self, buffer: Uint8GpuBufferWrapper, rgb_camera_handle: RGBCameraHandle, camera_type: CameraType = ...) -> RGBCameraCommand:
        """
        Creates a :class:`RGBCameraCommand` object using a `uint8` GPU buffer.
        """
    def create_rgb_camera_skybox_command(self, property: RGBCameraSkybox, buffer: Uint32GpuBufferWrapper, rgb_camera_def_handle: RGBCameraDefHandle, masks_buffer: BoolGpuBufferWrapper = None, camera_type: CameraType = ...) -> RGBCameraSkyboxCommand:
        """
        Creates a :class:`RGBCameraSkyboxCommand` object.
        """
    def create_rgb_camera_transform_command(self, buffer: Float32GpuBufferWrapper, rgb_camera_handle: RGBCameraHandle, masks_buffer: BoolGpuBufferWrapper = None, camera_type: CameraType = ...) -> RGBCameraTransformCommand:
        """
        Creates a :class:`RGBCameraTransformCommand` object.
        """
    def create_rgb_material_property_command(self, property: RGBMaterialProperty, buffer: Float32GpuBufferWrapper, rgb_material_handle: RGBMaterialHandle, masks_buffer: BoolGpuBufferWrapper = None) -> RGBMaterialPropertyCommand:
        """
        Creates a :class:`RGBMaterialPropertyCommand` object.
        """
    def create_rigid_body_def_command(self, buffer: Uint32GpuBufferWrapper, rigid_body_handle: RigidBodyHandle, masks_buffer: BoolGpuBufferWrapper = None) -> RigidBodyDefCommand:
        """
        Creates a :class:`RigidBodyDefCommand` object.
        
        This command swaps the :class:`RigidBodyDef` associated with the given :class:`RigidBody`.  The
        definitions are indexed in the same order as the list of :class:`RigidBodyDefHandle` provided to
        :meth:`EnvironmentDef.create_rigid_body()`.  For instance, index ``0`` corresponds to the first
        :class:`RigidBodyDefHandle`, index ``1`` corresponds to the second :class:`RigidBodyDefHandle`,
        and so on.
        
        Create a :class:`RigidBodyDefCommandGpuArray` with
        :meth:`Gym.create_rigid_body_def_command_gpu_array()`.
        
        :param buffer: buffer for writing definition indices.  Buffer size must be
            equal to ``num_envs``
        :type buffer: Uint32GpuBufferWrapper
        
        :param rigid_body_handle: handle of rigid body to act on
        :type rigid_body_handle: RigidBodyHandle
        
        :param masks_buffer: masks buffer to control which environments to act on (default all active).
            Masks buffer size must be equal to ``num_envs``
        :type masks_buffer: BoolGpuBufferWrapper
        """
    def create_rigid_body_external_force_command(self, buffer: Float32GpuBufferWrapper, rigid_body_handle: RigidBodyHandle = None, frame_type: FrameType = ..., force_type: ForceType = ..., masks_buffer: BoolGpuBufferWrapper = None, rigid_body_handle_list: list[RigidBodyHandle] = [], indices_buffer: Uint32GpuBufferWrapper = None) -> RigidBodyExternalForceCommand:
        """
        Creates a :class:`RigidBodyExternalForceCommand` object.
        
        You have to provide either a ``rigid_body_handle`` (non-indexed command) or the combination of a
        ``rigid_body_handle_list`` and ``indices_buffer`` (indexed command).
        
        :param buffer: buffer for writing external forces. Buffer size must be equal to
            ``num_envs * 6``
        :type buffer: Float32GpuBufferWrapper
        
        :param rigid_body_handle: handle of rigid body (default: ``None``)
        :type rigid_body_handle: RigidBodyHandle
        
        :param frame_type: the frame used to specify forces in (default: ``ENVIRONMENT``).
        :type frame_type: FrameType
        
        :param force_type: the format used to specify forces with (default: ``FORCE_TORQUE``).
        :type force_type: ForceType
        
        :param masks_buffer: masks buffer to control which environments to act on (default all active).
            Masks buffer size must be equal to ``num_envs``
        :type masks_buffer: BoolGpuBufferWrapper
        
        :param rigid_body_handle_list: a list of rigid body handles to select using ``indices_buffer``
            (default: ``[]``)
        :type rigid_body_handle_list: List[RigidBodyHandle]
        
        :param indices_buffer: buffer of indices for selecting rigid body handles.  Valid indices are
            in the range of ``[0, len(rigid_body_handle_list))``.  Buffer size must be equal
            to ``num_envs``. (default: ``None``).
        :type indices_buffer: Uint32GpuBufferWrapper
        
        :return: command that can be used in an :class:`RigidBodyExternalForceCommandGpuArray` with
            :meth:`Gym.set_rigid_body_external_forces()`
        :rtype: RigidBodyExternalForceCommand
        """
    def create_rigid_body_kinematic_state_command(self, transforms_buffer: Float32GpuBufferWrapper, velocities_buffer: Float32GpuBufferWrapper, rigid_body_handle: RigidBodyHandle = None, transform_type: TransformType = ..., frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None, rigid_body_handle_list: list[RigidBodyHandle] = [], indices_buffer: Uint32GpuBufferWrapper = None) -> RigidBodyKinematicStateCommand:
        """
        Creates a :class:`RigidBodyKinematicStateCommand` object.
        """
    def create_rigid_body_property_command(self, property: RigidBodyProperty, buffer: Float32GpuBufferWrapper, rigid_body_def_handle: RigidBodyDefHandle, masks_buffer: BoolGpuBufferWrapper = None) -> RigidBodyPropertyCommand:
        """
        Creates a :class:`RigidBodyPropertyCommand` object.
        """
    def create_rigid_body_transform_command(self, buffer: Float32GpuBufferWrapper, rigid_body_handle: RigidBodyHandle = None, transform_type: TransformType = ..., frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None, rigid_body_handle_list: list[RigidBodyHandle] = [], indices_buffer: Uint32GpuBufferWrapper = None) -> RigidBodyTransformCommand:
        """
        Creates a :class:`RigidBodyTransformCommand` object.
        """
    def create_rigid_body_velocity_command(self, buffer: Float32GpuBufferWrapper, rigid_body_handle: RigidBodyHandle = None, frame_type: FrameType = ..., masks_buffer: BoolGpuBufferWrapper = None, rigid_body_handle_list: list[RigidBodyHandle] = [], indices_buffer: Uint32GpuBufferWrapper = None) -> RigidBodyVelocityCommand:
        """
        Creates a :class:`RigidBodyVelocityCommand` object.
        """
    def create_rigid_distance_joint_property_command(self, property: RigidDistanceJointProperty, buffer: Float32GpuBufferWrapper, rigid_distance_joint_handle: RigidDistanceJointHandle, masks_buffer: BoolGpuBufferWrapper = None) -> RigidDistanceJointPropertyCommand:
        """
        Creates a :class:`RigidDistanceJointPropertyCommand` object.
        """
    def create_rigid_material_property_command(self, property: RigidMaterialProperty, buffer: Float32GpuBufferWrapper, rigid_material_handle: RigidMaterialHandle, masks_buffer: BoolGpuBufferWrapper = None) -> RigidMaterialPropertyCommand:
        """
        Creates a :class:`RigidMaterialPropertyCommand` object.
        """
    def create_segmented_depth_camera_command(self, buffer: Uint8GpuBufferWrapper, depth_camera_handle: DepthCameraHandle, segmented_handles: list[int], camera_type: CameraType = ...) -> SegmentedDepthCameraCommand:
        """
        Creates a :class:`SegmentedDepthCameraCommand` object using a `uint8` GPU buffer.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
    @typing.overload
    def create_segmented_depth_camera_to_rgb_command(self, buffer: Float32GpuBufferWrapper, depth_camera_handle: DepthCameraHandle, segmented_handles: list[int], camera_type: CameraType = ...) -> SegmentedDepthCameraCommand:
        """
        Creates a :class:`SegmentedDepthCameraCommand` object using a `float32` GPU buffer.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
    @typing.overload
    def create_segmented_depth_camera_to_rgb_command(self, buffer: Uint8GpuBufferWrapper, depth_camera_handle: DepthCameraHandle, segmented_handles: list[int], camera_type: CameraType = ...) -> SegmentedDepthCameraCommand:
        """
        Creates a :class:`SegmentedDepthCameraCommand` object using a `uint8` GPU buffer.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
    def create_segmented_rgb_camera_command(self, buffer: Uint8GpuBufferWrapper, rgb_camera_handle: RGBCameraHandle, segmented_handles: list[int], camera_type: CameraType = ...) -> SegmentedRGBCameraCommand:
        """
        Creates a :class:`SegmentedRGBCameraCommand` object using a `uint8` GPU buffer.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
    @typing.overload
    def create_segmented_rgb_camera_to_rgb_command(self, buffer: Float32GpuBufferWrapper, rgb_camera_handle: RGBCameraHandle, segmented_handles: list[int], camera_type: CameraType = ...) -> SegmentedRGBCameraCommand:
        """
        Creates a :class:`createSegmentedRGBToRGBCameraCommand` object using a `float32` GPU
        buffer.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
    @typing.overload
    def create_segmented_rgb_camera_to_rgb_command(self, buffer: Uint8GpuBufferWrapper, rgb_camera_handle: RGBCameraHandle, segmented_handles: list[int], camera_type: CameraType = ...) -> SegmentedRGBCameraCommand:
        """
        Creates a :class:`createSegmentedRGBToRGBCameraCommand` object using a `uint8` GPU
        buffer.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
    def create_spatial_tendon_control_command(self, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> SpatialTendonControlCommand:
        """
        Creates a :class:`SpatialTendonControlCommand` that acts on the environment group.
        
        Create a :class:`SpatialTendonControlCommandGpuArray` with
        :meth:`Gym.create_spatial_tendon_control_command_gpu_array()`.
        
        :param buffer: buffer for writing tendon offsets, forces, or activations.  Buffer size must be
            equal to ``num_envs * (index_range[1] - index_range[0])``
        :type buffer: Float32GpuBufferWrapper
        
        :param articulation_handle: handle of articulation to act on
        :type articulation_handle: ArticulationHandle
        
        :param index_range: the range of spatial tendons to act on (default: ``[0,
            num_spatial_tendons)`` where ``num_spatial_tendons`` is the total number of spatial tendons)
        :type index_range: Tuple[int, int]
        
        :return: command that can be used in an :class:`SpatialTendonControlCommandGpuArray` with
            :meth:`Gym.set_spatial_tendon_offsets()`,
            :meth:`Gym.set_spatial_tendon_forces()`, and
            :meth:`Gym.set_spatial_tendon_activations()`.
        :rtype: SpatialTendonControlCommand
        """
    def create_spatial_tendon_state_command(self, state: SpatialTendonState, buffer: Float32GpuBufferWrapper, articulation_handle: ArticulationHandle, index_range: tuple[int, int] = ..., masks_buffer: BoolGpuBufferWrapper = None) -> SpatialTendonStateCommand:
        """
        Creates a :class:`SpatialTendonStateCommand` that acts on the environment group.
        
        Create a :class:`SpatialTendonStateCommandGpuArray` with
        :meth:`Gym.create_spatial_tendon_state_command_gpu_array()`.
        
        :param buffer: buffer for getting tendon length, velocity, or force.  Buffer size must be
            equal to ``num_envs * (index_range[1] - index_range[0])``
        :type buffer: Float32GpuBufferWrapper
        
        :param articulation_handle: handle of articulation to act on
        :type articulation_handle: ArticulationHandle
        
        :param index_range: the range of spatial tendons to act on (default: ``[0,
            num_spatial_tendons)`` where ``num_spatial_tendons`` is the total number of spatial tendons)
        :type index_range: Tuple[int, int]
        
        :return: command that can be used in an :class:`SpatialTendonStateCommandGpuArray` with
            :meth:`Gym.get_spatial_tendon_states()`.
        :rtype: SpatialTendonStateCommand
        """
    def create_transform_command(self, buffer: Float32GpuBufferWrapper, transform_handle: TransformHandle) -> TransformCommand:
        """
        Creates a :class:`TransformCommand` object.
        """
    def finalize(self) -> None:
        """
        Finalize EnvironmentGroup to lock environment sets.
        
        Must be called before creating command.
        """
    def get_environment_def(self) -> EnvironmentDef:
        """
        Gets the read-only environment def
        """
    def get_environment_set(self, env_set_handle: EnvironmentSetHandle) -> EnvironmentSet:
        """
        Returns environment set for the given environment set handle
        """
    def get_environment_set_handle(self, environment_set_index: int) -> EnvironmentSetHandle:
        """
        Gets handle of the class :class:`EnvironmentSet` at index ``environment_set_index``.
        """
    def get_handle(self) -> EnvironmentGroupHandle:
        """
        Gets the handle of this :class:`EnvironmentGroup`
        """
    def get_name(self) -> str:
        """
        Gets the name of the environment group.
        """
    def get_num_environment_sets(self) -> int:
        """
        Gets the number of environment set owned by the environment group.
        """
    def get_num_environments(self) -> list[int]:
        """
        Gets the number of environments for each environment set owned by the environment group as a
        list
        """
    def get_total_num_environments(self) -> int:
        """
        Gets the total number of environments across all environment sets owned by the environment
        group.
        """
class EnvironmentGroupHandle:
    """
    A class wrapping a handle for an environment group.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: EnvironmentGroupHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def get_index(self) -> int:
        """
        Gets the underlying index of the environment group.
        """
class EnvironmentHandle:
    """
    A class wrapping a handle for an environment instance.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: EnvironmentHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def get_group_index(self) -> int:
        """
        Gets the underlying index of the environment group
        """
    def get_index(self) -> int:
        """
        Gets the underlying index of the environment
        """
    def get_set_index(self) -> int:
        """
        Gets the underlying index of the environment set
        """
class EnvironmentSet:
    """
    A collection owning a set of environments.
    """
    def __len__(self) -> int:
        """
        Gets the number of environments contained in the set.
        """
    def get_environment(self, handle: EnvironmentHandle) -> Environment:
        """
        Gets an environment by its ``handle``.
        """
    def get_environment_handle(self, environment_index: int) -> EnvironmentHandle:
        """
        Gets the handle of the :class:`Environment` at index ``environment_index``.
        """
    def get_handle(self) -> EnvironmentSetHandle:
        """
        Gets the handle of this :class:`EnvironmentSet`
        """
    def get_name(self) -> str:
        """
        Gets the name of this environment set.
        """
    def get_num_environments(self) -> int:
        """
        Gets the number of environments contained in the set.
        """
class EnvironmentSetHandle:
    """
    A class wrapping a handle for an environment set.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: EnvironmentSetHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def get_group_index(self) -> int:
        """
        Gets the underlying index of the environment set.
        """
    def get_index(self) -> int:
        """
        Gets the underlying index of the environment set.
        """
class ErasePatchDepthCommand:
    """
    A command for depth cameras.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def depth_camera_handle(self) -> DepthCameraHandle:
        """
        :class:`DepthCameraHandle` of the depth camera being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class ErasePatchDepthCommandGpuArray:
    """
    A GPU array of :class:`ErasePatchDepthCommand`.
    
    No constructor defined. Use :meth:`Gym.create_erase_patch_depth_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`ErasePatchDepthCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[ErasePatchDepthCommand]:
        """
        TODO: document
        """
class ErasePatchRGBCommand:
    """
    A command for rgb cameras.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rgb_camera_handle(self) -> RGBCameraHandle:
        """
        :class:`RGBCameraHandle` of the rgb camera being accessed.
        """
class ErasePatchRGBCommandGpuArray:
    """
    A GPU array of :class:`ErasePatchRGBCommand`.
    
    No constructor defined. Use :meth:`Gym.create_erase_patch_rgb_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`ErasePatchRGBCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[ErasePatchRGBCommand]:
        """
        TODO: document
        """
class ErasePixelsDepthCommand:
    """
    A command for depth cameras.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def depth_camera_handle(self) -> DepthCameraHandle:
        """
        :class:`DepthCameraHandle` of the depth camera being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class ErasePixelsDepthCommandGpuArray:
    """
    A GPU array of :class:`ErasePixelsDepthCommand`.
    
    No constructor defined. Use :meth:`Gym.create_erase_pixels_depth_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`ErasePixelsDepthCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[ErasePixelsDepthCommand]:
        """
        TODO: document
        """
class ErasePixelsRGBCommand:
    """
    A command for rgb cameras.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rgb_camera_handle(self) -> RGBCameraHandle:
        """
        :class:`RGBCameraHandle` of the rgb camera being accessed.
        """
class ErasePixelsRGBCommandGpuArray:
    """
    A GPU array of :class:`ErasePixelsRGBCommand`.
    
    No constructor defined. Use :meth:`Gym.create_erase_pixels_rgb_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`ErasePixelsRGBCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[ErasePixelsRGBCommand]:
        """
        TODO: document
        """
class Finalizable:
    """
    Base class for finalizable classes
    """
    def is_finalized(self) -> bool:
        """
        Returns whether this object is finalized.
        """
class FixedTendonControlCommand:
    """
    A command for setting the control values of fixed tendons.
    
    When used with :meth:`Gym.set_fixed_tendon_controls()`, fixed tendons solve a constraint where the
    weighted sum of joint positions should add up to the control value.  The control value is defaulted
    to zero.
    
    When used with :meth:`Gym.set_fixed_tendon_forces()`, the control values are interpreted as the
    force applied on the tendon.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_fixed_tendon_control_command` instead to create these.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def fixed_tendon_control_handle(self) -> ArticulationFixedTendonHandle:
        """
        :class:`ArticulationFixedTendonHandle` of the Fixed tendon control being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class FixedTendonControlCommandGpuArray:
    """
    A GPU array of :class:`FixedTendonControlCommand`.
    
    No constructor defined. Use :meth:`Gym.create_fixed_tendon_control_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`FixedTendonControlCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[FixedTendonControlCommand]:
        """
        TODO: document
        """
class FixedTendonDef(TendonDef):
    """
    A class holding the data for a fixed tendon definition.
    """
class FixedTendonProperty:
    """
    Members:
    
      LOW_LIMIT
    
      HIGH_LIMIT
    """
    HIGH_LIMIT: typing.ClassVar[FixedTendonProperty]  # value = <FixedTendonProperty.HIGH_LIMIT: 1>
    LOW_LIMIT: typing.ClassVar[FixedTendonProperty]  # value = <FixedTendonProperty.LOW_LIMIT: 0>
    __members__: typing.ClassVar[dict[str, FixedTendonProperty]]  # value = {'LOW_LIMIT': <FixedTendonProperty.LOW_LIMIT: 0>, 'HIGH_LIMIT': <FixedTendonProperty.HIGH_LIMIT: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class FixedTendonPropertyCommand:
    """
    TODO: document
    """
    @property
    def data_ptr(self) -> float:
        """
        TODO: document
        """
    @property
    def fixed_tendon_def_handle(self) -> ...:
        """
        TODO: document
        """
    @property
    def property(self) -> FixedTendonProperty:
        """
        TODO: document
        """
class FixedTendonPropertyCommandGpuArray:
    """
    A GPU array of :class:`FixedTendonPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_fixed_tendon_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`FixedTendonPropertyCommand` elements in the GPU array.
        """
class Float32GpuArray:
    """
    A GPU array of `float32`.
    
    No constructor defined. Use :meth:`Gym.create_float32_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of `float32` elements in the GPU array.
        """
    def data_ptr(self, offset: int = 0) -> int:
        """
        Gets the underlying GPU pointer cast to an integer type big enough to hold the pointer.
        
        You can specify an offset in the array using the optional ``offset`` argument.
        """
    def get(self) -> list[float]:
        """
        Returns the GPU array data as a vector of `float32` elements on the CPU.
        """
    def set(self, host_array: list[float]) -> None:
        """
        Fills the GPU array with data from the supplied ``host_array`` vector of `float32` elements on
        the CPU.
        """
class Float32GpuBufferWrapper:
    """
    Wrapper class for user-provided GPU buffer of 32-bit floats
    """
    def __init__(self, data_ptr: int, size: int) -> None:
        """
        Wraps a GPU buffer using its data pointer and size in bytes
        """
    @property
    def data_ptr(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
class ForceSensorCommand:
    """
    A command to access force sensor information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_force_sensor_command` instead to create these.
    
    :class:`SpatialVector` objects are written to and
    read from data buffers as ``(top.x, top.y, top.z, bottom.x, bottom.y, bottom.z)``.
    
    The linear force is stored in the ``top`` part of the :class:`SpatialVector`, and
    the angular torque is stored in the ``bottom`` part of the :class:`SpatialVector`.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``sensor_data`` cast to a number type.
        """
    @property
    def force_sensor_handle(self) -> ArticulationForceSensorHandle:
        """
        :class:`ArticulationForceSensorHandle` of the force sensor being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class ForceSensorCommandGpuArray:
    """
    A GPU array of :class:`ForceSensorCommand`.
    
    No constructor defined. Use :meth:`Gym.create_force_sensor_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`ForceSensorCommand` elements in the GPU array.
        """
class ForceSensorDef:
    """
    A class holding the data for a force sensor def.
    """
    @property
    def link_index(self) -> int:
        """
        The index of the link that the force sensor def is attached to.
        """
    @property
    def link_name(self) -> str:
        """
        The name of the link that the force sensor def is attached to.
        """
    @property
    def max_num_transform_handles(self) -> int:
        """
        Maximum number of transform handles in filter
        """
    @max_num_transform_handles.setter
    def max_num_transform_handles(self, arg1: int) -> None:
        ...
    @property
    def name(self) -> str:
        """
        The name of the force sensor def.
        """
    @property
    def offset(self) -> Vec3:
        """
        The offset of the force sensor def relative to the link COM.
        """
    @offset.setter
    def offset(self, arg1: Vec3) -> None:
        ...
class ForceType:
    """
    The format used to specify forces.
    
    Specify the frame in which to represent forces using :class:`FrameType`.
    
    
    Members:
    
      FORCE_TORQUE : Specify forces using linear force and torque.
    
      FORCE_POSITION : Specify forces using linear force and position.
    """
    FORCE_POSITION: typing.ClassVar[ForceType]  # value = <ForceType.FORCE_POSITION: 1>
    FORCE_TORQUE: typing.ClassVar[ForceType]  # value = <ForceType.FORCE_TORQUE: 0>
    __members__: typing.ClassVar[dict[str, ForceType]]  # value = {'FORCE_TORQUE': <ForceType.FORCE_TORQUE: 0>, 'FORCE_POSITION': <ForceType.FORCE_POSITION: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class FrameType:
    """
    The frame in which to represent transforms, velocities, forces, etc.
    
    Members:
    
      WORLD : World frame.
    
      ENVIRONMENT : Environment frame.
    
      LOCAL : Local frame.
    """
    ENVIRONMENT: typing.ClassVar[FrameType]  # value = <FrameType.ENVIRONMENT: 1>
    LOCAL: typing.ClassVar[FrameType]  # value = <FrameType.LOCAL: 2>
    WORLD: typing.ClassVar[FrameType]  # value = <FrameType.WORLD: 0>
    __members__: typing.ClassVar[dict[str, FrameType]]  # value = {'WORLD': <FrameType.WORLD: 0>, 'ENVIRONMENT': <FrameType.ENVIRONMENT: 1>, 'LOCAL': <FrameType.LOCAL: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GaussianNoiseDepthCommand:
    """
    A command for depth cameras.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def depth_camera_handle(self) -> DepthCameraHandle:
        """
        :class:`DepthCameraHandle` of the depth camera being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class GaussianNoiseDepthCommandGpuArray:
    """
    A GPU array of :class:`GaussianNoiseDepthCommand`.
    
    No constructor defined. Use :meth:`Gym.create_gaussian_noise_depth_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`GaussianNoiseDepthCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[GaussianNoiseDepthCommand]:
        """
        TODO: document
        """
class GaussianNoiseRGBCommand:
    """
    A command for rgb cameras.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rgb_camera_handle(self) -> RGBCameraHandle:
        """
        :class:`RGBCameraHandle` of the rgb camera being accessed.
        """
class GaussianNoiseRGBCommandGpuArray:
    """
    A GPU array of :class:`GaussianNoiseRGBCommand`.
    
    No constructor defined. Use :meth:`Gym.create_gaussian_noise_rgb_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`GaussianNoiseRGBCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[GaussianNoiseRGBCommand]:
        """
        TODO: document
        """
class GravityCompensationCommand:
    """
    A command for gravity compensation that computes the joint forces required to counteract
    gravity and returns the computed joint forces in the ``joint_forces`` GPU buffer.
    
    Created by :meth:`EnvironmentGroup.create_gravity_compensation_command()`
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
class GravityCompensationCommandGpuArray:
    """
    A GPU array of :class:`GravityCompensationCommand`.
    
    No constructor defined. Use :meth:`Gym.create_gravity_compensation_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        ...
class Gym:
    """
    The Gym class is the top-level interface for Vlearn and entry point for the
    creation of environment definitions and environment groups.
    
    Create a Gym instance with :func:`vlearn.create_gym`.
    
    Get the current Gym instance with :func:`vlearn.get_gym`.
    
    Delete the current Gym instance with :func:`vlearn.delete_gym`.
    """
    def _check_for_cuda_errors(self, message: str = '') -> None:
        """
        Prints cuda error message if found
        
        Warning: use ONLY for debugging.  This method synchronizes the device and
        therefore introduces significant latency.
        """
    def _find_closest_point(self, p: Vec3, closestP: Vec3, gradient: Vec3) -> None:
        ...
    def _step(self) -> None:
        """
        Steps the Gym's simulation
        """
    def apply_depth_camera_erase_patch(self, array: ErasePatchDepthCommandGpuArray) -> None:
        """
        erase patch filter
        """
    def apply_depth_camera_erase_pixels(self, array: ErasePixelsDepthCommandGpuArray) -> None:
        """
        erase pixels filter
        """
    def apply_depth_camera_gaussian_noise(self, array: GaussianNoiseDepthCommandGpuArray) -> None:
        """
        Gaussian noise filter
        """
    def apply_rgb_camera_erase_patch(self, array: ErasePatchRGBCommandGpuArray) -> None:
        """
        erase patch filter
        """
    def apply_rgb_camera_erase_pixels(self, array: ErasePixelsRGBCommandGpuArray) -> None:
        """
        erase pixels filter
        """
    def apply_rgb_camera_gaussian_noise(self, array: GaussianNoiseRGBCommandGpuArray) -> None:
        """
        Gaussian noise filter
        """
    def articulation_component_finalize(self) -> None:
        """
        Finalizes all articulations, uploading their data to the GPU.
        """
    def clone_environment_def(self, env_def_handle: EnvironmentDefHandle, name: str = 'default_env_def') -> EnvironmentDefHandle:
        """
        Clone environment definition, resulting in an unfinalized copy of the original environment
        def.
        
        :param env_def_handle: handle of environment def to be copied
        :type env_def_handle: EnvironmentDefHandle
        :param name: name of new environment def (default: "default_env_def")
        :type name: str
        :return: environment definition handle of new environment def
        :rtype: EnvironmentDefHandle
        """
    def compute_gravity_compensations(self, array: GravityCompensationCommandGpuArray) -> None:
        """
        Compute gravity compensation terms.
        """
    def compute_inverse_dynamics(self, array: InverseDynamicsCommandGpuArray) -> None:
        """
        Compute inverse dynamics.
        """
    def compute_inverse_kinematics(self, array: InverseKinematicsCommandGpuArray) -> None:
        """
        Note: only one effector end link is currently supported.
        """
    def compute_jacobians(self, array: JacobianCommandGpuArray) -> None:
        """
        Compute Jacobian matrices.
        """
    def compute_kinematics(self) -> None:
        """
        Computes the kinematic state for actors (rigid bodies and articulations) and updates the
        renderer if one is present.
        
        This should be called after creating or modifying actor definitions.
        """
    def compute_mass_matrices(self, array: MassMatrixCommandGpuArray) -> None:
        """
        Compute generalized mass matrices.
        """
    def convert_mjcf_to_vsim(self, mjcf_file: str, vsim_file: str, model_index: int) -> None:
        """
        Converts MJCF file to VSIM file.
        
        Since MJCF files may contain multiple models inside them, the `model_index` argument controls which one gets exported into the VSIM file.
        
        :param mjcf_file: input MJCF file
        :type mjcf_file: str
        :param vsim_file: output VSIM file
        :type vsim_file: str
        :param model_index: index of model within MJCF file to be converted
        :type model_index: int
        """
    def convert_urdf_to_vsim(self, urdf_file: str, vsim_file: str) -> None:
        """
        Converts URDF file to VSIM file.
        
        :param urdf_file: input URDF file
        :type urdf_file: str
        :param vsim_file: output VSIM file
        :type vsim_file: str
        """
    def convert_vsim_to_vsim(self, in_file: str, out_file: str) -> None:
        """
        Converts VSIM file to VSIM file.
        
        Useful for adding collision pair filters to an existing VSIM model file.
        
        :param in_file: input VSIM file
        :type in_file: str
        :param out_file: output VSIM file
        :type out_file: str
        """
    def create_articulation_kinematic_state_command_gpu_array(self, host_array: list[ArticulationKinematicStateCommand]) -> ArticulationKinematicStateCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`ArticulationKinematicStateCommand` objects.
        
        :param host_array: host array of :class:`ArticulationKinematicStateCommand`
        :type host_array: List[ArticulationKinematicStateCommand]
        :return: GPU array of :class:`ArticulationKinematicStateCommand`
        :rtype: ArticulationKinematicStateCommandGpuArray
        """
    @typing.overload
    def create_bool_gpu_array(self, host_array: list[bool]) -> BoolGpuArray:
        """
        Creates a GPU array from a host array of :class:`bool` objects.
        
        :param host_array: host array of :class:`bool`
        :type host_array: List[bool]
        :return: GPU array of :class:`bool`
        :rtype: boolGpuArray
        """
    @typing.overload
    def create_bool_gpu_array(self, num_elements: int) -> BoolGpuArray:
        """
        Creates a GPU array of size ``num_elements`` of :class:`bool` elements.
        """
    def create_contact_filter_command_gpu_array(self, host_array: list[ContactFilterCommand]) -> ContactFilterCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`ContactFilterCommand` objects.
        
        :param host_array: host array of :class:`ContactFilterCommand`
        :type host_array: List[ContactFilterCommand]
        :return: GPU array of :class:`ContactFilterCommand`
        :rtype: ContactFilterCommandGpuArray
        """
    def create_deformable_material_property_command_gpu_array(self, host_array: list[DeformableMaterialPropertyCommand]) -> DeformableMaterialPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`DeformableMaterialPropertyCommand` objects.
        
        :param host_array: host array of :class:`DeformableMaterialPropertyCommand`
        :type host_array: List[DeformableMaterialPropertyCommand]
        :return: GPU array of :class:`DeformableMaterialPropertyCommand`
        :rtype: DeformableMaterialPropertyCommandGpuArray
        """
    def create_deformable_transform_command_gpu_array(self, host_array: list[DeformableTransformCommand], masks_buffer: BoolGpuBufferWrapper = None) -> DeformableTransformCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`DeformableTransformCommand` objects.
        
        :param host_array: host array of :class:`DeformableTransformCommand`
        :type host_array: List[DeformableTransformCommand]
        :return: GPU array of :class:`DeformableTransformCommand`
        :rtype: DeformableTransformCommandGpuArray
        """
    def create_depth_camera_command_gpu_array(self, host_array: list[DepthCameraCommand]) -> DepthCameraCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`DepthCameraCommand` objects.
        
        :param host_array: host array of :class:`DepthCameraCommand`
        :type host_array: List[DepthCameraCommand]
        :return: GPU array of :class:`DepthCameraCommand`
        :rtype: DepthCameraCommandGpuArray
        """
    def create_depth_camera_transform_command_gpu_array(self, host_array: list[DepthCameraTransformCommand]) -> DepthCameraTransformCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`DepthCameraTransformCommand` objects.
        
        :param host_array: host array of :class:`DepthCameraTransformCommand`
        :type host_array: List[DepthCameraTransformCommand]
        :return: GPU array of :class:`DepthCameraTransformCommand`
        :rtype: DepthCameraTransformCommandGpuArray
        """
    def create_elastic_material_property_command_gpu_array(self, host_array: list[ElasticMaterialPropertyCommand]) -> ElasticMaterialPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`ElasticMaterialPropertyCommand` objects.
        
        :param host_array: host array of :class:`ElasticMaterialPropertyCommand`
        :type host_array: List[ElasticMaterialPropertyCommand]
        :return: GPU array of :class:`ElasticMaterialPropertyCommand`
        :rtype: ElasticMaterialPropertyCommandGpuArray
        """
    def create_environment_def(self, name: str = 'default_env_def') -> EnvironmentDefHandle:
        """
        Create environment definition.
        
        :param name: name for environment definition (default: "default_env_def")
        :type name: str
        :return: environment definition handle
        :rtype: EnvironmentDefHandle
        """
    def create_environment_group(self, env_def_handle: EnvironmentDefHandle, name: str = 'default_env_group') -> EnvironmentGroupHandle:
        """
        Creates an environment group from the given :class:`EnvironmentDefHandle`.
        """
    def create_erase_patch_depth_command_gpu_array(self, host_array: list[ErasePatchDepthCommand]) -> ErasePatchDepthCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`ErasePatchDepthCommand` objects.
        
        :param host_array: host array of :class:`ErasePatchDepthCommand`
        :type host_array: List[ErasePatchDepthCommand]
        :return: GPU array of :class:`ErasePatchDepthCommand`
        :rtype: ErasePatchDepthCommandGpuArray
        """
    def create_erase_patch_rgb_command_gpu_array(self, host_array: list[ErasePatchRGBCommand]) -> ErasePatchRGBCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`ErasePatchRGBCommand` objects.
        
        :param host_array: host array of :class:`ErasePatchRGBCommand`
        :type host_array: List[ErasePatchRGBCommand]
        :return: GPU array of :class:`ErasePatchRGBCommand`
        :rtype: ErasePatchRGBCommandGpuArray
        """
    def create_erase_pixels_depth_command_gpu_array(self, host_array: list[ErasePixelsDepthCommand]) -> ErasePixelsDepthCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`ErasePixelsDepthCommand` objects.
        
        :param host_array: host array of :class:`ErasePixelsDepthCommand`
        :type host_array: List[ErasePixelsDepthCommand]
        :return: GPU array of :class:`ErasePixelsDepthCommand`
        :rtype: ErasePixelsDepthCommandGpuArray
        """
    def create_erase_pixels_rgb_command_gpu_array(self, host_array: list[ErasePixelsRGBCommand]) -> ErasePixelsRGBCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`ErasePixelsRGBCommand` objects.
        
        :param host_array: host array of :class:`ErasePixelsRGBCommand`
        :type host_array: List[ErasePixelsRGBCommand]
        :return: GPU array of :class:`ErasePixelsRGBCommand`
        :rtype: ErasePixelsRGBCommandGpuArray
        """
    def create_fixed_tendon_control_command_gpu_array(self, host_array: list[FixedTendonControlCommand]) -> FixedTendonControlCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`FixedTendonControlCommand` objects.
        
        :param host_array: host array of :class:`FixedTendonControlCommand`
        :type host_array: List[FixedTendonControlCommand]
        :return: GPU array of :class:`FixedTendonControlCommand`
        :rtype: FixedTendonControlCommandGpuArray
        """
    def create_fixed_tendon_property_command_gpu_array(self, host_array: list[FixedTendonPropertyCommand]) -> FixedTendonPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`FixedTendonPropertyCommand` objects.
        
        :param host_array: host array of :class:`FixedTendonPropertyCommand`
        :type host_array: List[FixedTendonPropertyCommand]
        :return: GPU array of :class:`FixedTendonPropertyCommand`
        :rtype: FixedTendonPropertyCommandGpuArray
        """
    @typing.overload
    def create_float32_gpu_array(self, host_array: list[float]) -> Float32GpuArray:
        """
        Creates a GPU array from a host array of :class:`VsReal` objects.
        
        :param host_array: host array of :class:`VsReal`
        :type host_array: List[VsReal]
        :return: GPU array of :class:`VsReal`
        :rtype: VsRealGpuArray
        """
    @typing.overload
    def create_float32_gpu_array(self, num_elements: int) -> Float32GpuArray:
        """
        Creates a GPU array of size ``num_elements`` of :class:`VsReal` elements.
        """
    def create_force_sensor_command_gpu_array(self, host_array: list[ForceSensorCommand]) -> ForceSensorCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`ForceSensorCommand` objects.
        
        :param host_array: host array of :class:`ForceSensorCommand`
        :type host_array: List[ForceSensorCommand]
        :return: GPU array of :class:`ForceSensorCommand`
        :rtype: ForceSensorCommandGpuArray
        """
    def create_gaussian_noise_depth_command_gpu_array(self, host_array: list[GaussianNoiseDepthCommand]) -> GaussianNoiseDepthCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`GaussianNoiseDepthCommand` objects.
        
        :param host_array: host array of :class:`GaussianNoiseDepthCommand`
        :type host_array: List[GaussianNoiseDepthCommand]
        :return: GPU array of :class:`GaussianNoiseDepthCommand`
        :rtype: GaussianNoiseDepthCommandGpuArray
        """
    def create_gaussian_noise_rgb_command_gpu_array(self, host_array: list[GaussianNoiseRGBCommand]) -> GaussianNoiseRGBCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`GaussianNoiseRGBCommand` objects.
        
        :param host_array: host array of :class:`GaussianNoiseRGBCommand`
        :type host_array: List[GaussianNoiseRGBCommand]
        :return: GPU array of :class:`GaussianNoiseRGBCommand`
        :rtype: GaussianNoiseRGBCommandGpuArray
        """
    def create_gravity_compensation_command_gpu_array(self, host_array: list[GravityCompensationCommand]) -> GravityCompensationCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`GravityCompensationCommand` objects.
        
        :param host_array: host array of :class:`GravityCompensationCommand`
        :type host_array: List[GravityCompensationCommand]
        :return: GPU array of :class:`GravityCompensationCommand`
        :rtype: GravityCompensationCommandGpuArray
        """
    def create_hill_material_property_command_gpu_array(self, host_array: list[HillMaterialPropertyCommand]) -> HillMaterialPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`HillMaterialPropertyCommand` objects.
        
        :param host_array: host array of :class:`HillMaterialPropertyCommand`
        :type host_array: List[HillMaterialPropertyCommand]
        :return: GPU array of :class:`HillMaterialPropertyCommand`
        :rtype: HillMaterialPropertyCommandGpuArray
        """
    def create_inverse_dynamics_command_gpu_array(self, host_array: list[InverseDynamicsCommand]) -> InverseDynamicsCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`InverseDynamicsCommand` objects.
        
        :param host_array: host array of :class:`InverseDynamicsCommand`
        :type host_array: List[InverseDynamicsCommand]
        :return: GPU array of :class:`InverseDynamicsCommand`
        :rtype: InverseDynamicsCommandGpuArray
        """
    def create_inverse_kinematics_command_gpu_array(self, host_array: list[InverseKinematicsCommand]) -> InverseKinematicsCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`InverseKinematicsCommand` objects.
        
        :param host_array: host array of :class:`InverseKinematicsCommand`
        :type host_array: List[InverseKinematicsCommand]
        :return: GPU array of :class:`InverseKinematicsCommand`
        :rtype: InverseKinematicsCommandGpuArray
        """
    def create_jacobian_command_gpu_array(self, host_array: list[JacobianCommand]) -> JacobianCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`JacobianCommand` objects.
        
        :param host_array: host array of :class:`JacobianCommand`
        :type host_array: List[JacobianCommand]
        :return: GPU array of :class:`JacobianCommand`
        :rtype: JacobianCommandGpuArray
        """
    def create_joint_active_mask_command_gpu_array(self, host_array: list[JointActiveMaskCommand]) -> JointActiveMaskCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`JointActiveMaskCommand` objects.
        
        :param host_array: host array of :class:`JointActiveMaskCommand`
        :type host_array: List[JointActiveMaskCommand]
        :return: GPU array of :class:`JointActiveMaskCommand`
        :rtype: JointActiveMaskCommandGpuArray
        """
    def create_joint_dof_property_command_gpu_array(self, host_array: list[JointDofPropertyCommand]) -> JointDofPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`JointDofPropertyCommand` objects.
        
        :param host_array: host array of :class:`JointDofPropertyCommand`
        :type host_array: List[JointDofPropertyCommand]
        :return: GPU array of :class:`JointDofPropertyCommand`
        :rtype: JointDofPropertyCommandGpuArray
        """
    def create_joint_force_sensor_command_gpu_array(self, host_array: list[JointForceSensorCommand]) -> JointForceSensorCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`JointForceSensorCommand` objects.
        
        :param host_array: host array of :class:`JointForceSensorCommand`
        :type host_array: List[JointForceSensorCommand]
        :return: GPU array of :class:`JointForceSensorCommand`
        :rtype: JointForceSensorCommandGpuArray
        """
    def create_joint_property_command_gpu_array(self, host_array: list[JointPropertyCommand]) -> JointPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`JointPropertyCommand` objects.
        
        :param host_array: host array of :class:`JointPropertyCommand`
        :type host_array: List[JointPropertyCommand]
        :return: GPU array of :class:`JointPropertyCommand`
        :rtype: JointPropertyCommandGpuArray
        """
    def create_joint_state_command_gpu_array(self, host_array: list[JointStateCommand]) -> JointStateCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`JointStateCommand` objects.
        
        :param host_array: host array of :class:`JointStateCommand`
        :type host_array: List[JointStateCommand]
        :return: GPU array of :class:`JointStateCommand`
        :rtype: JointStateCommandGpuArray
        """
    def create_kinematic_sensor_state_command_gpu_array(self, host_array: list[KinematicSensorStateCommand]) -> KinematicSensorStateCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`KinematicSensorStateCommand` objects.
        
        :param host_array: host array of :class:`KinematicSensorStateCommand`
        :type host_array: List[KinematicSensorStateCommand]
        :return: GPU array of :class:`KinematicSensorStateCommand`
        :rtype: KinematicSensorStateCommandGpuArray
        """
    def create_kinematic_sensor_transform_command_gpu_array(self, host_array: list[KinematicSensorTransformCommand]) -> KinematicSensorTransformCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`KinematicSensorTransformCommand` objects.
        
        :param host_array: host array of :class:`KinematicSensorTransformCommand`
        :type host_array: List[KinematicSensorTransformCommand]
        :return: GPU array of :class:`KinematicSensorTransformCommand`
        :rtype: KinematicSensorTransformCommandGpuArray
        """
    def create_kinematic_sensor_velocity_command_gpu_array(self, host_array: list[KinematicSensorVelocityCommand]) -> KinematicSensorVelocityCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`KinematicSensorVelocityCommand` objects.
        
        :param host_array: host array of :class:`KinematicSensorVelocityCommand`
        :type host_array: List[KinematicSensorVelocityCommand]
        :return: GPU array of :class:`KinematicSensorVelocityCommand`
        :rtype: KinematicSensorVelocityCommandGpuArray
        """
    def create_light_transform_command_gpu_array(self, host_array: list[LightTransformCommand]) -> LightTransformCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`LightTransformCommand` objects.
        
        :param host_array: host array of :class:`LightTransformCommand`
        :type host_array: List[LightTransformCommand]
        :return: GPU array of :class:`LightTransformCommand`
        :rtype: LightTransformCommandGpuArray
        """
    def create_link_external_force_command_gpu_array(self, host_array: list[LinkExternalForceCommand]) -> LinkExternalForceCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`LinkExternalForceCommand` objects.
        
        :param host_array: host array of :class:`LinkExternalForceCommand`
        :type host_array: List[LinkExternalForceCommand]
        :return: GPU array of :class:`LinkExternalForceCommand`
        :rtype: LinkExternalForceCommandGpuArray
        """
    def create_link_property_command_gpu_array(self, host_array: list[LinkPropertyCommand]) -> LinkPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`LinkPropertyCommand` objects.
        
        :param host_array: host array of :class:`LinkPropertyCommand`
        :type host_array: List[LinkPropertyCommand]
        :return: GPU array of :class:`LinkPropertyCommand`
        :rtype: LinkPropertyCommandGpuArray
        """
    def create_link_transform_command_gpu_array(self, host_array: list[LinkTransformCommand]) -> LinkTransformCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`LinkTransformCommand` objects.
        
        :param host_array: host array of :class:`LinkTransformCommand`
        :type host_array: List[LinkTransformCommand]
        :return: GPU array of :class:`LinkTransformCommand`
        :rtype: LinkTransformCommandGpuArray
        """
    def create_link_velocity_command_gpu_array(self, host_array: list[LinkVelocityCommand]) -> LinkVelocityCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`LinkVelocityCommand` objects.
        
        :param host_array: host array of :class:`LinkVelocityCommand`
        :type host_array: List[LinkVelocityCommand]
        :return: GPU array of :class:`LinkVelocityCommand`
        :rtype: LinkVelocityCommandGpuArray
        """
    def create_mass_matrix_command_gpu_array(self, host_array: list[MassMatrixCommand]) -> MassMatrixCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`MassMatrixCommand` objects.
        
        :param host_array: host array of :class:`MassMatrixCommand`
        :type host_array: List[MassMatrixCommand]
        :return: GPU array of :class:`MassMatrixCommand`
        :rtype: MassMatrixCommandGpuArray
        """
    def create_motor_control_command_gpu_array(self, host_array: list[MotorControlCommand]) -> MotorControlCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`MotorControlCommand` objects.
        
        :param host_array: host array of :class:`MotorControlCommand`
        :type host_array: List[MotorControlCommand]
        :return: GPU array of :class:`MotorControlCommand`
        :rtype: MotorControlCommandGpuArray
        """
    def create_motor_property_command_gpu_array(self, host_array: list[MotorPropertyCommand]) -> MotorPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`MotorPropertyCommand` objects.
        
        :param host_array: host array of :class:`MotorPropertyCommand`
        :type host_array: List[MotorPropertyCommand]
        :return: GPU array of :class:`MotorPropertyCommand`
        :rtype: MotorPropertyCommandGpuArray
        """
    def create_pid_control_command_gpu_array(self, host_array: list[PIDControlCommand]) -> PIDControlCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`PIDControlCommand` objects.
        
        :param host_array: host array of :class:`PIDControlCommand`
        :type host_array: List[PIDControlCommand]
        :return: GPU array of :class:`PIDControlCommand`
        :rtype: PIDControlCommandGpuArray
        """
    def create_pid_property_command_gpu_array(self, host_array: list[PIDPropertyCommand]) -> PIDPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`PIDPropertyCommand` objects.
        
        :param host_array: host array of :class:`PIDPropertyCommand`
        :type host_array: List[PIDPropertyCommand]
        :return: GPU array of :class:`PIDPropertyCommand`
        :rtype: PIDPropertyCommandGpuArray
        """
    def create_raycast_command_gpu_array(self, host_array: list[RaycastCommand]) -> RaycastCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RaycastCommand` objects.
        
        :param host_array: host array of :class:`RaycastCommand`
        :type host_array: List[RaycastCommand]
        :return: GPU array of :class:`RaycastCommand`
        :rtype: RaycastCommandGpuArray
        """
    def create_rgb_camera_command_gpu_array(self, host_array: list[RGBCameraCommand]) -> RGBCameraCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RGBCameraCommand` objects.
        
        :param host_array: host array of :class:`RGBCameraCommand`
        :type host_array: List[RGBCameraCommand]
        :return: GPU array of :class:`RGBCameraCommand`
        :rtype: RGBCameraCommandGpuArray
        """
    def create_rgb_camera_skybox_command_gpu_array(self, host_array: list[RGBCameraSkyboxCommand]) -> RGBCameraSkyboxCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RGBCameraTransformCommand` objects.
        
        :param host_array: host array of :class:`RGBCameraTransformCommand`
        :type host_array: List[RGBCameraTransformCommand]
        :return: GPU array of :class:`RGBCameraTransformCommand`
        :rtype: RGBCameraTransformCommandGpuArray
        """
    def create_rgb_camera_transform_command_gpu_array(self, host_array: list[RGBCameraTransformCommand]) -> RGBCameraTransformCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RGBCameraTransformCommand` objects.
        
        :param host_array: host array of :class:`RGBCameraTransformCommand`
        :type host_array: List[RGBCameraTransformCommand]
        :return: GPU array of :class:`RGBCameraTransformCommand`
        :rtype: RGBCameraTransformCommandGpuArray
        """
    def create_rgb_material_property_command_gpu_array(self, host_array: list[RGBMaterialPropertyCommand]) -> RGBMaterialPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RGBMaterialPropertyCommand` objects.
        
        :param host_array: host array of :class:`RGBMaterialPropertyCommand`
        :type host_array: List[RGBMaterialPropertyCommand]
        :return: GPU array of :class:`RGBMaterialPropertyCommand`
        :rtype: RGBMaterialPropertyCommandGpuArray
        """
    def create_rigid_body_def_command_gpu_array(self, host_array: list[RigidBodyDefCommand]) -> RigidBodyDefCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RigidBodyDefCommand` objects.
        
        :param host_array: host array of :class:`RigidBodyDefCommand`
        :type host_array: List[RigidBodyDefCommand]
        :return: GPU array of :class:`RigidBodyDefCommand`
        :rtype: RigidBodyDefCommandGpuArray
        """
    def create_rigid_body_external_force_command_gpu_array(self, host_array: list[RigidBodyExternalForceCommand]) -> RigidBodyExternalForceCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RigidBodyExternalForceCommand` objects.
        
        :param host_array: host array of :class:`RigidBodyExternalForceCommand`
        :type host_array: List[RigidBodyExternalForceCommand]
        :return: GPU array of :class:`RigidBodyExternalForceCommand`
        :rtype: RigidBodyExternalForceCommandGpuArray
        """
    def create_rigid_body_kinematic_state_command_gpu_array(self, host_array: list[RigidBodyKinematicStateCommand], masks_buffer: BoolGpuBufferWrapper = None) -> RigidBodyKinematicStateCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RigidBodyKinematicStateCommand` objects.
        
        :param host_array: host array of :class:`RigidBodyKinematicStateCommand`
        :type host_array: List[RigidBodyKinematicStateCommand]
        :return: GPU array of :class:`RigidBodyKinematicStateCommand`
        :rtype: RigidBodyKinematicStateCommandGpuArray
        """
    def create_rigid_body_property_command_gpu_array(self, host_array: list[RigidBodyPropertyCommand]) -> RigidBodyPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RigidBodyPropertyCommand` objects.
        
        :param host_array: host array of :class:`RigidBodyPropertyCommand`
        :type host_array: List[RigidBodyPropertyCommand]
        :return: GPU array of :class:`RigidBodyPropertyCommand`
        :rtype: RigidBodyPropertyCommandGpuArray
        """
    def create_rigid_body_set(self, max_num_rigid_body_defs: int, max_num_rigid_bodies: int) -> VsRigidBodySetHandle:
        """
        Creates a rigid body set with a maximum definition and instance capacity.
        """
    def create_rigid_body_transform_command_gpu_array(self, host_array: list[RigidBodyTransformCommand], masks_buffer: BoolGpuBufferWrapper = None) -> RigidBodyTransformCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RigidBodyTransformCommand` objects.
        
        :param host_array: host array of :class:`RigidBodyTransformCommand`
        :type host_array: List[RigidBodyTransformCommand]
        :return: GPU array of :class:`RigidBodyTransformCommand`
        :rtype: RigidBodyTransformCommandGpuArray
        """
    def create_rigid_body_velocity_command_gpu_array(self, host_array: list[RigidBodyVelocityCommand], masks_buffer: BoolGpuBufferWrapper = None) -> RigidBodyVelocityCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RigidBodyVelocityCommand` objects.
        
        :param host_array: host array of :class:`RigidBodyVelocityCommand`
        :type host_array: List[RigidBodyVelocityCommand]
        :return: GPU array of :class:`RigidBodyVelocityCommand`
        :rtype: RigidBodyVelocityCommandGpuArray
        """
    def create_rigid_distance_joint_property_command_gpu_array(self, host_array: list[RigidDistanceJointPropertyCommand]) -> RigidDistanceJointPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RigidDistanceJointPropertyCommand` objects.
        
        :param host_array: host array of :class:`RigidDistanceJointPropertyCommand`
        :type host_array: List[RigidDistanceJointPropertyCommand]
        :return: GPU array of :class:`RigidDistanceJointPropertyCommand`
        :rtype: RigidDistanceJointPropertyCommandGpuArray
        """
    def create_rigid_material(self, rigid_material: RigidMaterial) -> VsMaterialHandle:
        """
        Creates a rigid material with the provided parameters and returns its handle.
        """
    def create_rigid_material_property_command_gpu_array(self, host_array: list[RigidMaterialPropertyCommand]) -> RigidMaterialPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`RigidMaterialPropertyCommand` objects.
        
        :param host_array: host array of :class:`RigidMaterialPropertyCommand`
        :type host_array: List[RigidMaterialPropertyCommand]
        :return: GPU array of :class:`RigidMaterialPropertyCommand`
        :rtype: RigidMaterialPropertyCommandGpuArray
        """
    def create_segmented_depth_camera_command_gpu_array(self, host_array: list[SegmentedDepthCameraCommand]) -> SegmentedDepthCameraCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`DepthCameraCommand` objects.
        
        :param host_array: host array of :class:`DepthCameraCommand`
        :type host_array: List[DepthCameraCommand]
        :return: GPU array of :class:`DepthCameraCommand`
        :rtype: DepthCameraCommandGpuArray
        """
    def create_segmented_rgb_camera_command_gpu_array(self, host_array: list[SegmentedRGBCameraCommand]) -> SegmentedRGBCameraCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`SegmentedRGBCameraCommand` objects.
        
        :param host_array: host array of :class:`SegmentedRGBCameraCommand`
        :type host_array: List[SegmentedRGBCameraCommand]
        :return: GPU array of :class:`SegmentedRGBCameraCommand`
        :rtype: SegmentedRGBCameraCommandGpuArray
        """
    def create_spatial_tendon_control_command_gpu_array(self, host_array: list[SpatialTendonControlCommand]) -> SpatialTendonControlCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`SpatialTendonControlCommand` objects.
        
        :param host_array: host array of :class:`SpatialTendonControlCommand`
        :type host_array: List[SpatialTendonControlCommand]
        :return: GPU array of :class:`SpatialTendonControlCommand`
        :rtype: SpatialTendonControlCommandGpuArray
        """
    def create_spatial_tendon_property_command_gpu_array(self, host_array: list[SpatialTendonPropertyCommand]) -> SpatialTendonPropertyCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`SpatialTendonPropertyCommand` objects.
        
        :param host_array: host array of :class:`SpatialTendonPropertyCommand`
        :type host_array: List[SpatialTendonPropertyCommand]
        :return: GPU array of :class:`SpatialTendonPropertyCommand`
        :rtype: SpatialTendonPropertyCommandGpuArray
        """
    def create_spatial_tendon_state_command_gpu_array(self, host_array: list[SpatialTendonStateCommand]) -> SpatialTendonStateCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`SpatialTendonStateCommand` objects.
        
        :param host_array: host array of :class:`SpatialTendonStateCommand`
        :type host_array: List[SpatialTendonStateCommand]
        :return: GPU array of :class:`SpatialTendonStateCommand`
        :rtype: SpatialTendonStateCommandGpuArray
        """
    def create_spline(self, control_points: list[Vec3]) -> None:
        """
        Create spline.
        
        Argument: list of :class:`Vec3` control points.
        """
    def create_transform_command_gpu_array(self, host_array: list[TransformCommand]) -> TransformCommandGpuArray:
        """
        Creates a GPU array from a host array of :class:`TransformCommand` objects.
        
        :param host_array: host array of :class:`TransformCommand`
        :type host_array: List[TransformCommand]
        :return: GPU array of :class:`TransformCommand`
        :rtype: TransformCommandGpuArray
        """
    def create_triangle_mesh(self, filename: str, scale: Vec3, rigid_set_handle: VsRigidBodySetHandle, local_transform: Transform, transform: Transform, material_handle: VsMaterialHandle) -> VsRigidBodyHandle:
        ...
    @typing.overload
    def create_uint16_gpu_array(self, host_array: list[int]) -> Uint16GpuArray:
        """
        Creates a GPU array from a host array of :class:`VsU16` objects.
        
        :param host_array: host array of :class:`VsU16`
        :type host_array: List[VsU16]
        :return: GPU array of :class:`VsU16`
        :rtype: VsU16GpuArray
        """
    @typing.overload
    def create_uint16_gpu_array(self, num_elements: int) -> Uint16GpuArray:
        """
        Creates a GPU array of size ``num_elements`` of :class:`VsU16` elements.
        """
    @typing.overload
    def create_uint32_gpu_array(self, host_array: list[int]) -> Uint32GpuArray:
        """
        Creates a GPU array from a host array of :class:`VsU32` objects.
        
        :param host_array: host array of :class:`VsU32`
        :type host_array: List[VsU32]
        :return: GPU array of :class:`VsU32`
        :rtype: VsU32GpuArray
        """
    @typing.overload
    def create_uint32_gpu_array(self, num_elements: int) -> Uint32GpuArray:
        """
        Creates a GPU array of size ``num_elements`` of :class:`VsU32` elements.
        """
    @typing.overload
    def create_uint8_gpu_array(self, host_array: list[int]) -> Uint8GpuArray:
        """
        Creates a GPU array from a host array of :class:`VsU8` objects.
        
        :param host_array: host array of :class:`VsU8`
        :type host_array: List[VsU8]
        :return: GPU array of :class:`VsU8`
        :rtype: VsU8GpuArray
        """
    @typing.overload
    def create_uint8_gpu_array(self, num_elements: int) -> Uint8GpuArray:
        """
        Creates a GPU array of size ``num_elements`` of :class:`VsU8` elements.
        """
    def end_recording(self, filename: str, mask: list[bool] = [], max_file_size: int = 1073741824) -> None:
        """
        Stop recording and save data to file.
        
        :param filename: output file
        :type filename: str
        :param mask: save only the environments where ``mask`` evaluates to true.  Save all environments if
            empty list (default empty list)
        :type mask: list
        :param max_file_size: maximum output file size in bytes (default 1GiB)
        :type max_file_size: int
        """
    def end_streaming(self) -> None:
        """
        Stops streaming kinematic state of an environment
        """
    def finalize_filtered_force_sensors(self) -> None:
        """
        Computes the filtered force sensors.
        
        This requires an additional processing pass over the collision data, so it is not automatically
        performed as part of the step unless explicitly requested.
        
        Hence, this function must be called **after** :meth:`Gym.step()`.
        
        Note: if you want to set new contact filters, you must call :meth:`Gym.set_contact_filters()`
        **before** :meth:`Gym.step()` in order for the updated contact filters to take effect.
        
        The complete order of operations for getting force sensor values after setting contact filters
        is as follows:
        
        1. :meth:`Gym.set_contact_filters()`
        2. :meth:`Gym.step()`
        3. :meth:`Gym.finalize_filtered_force_sensors()`
        4. :meth:`Gym.get_sensor_forces()`
        """
    def find_closest_points(self, p_buffer: Float32GpuBufferWrapper, closest_p_buffer: Float32GpuBufferWrapper, gradient_buffer: Float32GpuBufferWrapper, num_queries: int, max_iterations: int = 8) -> None:
        """
        Find closest points to array of points of size ``num_queries``
        
        :param p_buffer: GPU buffer containing :class:`Vec3` query points (input)
        :type p_buffer: Float32GpuBufferWrapper
        
        :param closest_p_buffer: GPU buffer containing :class:`Vec3` closest points (output)
        :type closest_p_buffer: Float32GpuBufferWrapper
        
        :param gradient_buffer: GPU buffer containing :class:`Vec3` gradients (output)
        :type gradient_buffer: Float32GpuBufferWrapper
        
        :param num_queries: number of queries
        :type num_queries: int
        
        :param max_iterations: maximum number of iterations (default 8)
        :type max_iterations: int
        """
    def find_spline_points_and_gradients(self, u_buffer: Float32GpuBufferWrapper, point_buffer: Float32GpuBufferWrapper, gradient_buffer: Float32GpuBufferWrapper, num_queries: int) -> None:
        """
        Find points and gradients for array of coordinates ``u`` of size ``num_queries``
        
        :param u_buffer: input buffer containing ``num_queries`` coordinates.  Array size is
            ``num_queries``.
        :type u_buffer: Float32GpuBufferWrapper
        :param point_buffer: output buffer containing ``num_queries`` points as :class:`Vec3` objects.
            Array size is ``num_queries * 3``.
        :type point_buffer: Float32GpuBufferWrapper
        :param gradient_buffer: output buffer containing ``num_queries`` gradients as :class:`Vec3` objects.
            Array size is ``num_queries * 3``.
        :type gradient_buffer: Float32GpuBufferWrapper
        :param num_queries: number of queries
        :type num_queries: int
        """
    def get_articulation_kinematic_states(self, array: ArticulationKinematicStateCommandGpuArray) -> None:
        """
        Executes an array of commands that write kinematic states from the simulation into the supplied buffers.
        
        See :class:`ArticulationKinematicStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`ArticulationKinematicStateCommand` commands
        :type array: ArticulationKinematicStateCommandGpuArray
        """
    def get_deformable_material_properties(self, array: DeformableMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_deformable_transforms(self, array: DeformableTransformCommandGpuArray) -> None:
        """
        Executes an array of commands that write deformable transforms from the simulation into the supplied buffers.
        
        See :class:`DeformableTransformCommand` for more information on this process.
        
        :param array: GPU array of :class:`DeformableTransformCommand` commands
        :type array: DeformableTransformCommandGpuArray
        """
    def get_depth_camera_images(self, array: DepthCameraCommandGpuArray) -> None:
        """
        Get depth camera images
        """
    def get_depth_camera_images_in_rgb(self, array: DepthCameraCommandGpuArray) -> None:
        """
        Get depth camera images
        """
    def get_depth_camera_transforms(self, array: DepthCameraTransformCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_elastic_material_properties(self, array: ElasticMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_environment(self, environment_handle: EnvironmentHandle) -> Environment:
        """
        Gets the :class:`Environment` with the provided ``environment_handle``.
        """
    def get_environment_def(self, environment_def_handle: EnvironmentDefHandle) -> EnvironmentDef:
        """
        Get environment definition object.
        
        :param environment_def_handle: environment definition handle
        :type environment_def_handle: EnvironmentDefHandle
        :return: environment definition object
        :rtype: EnvironmentDef
        """
    def get_environment_def_handle(self, environment_def_index: int) -> EnvironmentDefHandle:
        """
        Gets the handle of the :class:`EnvironmentDef` at index ``environment_def_index``.
        """
    def get_environment_def_handles(self) -> list[EnvironmentDefHandle]:
        """
        Forms and returns an array of all current :class:`EnvironmentDefHandle` handles.
        """
    @typing.overload
    def get_environment_group(self, environment_group_handle: EnvironmentGroupHandle) -> EnvironmentGroup:
        """
        Gets the :class:`EnvironmentGroup` with the provided ``environment_group_handle``.
        """
    @typing.overload
    def get_environment_group(self, environment_set_handle: EnvironmentSetHandle) -> EnvironmentGroup:
        """
        Gets the :class:`EnvironmentGroup` with the provided ``environment_set_handle``.
        """
    @typing.overload
    def get_environment_group(self, environment_handle: EnvironmentHandle) -> EnvironmentGroup:
        """
        Gets the :class:`EnvironmentGroup` with the provided ``environment_handle``.
        """
    def get_environment_group_handle(self, environment_group_index: int) -> EnvironmentGroupHandle:
        """
        Gets the handle of the :class:`EnvironmentGroup` at index ``environment_group_index``.
        """
    def get_environment_handle_for_vs_transform_handle(self, arg0: VsTransformHandle) -> EnvironmentHandle:
        ...
    @typing.overload
    def get_environment_set(self, environment_set_handle: EnvironmentSetHandle) -> EnvironmentSet:
        """
        Gets the :class:`EnvironmentSet` with the provided ``environment_set_handle``.
        """
    @typing.overload
    def get_environment_set(self, environment_handle: EnvironmentHandle) -> EnvironmentSet:
        """
        Gets the :class:`EnvironmentSet` with the provided ``environment_handle``.
        """
    def get_environment_transform(self, environment_handle: EnvironmentHandle) -> Transform:
        """
        Get the transform offset of the environment referenced by `environment_handle`.
        
        :param environment_handle: environment handle
        :type environment_handle: EnvironmentHandle
        :returns: transform offset
        :rtype: Transform
        """
    def get_gravity(self) -> Vec3:
        ...
    def get_hill_material_properties(self, array: HillMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_joint_active_masks(self, array: JointActiveMaskCommandGpuArray) -> None:
        """
        Executes an array of commands that read joint active masks from the supplied buffers and write them to the simulation.
        
        See :class:`JointActiveMaskCommand` for more information on this process.
        
        :param array: GPU array of :class:`JointActiveMaskCommand` commands
        :type array: JointActiveMaskCommandGpuArray
        """
    def get_joint_dof_properties(self, array: JointDofPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_joint_forces(self, array: JointStateCommandGpuArray) -> None:
        """
        Executes an array of commands that read joint forces from the supplied buffers and write them to the simulation.
        
        See :class:`JointStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`JointStateCommand` commands
        :type array: JointStateCommandGpuArray
        """
    def get_joint_positions(self, array: JointStateCommandGpuArray) -> None:
        """
        Executes an array of commands that write joint positions from the simulation into the supplied buffers.
        
        See :class:`JointStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`JointStateCommand` commands
        :type array: JointStateCommandGpuArray
        """
    def get_joint_properties(self, array: JointPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_joint_sensor_forces(self, array: JointForceSensorCommandGpuArray) -> None:
        """
        Executes an array of commands that write joint sensor forces from the simulation into the supplied buffers.
        
        See :class:`JointForceSensorCommand` for more information on this process.
        
        :param array: GPU array of :class:`JointForceSensorCommand` commands
        :type array: JointForceSensorCommandGpuArray
        """
    def get_joint_velocities(self, array: JointStateCommandGpuArray) -> None:
        """
        Executes an array of commands that write joint velocities from the simulation into the supplied buffers.
        
        See :class:`JointStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`JointStateCommand` commands
        :type array: JointStateCommandGpuArray
        """
    def get_kinematic_sensor_states(self, array: KinematicSensorStateCommandGpuArray) -> None:
        """
        Executes an array of commands that write kinematic sensor states from the simulation into the supplied buffers.
        
        See :class:`KinematicSensorStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`KinematicSensorStateCommand` commands
        :type array: KinematicSensorStateCommandGpuArray
        """
    def get_kinematic_sensor_transforms(self, array: KinematicSensorTransformCommandGpuArray) -> None:
        """
        Executes an array of commands that write kinematic sensor transforms from the simulation into the supplied buffers.
        
        See :class:`KinematicSensorTransformCommand` for more information on this process.
        
        :param array: GPU array of :class:`KinematicSensorTransformCommand` commands
        :type array: KinematicSensorTransformCommandGpuArray
        """
    def get_kinematic_sensor_velocities(self, array: KinematicSensorVelocityCommandGpuArray) -> None:
        """
        Executes an array of commands that write kinematic sensor velocities from the simulation into the supplied buffers.
        
        See :class:`KinematicSensorVelocityCommand` for more information on this process.
        
        :param array: GPU array of :class:`KinematicSensorVelocityCommand` commands
        :type array: KinematicSensorVelocityCommandGpuArray
        """
    def get_light_transforms(self, array: LightTransformCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_link_properties(self, array: LinkPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_link_transforms(self, array: LinkTransformCommandGpuArray) -> None:
        """
        Executes an array of commands that write link transforms from the simulation into the supplied buffers.
        
        See :class:`LinkTransformCommand` for more information on this process.
        
        :param array: GPU array of :class:`LinkTransformCommand` commands
        :type array: LinkTransformCommandGpuArray
        """
    def get_link_velocities(self, array: LinkVelocityCommandGpuArray) -> None:
        """
        Executes an array of commands that write link velocities from the simulation into the supplied buffers.
        
        See :class:`LinkVelocityCommand` for more information on this process.
        
        :param array: GPU array of :class:`LinkVelocityCommand` commands
        :type array: LinkVelocityCommandGpuArray
        """
    def get_motor_properties(self, array: MotorPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_num_control_points(self) -> int:
        """
        Get number of control points of spline
        
        Return: number of control points.
        """
    def get_num_environment_defs(self) -> int:
        """
        Get the number of environment defs.
        """
    def get_num_environment_groups(self) -> int:
        """
        Gets the number of environment groups.
        """
    def get_num_solver_iterations(self) -> int:
        """
        Gets the number of solver iterations to perform.
        """
    def get_pid_properties(self, array: PIDPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_query_geometry_handle(self, rigid_body_handle: VsRigidBodyHandle, rigid_body_set_handle: VsRigidBodySetHandle) -> VsQueryGeometryHandle:
        """
        Get :class:`VsQueryGeometryHandle` for :class:`VsRigidBodyHandle` and
        :class:`VsRigidBodySetHandle`.
        
        Returns invalid handle if no handle was found.
        """
    def get_raycasts(self, array: RaycastCommandGpuArray) -> None:
        """
        Perform raycast commands specified in ``array``.
        
        See :class:`RaycastCommand` for more information.
        """
    def get_render(self) -> GymRender:
        """
        :return: :class:`GymRender` if rendering is enabled, else None
        :rtype: GymRender | None
        """
    def get_rgb_camera_images(self, array: RGBCameraCommandGpuArray) -> None:
        """
        Get RGB camera images
        """
    def get_rgb_camera_transforms(self, array: RGBCameraTransformCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_rgb_material_properties(self, array: RGBMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_rigid_body_kinematic_states(self, array: RigidBodyKinematicStateCommandGpuArray) -> None:
        """
        Executes an array of commands that write rigid body velocities from the simulation into the supplied buffers.
        
        See :class:`RigidBodyKinematicStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`RigidBodyKinematicStateCommand` commands
        :type array: RigidBodyKinematicStateCommandGpuArray
        """
    def get_rigid_body_properties(self, array: RigidBodyPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_rigid_body_transforms(self, array: RigidBodyTransformCommandGpuArray) -> None:
        """
        Executes an array of commands that write rigid body transforms from the simulation into the supplied buffers.
        
        See :class:`RigidBodyTransformCommand` for more information on this process.
        
        :param array: GPU array of :class:`RigidBodyTransformCommand` commands
        :type array: RigidBodyTransformCommandGpuArray
        """
    def get_rigid_body_velocities(self, array: RigidBodyVelocityCommandGpuArray) -> None:
        """
        Executes an array of commands that write rigid body velocities from the simulation into the supplied buffers.
        
        See :class:`RigidBodyVelocityCommand` for more information on this process.
        
        :param array: GPU array of :class:`RigidBodyVelocityCommand` commands
        :type array: RigidBodyVelocityCommandGpuArray
        """
    def get_rigid_contacts(self, normals_buffer: Float32GpuBufferWrapper, point_seps_buffer: Float32GpuBufferWrapper, id_a_buffer: Uint32GpuBufferWrapper, id_b_buffer: Uint32GpuBufferWrapper, num_output: int) -> int:
        """
        Gets info on all rigid contacts.
        
        This function will write up to ``num_output`` rigid contacts into the provided buffers.  The total
        number of rigid contacts is returned as an integer.
        
        If the total number of rigid contacts, denoted by ``num_contacts``, is less
        than or equal to``num_output``, then only the first ``num_contacts`` entries in
        the buffers are valid.
        
        If ``num_contacts`` is greater than ``num_output``, then only the first ``num_output`` rigid
        contacts are recorded in the buffers.
        
        :param normals_buffer: buffer for reading contact normals, formatted as as an array of (x, y, z).
            Buffer size must be equal to ``num_output * 3``.
        :type normals_buffer: Float32GpuBufferWrapper
        
        :param point_seps_buffer: buffer for reading point separations, formatted as an array of (x, y, z,
            w) where (x, y, z) is the position of the contact, and w is the separation value. Buffer size
            must be equal to ``num_output * 4``.
        :type point_seps_buffer: Float32GpuBufferWrapper
        
        :param id_a_buffer: buffer for reading the ID of rigid body A, which is composed of its
            :class:`EnvironmentHandle` and its :class:`TransformHandle`.  It is formatted as an array of
            four integers: (group_index, set_index, env_index, transform_index), where the first three
            integers correspond to the index values of :class:`EnvironmentHandle`, and the fourth value to
            the index value of :class:`TransformHandle`.  Buffer size must be equal to
            ``num_output * 4``.
        :type id_a_buffer: Uint32GpuBufferWrapper
        
        :param id_b_buffer: same as ``id_a_buffer``, but for body B.
        :type id_b_buffer: Uint32GpuBufferWrapper
        
        :param num_output: maximum number of contacts to write to buffers.
        :type num_output: int
        
        :return: total number of rigid contacts
        :rtype: int
        """
    def get_rigid_material_properties(self, array: RigidMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_segmented_depth_camera_images(self, array: SegmentedDepthCameraCommandGpuArray) -> None:
        """
        Get segmented depth camera images
        """
    def get_segmented_depth_camera_images_in_rgb(self, array: SegmentedDepthCameraCommandGpuArray) -> None:
        """
        Get segmented depth camera images in RGB
        """
    def get_segmented_rgb_camera_images(self, array: SegmentedRGBCameraCommandGpuArray) -> None:
        """
        Get segmented RGB camera images
        """
    def get_segmented_rgb_camera_images_in_rgb(self, array: SegmentedRGBCameraCommandGpuArray) -> None:
        """
        Get segmented RGB camera images in RGB
        """
    def get_sensor_forces(self, array: ForceSensorCommandGpuArray) -> None:
        """
        Executes an array of commands that write sensor forces from the simulation into the supplied buffers.
        
        See :class:`ForceSensorCommand` for more information on this process.
        
        :param array: GPU array of :class:`ForceSensorCommand` commands
        :type array: ForceSensorCommandGpuArray
        
        Note that if you are setting contact filters on force sensors, you must call
        :meth:`Gym.set_contact_filters()` **before** :meth:`Gym.step()` and
        :meth:`Gym.finalize_filtered_force_sensors` **after** :meth:`Gym.step()` to get the correct filtered
        force sensor values.
        
        The complete order of operations for getting force sensor values after setting contact filters
        is as follows:
        
        1. :meth:`Gym.set_contact_filters()`
        2. :meth:`Gym.step()`
        3. :meth:`Gym.finalize_filtered_force_sensors()`
        4. :meth:`Gym.get_sensor_forces()`
        """
    def get_spatial_tendon_properties(self, array: SpatialTendonPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def get_spatial_tendon_states(self, array: SpatialTendonStateCommandGpuArray) -> None:
        """
        Executes an array of commands that read spatial tendon force from the supplied buffers and write them to the simulation.
        
        See :class:`SpatialTendonControlCommand` for more information on this process.
        
        :param array: GPU array of :class:`SpatialTendonControlCommand` commands
        :type array: SpatialTendonControlCommandGpuArray
        """
    def get_spline_gradient(self, u: float) -> Vec3:
        """
        Get spline gradient at coordinate ``u``
        
        Argument: coordinate ``u`` in interval ``[0, num_control_points - 1]``
        
        Return: interpolated spline gradient at coordinate ``u``.
        """
    def get_spline_point(self, u: float) -> Vec3:
        """
        Get spline point at coordinate ``u``
        
        Argument: coordinate ``u`` in interval ``[0, num_control_points - 1]``
        
        Return: interpolated spline point at coordinate ``u``.
        """
    def get_timestep(self) -> float:
        """
        Get the simulation timestep.
        
        :returns: simulaton timestep
        :rtype: VsReal
        """
    def get_transform_handle_for_vs_transform_handle(self, arg0: VsTransformHandle) -> TransformHandle:
        ...
    def get_transforms(self, array: TransformCommandGpuArray) -> None:
        """
        Get transforms
        
        Note: you must call :meth:`Gym.compute_kinematics()` after calling :meth:`Gym.step()` and before
        calling this function in order to get up-to-date transforms.
        """
    def get_up_axis(self) -> Vec3:
        """
        :return: the up axis
        :rtype: Vec3
        """
    def get_vs_transform_handle_of_articulation_link(self, arg0: EnvironmentHandle, arg1: ArticulationHandle, arg2: int) -> VsTransformHandle:
        ...
    def get_vs_transform_handle_of_rigid_body(self, arg0: EnvironmentHandle, arg1: RigidBodyHandle) -> VsTransformHandle:
        ...
    def get_working_directory(self) -> str:
        """
        Get the current working directory. Used for finding assets and other files
        """
    def gym_finalize(self) -> None:
        """
        Finalizes the whole Gym and the whole underlying Vsim state. This syncs all the data to the
        GPU and includes all other finalize functions.
        
        If render is enabled, :meth:`Gym.compute_kinematics()` is also called.
        """
    def import_environment(self, filename: str) -> EnvironmentDefHandle:
        """
        Creates and environment definition from imported and environment definition file (*.vscn).
        If you're importing multiple environment definitions for heterogeneous training, simulations settings defined in those files must match exactly.
        """
    def rigid_body_component_finalize(self) -> None:
        """
        Finalizes all rigid bodies, uploading their data to the GPU.
        """
    def scene_finalize(self) -> None:
        """
        Finalizes the Vsim scene. This uploads all the scene specific data to the GPU. This is a super set of :meth:`articulation_component_finalize` and :meth:`rigid_body_component_finalize`.
        """
    def set_articulation_kinematic_states(self, array: ArticulationKinematicStateCommandGpuArray) -> None:
        """
        Executes an array of commands that read kinematic states from the supplied buffers and write them to the simulation.
        
        See :class:`ArticulationKinematicStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`ArticulationKinematicStateCommand` commands
        :type array: ArticulationKinematicStateCommandGpuArray
        """
    def set_cameras_not_rendered(self) -> None:
        """
        setCamerasNotRendered.
        """
    def set_contact_filters(self, array: ContactFilterCommandGpuArray) -> None:
        """
        Set contact filters on force sensor.
        
        Contact filters are persistent; when you set contact filters, they will remain in place until the
        next time that you set them.
        
        Note that the new contact filters do not take effect until **after** the next call to
        :meth:`Gym.step()`.
        
        Furthermore, you must call :meth:`Gym.finalize_filtered_force_sensors` **after** :meth:`Gym.step()`
        everytime you want to get filtered force sensor values.
        
        The complete order of operations for getting force sensor values after setting contact filters
        is as follows:
        
        1. :meth:`Gym.set_contact_filters()`
        2. :meth:`Gym.step()`
        3. :meth:`Gym.finalize_filtered_force_sensors()`
        4. :meth:`Gym.get_sensor_forces()`
        """
    def set_deformable_material_properties(self, array: DeformableMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_deformable_transforms(self, array: DeformableTransformCommandGpuArray) -> None:
        """
        Executes an array of commands that read deformable transforms from the supplied buffers and write them to the simulation.
        
        See :class:`DeformableTransformCommand` for more information on this process.
        
        :param array: GPU array of :class:`DeformableTransformCommand` commands
        :type array: DeformableTransformCommandGpuArray
        """
    def set_depth_camera_transforms(self, array: DepthCameraTransformCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_elastic_material_properties(self, array: ElasticMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_environment_transform(self, environment_handle: EnvironmentHandle, transform: Transform) -> None:
        """
        Set the transform offset for the environment referenced by `environment_handle`.
        
        Note that the environment transform is normalized before it is applied to the environment.
        
        Note that :meth:`Gym.gym_finalize()` must be called after this function for the environment
        transform to take effect.
        
        :param environment_handle: environment handle
        :type environment_handle: EnvironmentHandle
        :param transform: transform offset
        :type transform: Transform
        """
    def set_fixed_tendon_controls(self, array: FixedTendonControlCommandGpuArray) -> None:
        """
        Executes an array of commands that read fixed tendon offset from the supplied buffers and write them to the simulation.
        
        See :class:`FixedTendonControlCommand` for more information on this process.
        
        :param array: GPU array of :class:`FixedTendonControlCommand` commands
        :type array: FixedTendonControlCommandGpuArray
        """
    def set_fixed_tendon_forces(self, array: FixedTendonControlCommandGpuArray) -> None:
        """
        Executes an array of commands that read fixed tendon force from the supplied buffers and write them to the simulation.
        
        See :class:`FixedTendonControlCommand` for more information on this process.
        
        :param array: GPU array of :class:`FixedTendonControlCommand` commands
        :type array: FixedTendonControlCommandGpuArray
        """
    def set_fixed_tendon_properties(self, array: FixedTendonPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_gravity(self, gravity: Vec3) -> None:
        """
        Sets the gravity value of the scene.
        """
    def set_hill_material_properties(self, array: HillMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_joint_active_masks(self, array: JointActiveMaskCommandGpuArray) -> None:
        """
        Executes an array of commands that read joint active masks from the supplied buffers and write them to the simulation.
        
        See :class:`JointActiveMaskCommand` for more information on this process.
        
        :param array: GPU array of :class:`JointActiveMaskCommand` commands
        :type array: JointActiveMaskCommandGpuArray
        """
    def set_joint_dof_properties(self, array: JointDofPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_joint_forces(self, array: JointStateCommandGpuArray) -> None:
        """
        Executes an array of commands that read joint forces from the supplied buffers and write them to the simulation.
        
        See :class:`JointStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`JointStateCommand` commands
        :type array: JointStateCommandGpuArray
        """
    def set_joint_positions(self, array: JointStateCommandGpuArray) -> None:
        """
        Executes an array of commands that read joint positions from the supplied buffers and write them to the simulation.
        
        See :class:`JointStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`JointStateCommand` commands
        :type array: JointStateCommandGpuArray
        """
    def set_joint_properties(self, array: JointPropertyCommandGpuArray) -> None:
        """
        NOTE: when changing the mass, we scale the diagonalised inertia by the same factor
        """
    def set_joint_target_positions(self, array: PIDControlCommandGpuArray) -> None:
        """
        Executes an array of commands that read joint target positions from the supplied buffers and write them to the simulation.
        
        See :class:`PIDControlCommand` for more information on this process.
        
        :param array: GPU array of :class:`PIDControlCommand` commands
        :type array: PIDControlCommandGpuArray
        """
    def set_joint_target_velocities(self, array: PIDControlCommandGpuArray) -> None:
        """
        Executes an array of commands that read joint target velocities from the supplied buffers and write them to the simulation.
        
        See :class:`PIDControlCommand` for more information on this process.
        
        :param array: GPU array of :class:`PIDControlCommand` commands
        :type array: PIDControlCommandGpuArray
        """
    def set_joint_velocities(self, array: JointStateCommandGpuArray) -> None:
        """
        Executes an array of commands that read joint velocities from the supplied buffers and write them to the simulation.
        
        See :class:`JointStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`JointStateCommand` commands
        :type array: JointStateCommandGpuArray
        """
    def set_light_transforms(self, array: LightTransformCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_link_external_forces(self, array: LinkExternalForceCommandGpuArray) -> None:
        """
        Executes an array of commands that read link external forces from the supplied buffers and write them to the simulation.
        
        See :class:`LinkExternalForceCommand` for more information on this process.
        
        :param array: GPU array of :class:`LinkExternalForceCommand` commands
        :type array: LinkExternalForceCommandGpuArray
        """
    def set_link_properties(self, array: LinkPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_link_transforms(self, array: LinkTransformCommandGpuArray) -> None:
        """
        Executes an array of commands that read link transforms from the supplied buffers and write them to the simulation.
        
        See :class:`LinkTransformCommand` for more information on this process.
        
        :param array: GPU array of :class:`LinkTransformCommand` commands
        :type array: LinkTransformCommandGpuArray
        """
    def set_link_velocities(self, array: LinkVelocityCommandGpuArray) -> None:
        """
        Executes an array of commands that read link velocities from the supplied buffers and write them to the simulation.
        
        See :class:`LinkVelocityCommand` for more information on this process.
        
        :param array: GPU array of :class:`LinkVelocityCommand` commands
        :type array: LinkVelocityCommandGpuArray
        """
    def set_motor_forces(self, array: MotorControlCommandGpuArray) -> None:
        """
        Executes an array of commands that read motor forces from the supplied buffers and write them to the simulation.
        
        See :class:`MotorControlCommand` for more information on this process.
        
        :param array: GPU array of :class:`MotorControlCommand` commands
        :type array: MotorControlCommandGpuArray
        """
    def set_motor_properties(self, array: MotorPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_num_solver_iterations(self, num_iterations: int) -> None:
        """
        Sets the number of solver iterations to perform.
        """
    def set_pid_properties(self, array: PIDPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_rgb_camera_skyboxes(self, array: RGBCameraSkyboxCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_rgb_camera_transforms(self, array: RGBCameraTransformCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_rgb_material_properties(self, array: RGBMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_rigid_body_defs(self, array: RigidBodyDefCommandGpuArray) -> None:
        """
        Executes an array of commands that read rigid body def indices from the supplied buffers and write them to the simulation.
        
        See :class:`RigidBodyDefCommand` for more information on this process.
        
        :param array: GPU array of :class:`RigidBodyDefCommand` commands
        :type array: RigidBodyDefCommandGpuArray
        """
    def set_rigid_body_external_forces(self, array: RigidBodyExternalForceCommandGpuArray) -> None:
        """
        Executes an array of commands that read rigid body external forces from the supplied buffers and write them to the simulation.
        
        See :class:`RigidBodyExternalForceCommand` for more information on this process.
        
        :param array: GPU array of :class:`RigidBodyExternalForceCommand` commands
        :type array: RigidBodyExternalForceCommandGpuArray
        """
    def set_rigid_body_kinematic_states(self, array: RigidBodyKinematicStateCommandGpuArray) -> None:
        """
        Executes an array of commands that read rigid body kinematic states from the supplied buffers and write them to the simulation.
        
        See :class:`RigidBodyKinematicStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`RigidBodyKinematicStateCommand` commands
        :type array: RigidBodyKinematicStateCommandGpuArray
        """
    def set_rigid_body_properties(self, array: RigidBodyPropertyCommandGpuArray) -> None:
        """
        NOTE: when changing the mass, we scale the diagonalised inertia by the same factor
        """
    def set_rigid_body_transforms(self, array: RigidBodyTransformCommandGpuArray) -> None:
        """
        Executes an array of commands that read rigid body transforms from the supplied buffers and write them to the simulation.
        
        See :class:`RigidBodyTransformCommand` for more information on this process.
        
        :param array: GPU array of :class:`RigidBodyTransformCommand` commands
        :type array: RigidBodyTransformCommandGpuArray
        """
    def set_rigid_body_velocities(self, array: RigidBodyVelocityCommandGpuArray) -> None:
        """
        Executes an array of commands that read rigid body velocities from the supplied buffers and write them to the simulation.
        
        See :class:`RigidBodyVelocityCommand` for more information on this process.
        
        :param array: GPU array of :class:`RigidBodyVelocityCommand` commands
        :type array: RigidBodyVelocityCommandGpuArray
        """
    def set_rigid_distance_joint_properties(self, array: RigidDistanceJointPropertyCommandGpuArray) -> None:
        """
        set rigid distance joint properties
        """
    def set_rigid_material_properties(self, array: RigidMaterialPropertyCommandGpuArray) -> None:
        """
        TODO: document
        """
    def set_spatial_tendon_controls(self, array: SpatialTendonControlCommandGpuArray) -> None:
        """
        Executes an array of commands that read spatial tendon offset from the supplied buffers and write them to the simulation.
        
        See :class:`SpatialTendonControlCommand` for more information on this process.
        
        :param array: GPU array of :class:`SpatialTendonControlCommand` commands
        :type array: SpatialTendonControlCommandGpuArray
        """
    def set_spatial_tendon_forces(self, array: SpatialTendonControlCommandGpuArray) -> None:
        """
        Executes an array of commands that read spatial tendon force from the supplied buffers and write them to the simulation.
        
        See :class:`SpatialTendonStateCommand` for more information on this process.
        
        :param array: GPU array of :class:`SpatialTendonStateCommand` commands
        :type array: SpatialTendonStateCommandGpuArray
        """
    def set_timestep(self, timestep: float) -> None:
        """
        Set the simulation timestep.
        
        To maintain simulation stability and determinism, set the simulation timestep only once. Changing
        the simulation timestep during simulation is currently unsupported.
        
        :param timestep: simulation timestep
        :type timestep: VsReal
        """
    def set_working_directory(self, path: str) -> None:
        """
        Set the current working directory.
        """
    def sim_finalize(self) -> None:
        """
        Finalizes the global Vsim state. That uploads the underlying mesh asset, physics material, geometry and shape data to the GPU.
        """
    def start_recording(self, env_group_handle: EnvironmentGroupHandle = ..., max_num_frames: int = 64) -> None:
        """
        Start recording kinematic states of all actors in the given environment group.
        
        Only articulations and rigid bodies are supported at the moment.
        
        Only the last ``max_num_frames`` frames are saved.
        
        Stop recording with :meth:`Gym.end_recording()`.
        
        :param env_group_handle: handle of environment group to record (default 0)
        :type env_group_handle: EnvironmentGroupHandle
        :param max_num_frames: number of frames in ring buffer (default 64)
        :type max_num_frames: int
        """
    def start_streaming(self, env_handle: EnvironmentHandle, topic: str, address: str = '*', port: int = 5051) -> None:
        """
        Start streaming kinematic states of all actors in the given environment to an endpoint.
        
        You can only start streaming after calling :meth:`Gym.gym_finalize()`.
        
        Only articulations and rigid bodies are supported at the moment.
        
        This call will automatically stop any any streaming that was previously started for this or another environment.
        To manually stop streaming call :meth:`Gym.end_streaming()`.
        
        :param env_handle: handle of environment to record
        :type env_handle: EnvironmentHandle
        :param topic: topic name that should be used for streaming
        :type topic: string
        :param address: IP address that should be used for streaming. Defaults to "*", meaning that Vlearn listens for subscribers.
        :type address: string
        :param port: port that should be used for streaming. Defaults to 5050.
        :type port: int
        """
    def sync_device(self) -> None:
        """
        Synchronizes all outstanding work scheduled on the GPU.
        """
    def update_scene_dependent_components(self, force_update: bool = False) -> None:
        """
        updateSceneDependentComponents.
        """
class GymRender:
    """
    Class responsible for rendering the Gym.
    
    Get :class:`GymRender` from :class:`Gym` using :meth:`Gym.get_render()`.
    
    Use ``WASD`` keys to move the camera.
    
    Use the left mouse button to change the camera orientation.
    
    If scene querying was enabled in :meth:`vlearn.create_gym()`, you can use the right mouse button to
    pick objects.
    
    The default GUI menu has the following items:
    
    - Enable Render: enables rendering.
    - Paused: pause the simulation. Pressing ``P`` also toggles this box.
    - Step: step the simulation by one frame while paused. Pressing ``O`` also toggles this box.
    - Capped Step: synchronize the rendering rate with the simulation time step
    - Debug Collision Penetrations: render collisions.
    - Debug Collision Normals: render collision normals.
    - Show Origin: render origin.
    - Line Drawing: render meshes using lines.
    - Debug Inertias: render inertial frames.
    - Debug Joints: render joint positions and limits.
    - Debug Bounds: render bounding volumes of rigid bodies.
    - Debug Frames: render rigid body frame origins.
    - Spatial Tendons: render spatial tendons.
    - Debug Force Sensors: render force sensors.
    - Debug Camera Rays: render depth and RGB camera ray hits.
    - Debug Camera Frustrum: render depth and RGB camera frustrum.
    - Print Camera Eye and Dir: print the camera's eye and direction vectors to the terminal.
    - Debug Render Scale: scale the debug rendering.
    - Camera Speed: change the camera speed.
    - FPS: shows current frame rate.
    """
    def __register_line_shape(self, shape: UserLineShape) -> None:
        """
        Register a line shape derived from :class:`UserLineShape`.
        
        If a line shape is already registered an error will be printed and no duplicate will be inserted.
        
        Note: do not use this method directly.  Rather, use
        ``gym_render.register_line_shape()``, which ensures that registered shapes are not
        garbage-collected by Python.
        """
    def __register_menu_item(self, item: UserMenuItem) -> None:
        """
        Register a menu item derived from :class:`UserMenuItem`.
        
        If an item is already registered an error will be printed and no duplicate will be inserted.
        
        Note: do not use this method directly.  Rather, use
        ``gym_render.register_menu_item()``, which ensures that registered menu items
        are not garbage-collected by Python.
        """
    def __unregister_line_shape(self, shape: UserLineShape) -> None:
        """
        Unregister a line shape derived from :class:`UserLineShape`.
        
        Note: do not use this method directly.  Rather, use
        ``gym_render.unregister_line_shape()``, which ensures that unregistered shapes
        can be garbage-collected by Python.
        """
    def __unregister_menu_item(self, item: UserMenuItem) -> None:
        """
        Unregister a menu item derived from :class:`UserMenuItem`.
        
        Note: do not use this method directly.  Rather, use
        ``gym_render.unregister_menu_item()``, which ensures that unregistered menu
        items can be garbage-collected by Python.
        """
    def create_user_line(self, points: list[Vec3], color: Vec3, line_width: float = 1.0, env_handle: EnvironmentHandle = None) -> UserLine:
        """
        Creates a :class:`UserLine` with the ``color`` that sequentially visits the points in
        ``points``.
        
        :param points: a list of :class:`Vec3` points defining the line
        :type points: list[Vec3]
        :param color: color in RGB floating point values
        :type color: Vec3
        :param line_width: width of the rendered lines (default 1.0f)
        :type line_width: float
        :param env_handle: if supplied, transforms the points into the space of the given environment.
            If not supplied, the points are defined in world space (this is the default behaviour).
        :type env_handle: EnvironmentHandle
        :return: the created :class:`UserLine` object
        :rtype: UserLine
        """
    def create_user_line_cube(self, size: float, transform: Transform, color: Vec3, line_width: float = 1.0, env_handle: EnvironmentHandle = None) -> UserLineCube:
        """
        Constructs a line cube with the provided ``size``, ``transform``, ``color``.
        
        :param size: the size of the cube
        :type size: float
        :param transform: the transform of the cube
        :type transform: Transform
        :param color: color in RGB floating point values
        :type color: Vec3
        :param line_width: width of the rendered lines (default 1.0f)
        :type line_width: float
        :param env_handle: if supplied, transforms the cube into the space of the given environment.
            If not supplied, the cube defined in world space (this is the default behaviour).
        :type env_handle: EnvironmentHandle
        :return: the created :class:`UserLineCube` object
        :rtype: UserLineCube
        """
    def get_camera_dir(self) -> Vec3:
        """
        Get camera direction.
        """
    def get_camera_eye(self) -> Vec3:
        """
        Get camera eye.
        """
    def is_key_down(self, key: str) -> bool:
        """
        Returns whether the supplied key is pressed down.
        The function only takes alphanumeric characters as the ``key`` argument and returns false if it doesn't fall into that range.
        """
    def is_paused(self) -> bool:
        """
        Returns whether the renderer is currently paused.
        """
    def is_step_set(self) -> bool:
        """
        Returns whether the next simulation step will be performed when the simulation is paused.
        """
    def render_function(self) -> bool:
        """
        Renders the current simulation state together with the UI and debug drawings.
        
        :return: ``False`` if the render window was closed, ``True`` otherwise
        :rtype: bool
        """
    def reset_camera(self, eye: Vec3, dir: Vec3) -> None:
        """
        Reset camera to location ``eye`` pointing in direction ``dir``.
        """
    def set_paused(self, paused: bool) -> None:
        """
        Pauses or unpauses the renderer based on the provided ``paused`` value.
        """
    def set_step(self, step: bool) -> None:
        """
        Sets whether to perform the next simulation step if the simulation is paused. The value will fall back to ``False`` once the step is taken.
        Mostly useful to take a single step when the simulation is paused.
        """
    @property
    def capped_step(self) -> bool:
        """
        Capped step.
        """
    @capped_step.setter
    def capped_step(self, arg1: bool) -> None:
        ...
    @property
    def last_time_step(self) -> float:
        """
        Last time-step.
        """
    @last_time_step.setter
    def last_time_step(self, arg1: float) -> None:
        ...
class GymSingleton:
    """
    A static class that manages the :class:`Gym` singleton.
    """
    @staticmethod
    def create_gym(with_render: bool, enable_scene_query: bool, treat_warning_as_error: bool, broad_phase_type: BroadPhaseType, max_contact_pairs: int, up_axis: Vec3, with_window: bool, max_patches: int, max_contacts: int, check_overflow: bool, working_dir: str, update_scene_dependent_components_in_step: bool, compute_deformable_kinematic: bool, cuda_device: int, seed: int, enable_graph_captures: bool, enable_streams: bool, check_memory: bool, validate_kernels: bool) -> Gym:
        """
        Creates the :class:`Gym` singleton with the specified parameters.
        
        Calling this with the singleton already created will print a fatal error.
        """
    @staticmethod
    def delete_gym() -> None:
        """
        Deletes the :class:`Gym` singleton if it exists. Prints an error otherwise.
        """
    @staticmethod
    def get_gym() -> Gym:
        """
        Gets the :class:`Gym` singleton if it was created or null otherwise.
        """
class HillMaterial:
    """
    This class defines a Hill-type muscle material used in spatial tendons.
    
    Note that the values for ``p0``, ``alpha``, and ``k`` are derived from ``a`` in the following way:
    
    .. code-block:: python
    
       p0 = a / 0.257
       alpha = p0 / 0.1
       k = a / 25
    """
    a: float
    b: float
    time_const_act: float
    time_const_deact: float
    def __init__(self) -> None:
        """
        Default constructor
        """
    @property
    def alpha(self) -> float:
        ...
    @property
    def k(self) -> float:
        ...
    @property
    def p0(self) -> float:
        ...
class HillMaterialHandle:
    """
    A class wrapping a handle for an HillMaterial.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: HillMaterialHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the HillMaterial.
        """
class HillMaterialProperty:
    """
    Members:
    
      LSE0
    
      A
    
      B
    """
    A: typing.ClassVar[HillMaterialProperty]  # value = <HillMaterialProperty.A: 1>
    B: typing.ClassVar[HillMaterialProperty]  # value = <HillMaterialProperty.B: 2>
    LSE0: typing.ClassVar[HillMaterialProperty]  # value = <HillMaterialProperty.LSE0: 0>
    __members__: typing.ClassVar[dict[str, HillMaterialProperty]]  # value = {'LSE0': <HillMaterialProperty.LSE0: 0>, 'A': <HillMaterialProperty.A: 1>, 'B': <HillMaterialProperty.B: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class HillMaterialPropertyCommand:
    """
    A command to access hill material property information. No constructor defined. Use :meth:`EnvironmentGroup.create_hill_material_property_command` instead to create these.
    """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def hill_material_handle(self) -> HillMaterialHandle:
        """
        :class:`HillMaterialHandle` of the hill material being accessed.
        """
    @property
    def property(self) -> HillMaterialProperty:
        """
        Selects the property to be accessed.  See :class:`HillMaterialProperty` for options
        """
class HillMaterialPropertyCommandGpuArray:
    """
    A GPU array of :class:`HillMaterialPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_hill_material_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`HillMaterialPropertyCommand` elements in the GPU array.
        """
class InverseDynamicsCommand:
    """
    A command for inverse dynamics that computes the joint forces required to achieve the target
    joint accelerations in ``target_joint_accelerations`` and returns the computed joint forces in the
    ``joint_forces`` GPU buffer.
    
    Created by :meth:`EnvironmentGroup.create_inverse_dynamics_command()`
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def joint_forces_data_ptr(self) -> int:
        """
        Gets the joint forces data pointer of type `float32`.
        """
    @property
    def target_joint_accelerations_data_ptr(self) -> int:
        """
        Gets the target joint accelerations data pointer of type `float32`.
        """
class InverseDynamicsCommandGpuArray:
    """
    A GPU array of :class:`InverseDynamicsCommand`.
    
    No constructor defined. Use :meth:`Gym.create_inverse_dynamics_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        ...
class InverseKinematicsCommand:
    """
    A command for inverse kinematics that takes a ``target_transforms`` GPU buffer as input, and
    returns the joint positions in the ``joint_positions`` GPU buffer.
    
    Created by :meth:`EnvironmentGroup.create_inverse_kinematics_command()`.
    """
    @property
    def joint_positions_data_ptr(self) -> int:
        ...
    @property
    def target_transforms_data_ptr(self) -> int:
        ...
class InverseKinematicsCommandGpuArray:
    """
    A GPU array of :class:`InverseKinematicsCommand`.
    
    No constructor defined. Use :meth:`Gym.create_inverse_kinematics_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        ...
class JacobianCommand:
    """
    A command that computes and returns the Jacobian matrix relating joint velocities to link
    velocities.
    
    Created by :meth:`EnvironmentGroup.create_jacobian_command()`
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def data_ptr(self) -> int:
        """
        Gets the data pointer of type `float32`.
        """
class JacobianCommandGpuArray:
    """
    A GPU array of :class:`JacobianCommand`.
    
    No constructor defined. Use :meth:`Gym.create_jacobian_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        ...
class JointActiveMaskCommand:
    """
    A command to access joint information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_joint_active_mask_command` instead to create these.
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the joint to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` joints at indices 2, 3, and 4 will be accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def start_index(self) -> int:
        """
        First index of the joint to be accessed.
        """
class JointActiveMaskCommandGpuArray:
    """
    A GPU array of :class:`JointActiveMaskCommand`.
    
    No constructor defined. Use :meth:`Gym.create_joint_state_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`JointActiveMaskCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[JointActiveMaskCommand]:
        """
        TODO: document
        """
class JointDofProperty:
    """
    Members:
    
      ARMATURE
    
      LOW_LIMIT
    
      HIGH_LIMIT
    """
    ARMATURE: typing.ClassVar[JointDofProperty]  # value = <JointDofProperty.ARMATURE: 0>
    HIGH_LIMIT: typing.ClassVar[JointDofProperty]  # value = <JointDofProperty.HIGH_LIMIT: 2>
    LOW_LIMIT: typing.ClassVar[JointDofProperty]  # value = <JointDofProperty.LOW_LIMIT: 1>
    __members__: typing.ClassVar[dict[str, JointDofProperty]]  # value = {'ARMATURE': <JointDofProperty.ARMATURE: 0>, 'LOW_LIMIT': <JointDofProperty.LOW_LIMIT: 1>, 'HIGH_LIMIT': <JointDofProperty.HIGH_LIMIT: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class JointDofPropertyCommand:
    """
    A command to access joint dof property information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_joint_dof_property_command` instead to create these.
    """
    @property
    def articulation_def_handle(self) -> ArticulationDefHandle:
        """
        :class:`ArticulationDefHandle` of the articulation def being accessed.
        """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the joint dof to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` joint dofs at indices 2, 3, and 4 will be accessed.
        """
    @property
    def property(self) -> JointDofProperty:
        """
        Selects the property to be accessed.  See :class:`JointDofProperty` for options
        """
    @property
    def start_index(self) -> int:
        """
        First index of the joint dof to be accessed.
        """
class JointDofPropertyCommandGpuArray:
    """
    A GPU array of :class:`JointDofPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_joint_dof_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`JointDofPropertyCommand` elements in the GPU array.
        """
class JointForceSensorCommand:
    """
    A command to access joint force sensor information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_joint_force_sensor_command` instead to create these.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``joint_sensor_data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the joint force sensor to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` joint force sensors at indices 2, 3, and 4 will be accessed.
        """
    @property
    def force_sensor_handle(self) -> ArticulationJointSensorHandle:
        """
        :class:`ArticulationJointSensorHandle` of the joint force sensor being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def start_index(self) -> int:
        """
        First index of the joint force sensor to be accessed.
        """
class JointForceSensorCommandGpuArray:
    """
    A GPU array of :class:`JointForceSensorCommand`.
    
    No constructor defined. Use :meth:`Gym.create_joint_force_sensor_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`JointForceSensorCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[JointForceSensorCommand]:
        """
        Retrieves the net joint force acting on each joint. This includes actuation and constraint forces.
        """
class JointProperty:
    """
    Members:
    
      JOINT_FRICTION
    
      MAX_JOINT_VELOCITY
    
      CHILD_FRAME
    """
    CHILD_FRAME: typing.ClassVar[JointProperty]  # value = <JointProperty.CHILD_FRAME: 3>
    JOINT_FRICTION: typing.ClassVar[JointProperty]  # value = <JointProperty.JOINT_FRICTION: 0>
    MAX_JOINT_VELOCITY: typing.ClassVar[JointProperty]  # value = <JointProperty.MAX_JOINT_VELOCITY: 2>
    __members__: typing.ClassVar[dict[str, JointProperty]]  # value = {'JOINT_FRICTION': <JointProperty.JOINT_FRICTION: 0>, 'MAX_JOINT_VELOCITY': <JointProperty.MAX_JOINT_VELOCITY: 2>, 'CHILD_FRAME': <JointProperty.CHILD_FRAME: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class JointPropertyCommand:
    """
    A command to access joint property information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_joint_property_command` instead to create these.
    """
    @property
    def articulation_def_handle(self) -> ArticulationDefHandle:
        """
        :class:`ArticulationDefHandle` of the articulation def being accessed.
        """
    @articulation_def_handle.setter
    def articulation_def_handle(self, arg0: ArticulationDefHandle) -> None:
        ...
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the joint to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` joints at indices 2, 3, and 4 will be accessed.
        """
    @end_index.setter
    def end_index(self, arg0: int) -> None:
        ...
    @property
    def property(self) -> JointProperty:
        """
        Selects the property to be accessed.  See :class:`JointProperty` for options
        """
    @property.setter
    def property(self, arg0: JointProperty) -> None:
        ...
    @property
    def start_index(self) -> int:
        """
        First index of the joint to be accessed.
        """
    @start_index.setter
    def start_index(self, arg0: int) -> None:
        ...
class JointPropertyCommandGpuArray:
    """
    A GPU array of :class:`JointPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_joint_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`JointPropertyCommand` elements in the GPU array.
        """
class JointStateCommand:
    """
    A command to access joint information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_joint_state_command` instead to create these.
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the joint to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` joints at indices 2, 3, and 4 will be accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def start_index(self) -> int:
        """
        First index of the joint to be accessed.
        """
class JointStateCommandGpuArray:
    """
    A GPU array of :class:`JointStateCommand`.
    
    No constructor defined. Use :meth:`Gym.create_joint_state_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`JointStateCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[JointStateCommand]:
        """
        TODO: document
        """
class KinematicSensorHandle:
    """
    A class wrapping a handle for an articulation kinematic sensor.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: KinematicSensorHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def env_def_index(self) -> int:
        """
        Gets the underlying index of the :class:`EnvironmentDef` that this
        :class:`KinematicSensorHandle` belongs to.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the kinematic sensor.
        """
    def type(self) -> int:
        """
        Gets the type of the kinematic sensor.
        
        See :class:`KinematicSensorType` for the meaning of the type value.
        """
class KinematicSensorStateCommand:
    """
    A command to access kinematic sensor state information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_kinematic_sensor_state_command` instead to create these.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    
    :class:`SpatialVector` objects are written to and
    read from data buffers as ``(top.x, top.y, top.z, bottom.x, bottom.y, bottom.z)``.
    
    The angular velocity is stored in the ``top`` part of the :class:`SpatialVector`, and
    the linear velocity is stored in the ``bottom`` part of the :class:`SpatialVector`.
    
    Note: you need to run :meth:`Gym.step()` at least once before getting correct values.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``sensor_data`` cast to a number type.
        """
    @property
    def kinematic_sensor_handle(self) -> KinematicSensorHandle:
        """
        :class:`KinematicSensorHandle` of the kinematic sensor being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class KinematicSensorStateCommandGpuArray:
    """
    A GPU array of :class:`KinematicSensorStateCommand`.
    
    No constructor defined. Use :meth:`Gym.create_kinematic_sensor_state_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`KinematicSensorStateCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[KinematicSensorStateCommand]:
        """
        TODO: document
        """
class KinematicSensorTransformCommand:
    """
    A command to access kinematic sensor transform information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_kinematic_sensor_transform_command` instead to create these.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    
    Note: you need to run :meth:`Gym.step()` at least once before getting correct values.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``sensor_data`` cast to a number type.
        """
    @property
    def kinematic_sensor_handle(self) -> KinematicSensorHandle:
        """
        :class:`KinematicSensorHandle` of the kinematic sensor being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class KinematicSensorTransformCommandGpuArray:
    """
    A GPU array of :class:`KinematicSensorTransformCommand`.
    
    No constructor defined. Use :meth:`Gym.create_kinematic_sensor_transform_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`KinematicSensorTransformCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[KinematicSensorTransformCommand]:
        """
        TODO: document
        """
class KinematicSensorType:
    """
    The type of kinematic sensor.
    
    Members:
    
      ARTICULATION : The kinematic sensor belongs to an articulation.
    
      RIGID_BODY : The kinematic sensor belongs to a rigid body.
    """
    ARTICULATION: typing.ClassVar[KinematicSensorType]  # value = <KinematicSensorType.ARTICULATION: 0>
    RIGID_BODY: typing.ClassVar[KinematicSensorType]  # value = <KinematicSensorType.RIGID_BODY: 1>
    __members__: typing.ClassVar[dict[str, KinematicSensorType]]  # value = {'ARTICULATION': <KinematicSensorType.ARTICULATION: 0>, 'RIGID_BODY': <KinematicSensorType.RIGID_BODY: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KinematicSensorVelocityCommand:
    """
    A command to access kinematic sensor velocity information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_kinematic_sensor_velocity_command` instead to create these.
    
    :class:`SpatialVector` objects are written to and
    read from data buffers as ``(top.x, top.y, top.z, bottom.x, bottom.y, bottom.z)``.
    
    The angular velocity is stored in the ``top`` part of the :class:`SpatialVector`, and
    the linear velocity is stored in the ``bottom`` part of the :class:`SpatialVector`.
    
    Note: you need to run :meth:`Gym.step()` at least once before getting correct values.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``sensor_data`` cast to a number type.
        """
    @property
    def kinematic_sensor_handle(self) -> KinematicSensorHandle:
        """
        :class:`KinematicSensorHandle` of the kinematic sensor being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class KinematicSensorVelocityCommandGpuArray:
    """
    A GPU array of :class:`KinematicSensorVelocityCommand`.
    
    No constructor defined. Use :meth:`Gym.create_kinematic_sensor_velocity_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`KinematicSensorVelocityCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[KinematicSensorVelocityCommand]:
        """
        TODO: document
        """
class LightHandle:
    """
    A class wrapping a handle for a light.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: LightHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the light.
        """
class LightTransformCommand:
    """
    A command to access light transform information. No constructor defined. Use :meth:`EnvironmentGroup.create_light_transform_command` instead to create these.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def light_handle(self) -> LightHandle:
        """
        :class:`LightHandle` of the light being accessed.
        """
class LightTransformCommandGpuArray:
    """
    A GPU array of :class:`LightTransformCommand`.
    
    No constructor defined. Use :meth:`Gym.create_light_transform_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`LightTransformCommand` elements in the GPU array.
        """
class LinkExternalForceCommand:
    """
    A command to access link external force information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_link_external_force_command` instead to create these.
    
    :class:`SpatialVector` objects are written to and
    read from data buffers as ``(top.x, top.y, top.z, bottom.x, bottom.y, bottom.z)``.
    
    In the case of ``force_type=FORCE_TORQUE``, the force is
    stored in the ``top`` part of the :class:`SpatialVector`, and the torque is stored in the ``bottom``
    part of the :class:`SpatialVector`.
    
    In the case of ``force_type=FORCE_POSITION``, the force is stored in the ``top`` part of the
    :class:`SpatialVector`, and the position is stored in the ``bottom`` part of the
    :class:`SpatialVector`.
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the link external force to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` link external forces at indices 2, 3, and 4 will be accessed.
        """
    @property
    def force_type(self) -> ForceType:
        """
        Format in which forces are specified
        """
    @property
    def frame_type(self) -> FrameType:
        """
        Frame in which forces are specified
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def start_index(self) -> int:
        """
        First index of the link external force to be accessed.
        """
class LinkExternalForceCommandGpuArray:
    """
    A GPU array of :class:`LinkExternalForceCommand`.
    
    No constructor defined. Use :meth:`Gym.create_link_external_force_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`LinkExternalForceCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[LinkExternalForceCommand]:
        """
        TODO: document
        """
class LinkProperty:
    """
    Members:
    
      MASS
    """
    MASS: typing.ClassVar[LinkProperty]  # value = <LinkProperty.MASS: 0>
    __members__: typing.ClassVar[dict[str, LinkProperty]]  # value = {'MASS': <LinkProperty.MASS: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LinkPropertyCommand:
    """
    A command to access link property information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_link_property_command` instead to create these.
    """
    @property
    def articulation_def_handle(self) -> ArticulationDefHandle:
        """
        :class:`ArticulationDefHandle` of the articulation def being accessed.
        """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the link to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` links at indices 2, 3, and 4 will be accessed.
        """
    @property
    def property(self) -> LinkProperty:
        """
        Selects the property to be accessed.  See :class:`LinkProperty` for options
        """
    @property
    def start_index(self) -> int:
        """
        First index of the link to be accessed.
        """
class LinkPropertyCommandGpuArray:
    """
    A GPU array of :class:`LinkPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_link_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`LinkPropertyCommand` elements in the GPU array.
        """
class LinkTransformCommand:
    """
    A command to access link transform information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_link_transform_command` instead to create these.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the link transform to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` link transforms at indices 2, 3, and 4 will be accessed.
        """
    @property
    def frame_type(self) -> FrameType:
        """
        Frame of transform (only for getting transforms)
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def start_index(self) -> int:
        """
        First index of the link transform to be accessed.
        """
    @property
    def transform_type(self) -> TransformType:
        """
        Type of transform (only for getting transforms)
        """
class LinkTransformCommandGpuArray:
    """
    A GPU array of :class:`LinkTransformCommand`.
    
    No constructor defined. Use :meth:`Gym.create_link_transform_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`LinkTransformCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[LinkTransformCommand]:
        """
        TODO: document
        """
class LinkVelocityCommand:
    """
    A command to access link spatial velocity information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_link_velocity_command` instead to create these.
    
    :class:`SpatialVector` objects are written to and
    read from data buffers as ``(top.x, top.y, top.z, bottom.x, bottom.y, bottom.z)``.
    
    The angular velocity is stored in the ``top`` part of the :class:`SpatialVector`, and
    the linear velocity is stored in the ``bottom`` part of the :class:`SpatialVector`.
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the link spatial velocity to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` link spatial velocities at indices 2, 3, and 4 will be accessed.
        """
    @property
    def frame_type(self) -> FrameType:
        """
        Frame in which velocities are defined
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def start_index(self) -> int:
        """
        First index of the link spatial velocity to be accessed.
        """
class LinkVelocityCommandGpuArray:
    """
    A GPU array of :class:`LinkVelocityCommand`.
    
    No constructor defined. Use :meth:`Gym.create_link_velocity_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`LinkVelocityCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[LinkVelocityCommand]:
        """
        TODO: document
        """
class MassMatrixCommand:
    """
    A command that computes and returns the generalized mass matrix for an articulation.
    
    Created by :meth:`EnvironmentGroup.create_mass_matrix_command()`
    """
    @property
    def articulation_handle(self) -> ArticulationHandle:
        """
        :class:`ArticulationHandle` of the articulation being accessed.
        """
    @property
    def data_ptr(self) -> int:
        """
        Gets the data pointer of type `float32`.
        """
class MassMatrixCommandGpuArray:
    """
    A GPU array of :class:`MassMatrixCommand`.
    
    No constructor defined. Use :meth:`Gym.create_mass_matrix_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        ...
class Model(Finalizable):
    """
    Base class for ArticulationModel and RigidBodyModel
    """
    def get_depth_camera_handle(self, index: int, camera_type: CameraType = ...) -> DepthCameraHandle:
        """
        Gets the :class:`DepthCameraHandle` for `index` on this model instance.
        """
    def get_light_handle(self, index: int) -> LightHandle:
        """
        Gets the :class:`LightHandle` for `index` on this model instance.
        """
    def get_local_transform(self) -> Transform:
        """
        Gets the transform of the model in environment space.
        """
    def get_name(self) -> str:
        """
        Gets the name of the model.
        """
    def get_rgb_camera_handle(self, index: int, camera_type: CameraType = ...) -> RGBCameraHandle:
        """
        Gets the :class:`RGBCameraHandle` for `index` on this model instance.
        """
class ModelDef(Finalizable):
    """
    Base class for ArticulationDef and RigidBodyDef
    """
    def get_depth_camera(self, index: int, camera_type: CameraType = ...) -> DepthCamera:
        """
        Gets the depth camera at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_depth_camera_def(self, index: int, camera_type: CameraType = ...) -> DepthCameraDef:
        """
        Gets the depth camera definition at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_depth_camera_def_name(self, index: int, camera_type: CameraType = ...) -> str:
        """
        Gets the depth camera definition name at ``index``.
        """
    def get_depth_camera_name(self, index: int, camera_type: CameraType = ...) -> str:
        """
        Gets the depth camera name at ``index``.
        """
    def get_filename(self) -> str:
        """
        Gets the name of the file the definition was created from.
        """
    def get_format(self) -> str:
        """
        Gets the file format of the file the definition was imported from.
        """
    def get_name(self) -> str:
        """
        Gets the name of the definition
        """
    def get_num_depth_camera_defs(self, camera_type: CameraType = ...) -> int:
        """
        Gets the number of depth camera definitions contained in the articulation definition.
        """
    def get_num_depth_cameras(self, camera_type: CameraType = ...) -> int:
        """
        Gets the number of depth cameras contained in the definition.
        """
    def get_num_rgb_camera_defs(self, camera_type: CameraType = ...) -> int:
        """
        Gets the number of RGB camera definitions contained in the definition.
        """
    def get_num_rgb_cameras(self, camera_type: CameraType = ...) -> int:
        """
        Gets the number of RGB cameras contained in the definition.
        """
    def get_rgb_camera(self, index: int, camera_type: CameraType = ...) -> RGBCamera:
        """
        Gets the RGB camera at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_rgb_camera_def(self, index: int, camera_type: CameraType = ...) -> RGBCameraDef:
        """
        Gets the RGB camera definition at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_rgb_camera_def_handle(self, index: int, camera_type: CameraType = ...) -> RGBCameraDefHandle:
        """
        Gets the RGB camera definition handle at ``index``.
        """
    def get_rgb_camera_def_name(self, index: int, camera_type: CameraType = ...) -> str:
        """
        Gets the RGB camera definition name at ``index``.
        """
    def get_rgb_camera_name(self, index: int, camera_type: CameraType = ...) -> str:
        """
        Gets the pinhole RGB camera name at ``index``.
        """
    def get_root_link_transform(self) -> Transform:
        """
        Gets the :class:`Transform` of the root link.
        """
    @property
    def contact_offset(self) -> float:
        """
        Contact offset used to control the distance at which contacts are generated.
        
        .. note::
            A negative value means that the imported definition value will not be overriden.
        """
    @contact_offset.setter
    def contact_offset(self, arg1: float) -> None:
        ...
    @property
    def fixed(self) -> bool:
        """
        Controls whether the articulation or rigid body has its base link fixed in world space.
        """
    @fixed.setter
    def fixed(self, arg1: bool) -> None:
        ...
    @property
    def gravity_scale(self) -> float:
        """
        Gravity scaling applied to the body.
        
        .. note::
            A negative value means that the imported definition value will not be overriden.
        """
    @gravity_scale.setter
    def gravity_scale(self, arg1: float) -> None:
        ...
    @property
    def max_depenetration_velocity(self) -> float:
        """
        Max velocity the solver can introduce to correct collision errors.
        
        .. note::
            A negative value means that the imported definition value will not be overriden.
        """
    @max_depenetration_velocity.setter
    def max_depenetration_velocity(self, arg1: float) -> None:
        ...
    @property
    def rest_offset(self) -> float:
        """
        Rest offset is the distance that will be held when two object collide.
        
        .. note::
            A ``nan`` value means that the imported definition value will not be overriden.
        """
    @rest_offset.setter
    def rest_offset(self, arg1: float) -> None:
        ...
class MotorControlCommand:
    """
    A command to access motor control information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_motor_control_command` instead to create these.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the motor to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` motors at indices 2, 3, and 4 will be accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def motor_control_handle(self) -> ArticulationMotorHandle:
        """
        :class:`ArticulationMotorHandle` of the motor control being accessed.
        """
    @property
    def start_index(self) -> int:
        """
        First index of the motor to be accessed.
        """
class MotorControlCommandGpuArray:
    """
    A GPU array of :class:`MotorControlCommand`.
    
    No constructor defined. Use :meth:`Gym.create_motor_control_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`MotorControlCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[MotorControlCommand]:
        """
        TODO: document
        """
class MotorDef:
    """
    A class holding the data for a motor definition.
    """
    @property
    def dof_index(self) -> int:
        """
        The index of the DOF the motor is on.
        
        It is specified in the global space of the whole articulation, not locally to a particular
        link.
        """
    @property
    def gear_ratio(self) -> float:
        """
        Gear ratio of the motor.
        """
    @gear_ratio.setter
    def gear_ratio(self, arg1: float) -> None:
        ...
    @property
    def high_limit(self) -> float:
        """
        High limit of the motor.
        """
    @high_limit.setter
    def high_limit(self, arg1: float) -> None:
        ...
    @property
    def low_limit(self) -> float:
        """
        Low limit of the motor.
        """
    @low_limit.setter
    def low_limit(self, arg1: float) -> None:
        ...
    @property
    def name(self) -> str:
        """
        The name of the motor
        """
class MotorProperty:
    """
    Members:
    
      GEAR_RATIO
    
      LOW_LIMIT
    
      HIGH_LIMIT
    """
    GEAR_RATIO: typing.ClassVar[MotorProperty]  # value = <MotorProperty.GEAR_RATIO: 0>
    HIGH_LIMIT: typing.ClassVar[MotorProperty]  # value = <MotorProperty.HIGH_LIMIT: 2>
    LOW_LIMIT: typing.ClassVar[MotorProperty]  # value = <MotorProperty.LOW_LIMIT: 1>
    __members__: typing.ClassVar[dict[str, MotorProperty]]  # value = {'GEAR_RATIO': <MotorProperty.GEAR_RATIO: 0>, 'LOW_LIMIT': <MotorProperty.LOW_LIMIT: 1>, 'HIGH_LIMIT': <MotorProperty.HIGH_LIMIT: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MotorPropertyCommand:
    """
    A command to access motor property information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_motor_property_command` instead to create these.
    """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the motor to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` motors at indices 2, 3, and 4 will be accessed.
        """
    @property
    def motor_control_def_handle(self) -> ArticulationMotorDefHandle:
        """
        :class:`ArticulationMotorDefHandle` of the motor def being accessed.
        """
    @property
    def property(self) -> MotorProperty:
        """
        Selects the property to be accessed.  See :class:`MotorProperty` for options
        """
    @property
    def start_index(self) -> int:
        """
        First index of the motor to be accessed.
        """
class MotorPropertyCommandGpuArray:
    """
    A GPU array of :class:`MotorPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_motor_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`MotorPropertyCommand` elements in the GPU array.
        """
class PIDControlCommand:
    """
    A command to access PID control information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_pid_control_command` instead to create these.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the PID to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` motors at indices 2, 3, and 4 will be accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def pid_control_handle(self) -> ArticulationPIDHandle:
        """
        :class:`ArticulationPIDHandle` of the PID control being accessed.
        """
    @property
    def start_index(self) -> int:
        """
        First index of the PID to be accessed.
        """
class PIDControlCommandGpuArray:
    """
    A GPU array of :class:`PIDControlCommand`.
    
    No constructor defined. Use :meth:`Gym.create_pid_control_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`PIDControlCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[PIDControlCommand]:
        """
        TODO: document
        """
class PIDDef:
    """
    A class holding the data for a PID controller definition.
    """
    @property
    def damping(self) -> float:
        """
        The damping value of the PID controller's spring.
        """
    @damping.setter
    def damping(self, arg1: float) -> None:
        ...
    @property
    def integral(self) -> float:
        """
        The integral value of the PID controller.
        """
    @integral.setter
    def integral(self, arg1: float) -> None:
        ...
    @property
    def link_index(self) -> int:
        """
        Index of the link this PID controller belongs to.
        """
    @property
    def local_dof_index(self) -> int:
        """
        Index of the DOF this PID controller controls. Specified local to the link at
        :attr:`link_index`.
        """
    @property
    def max_force(self) -> float:
        """
        Maximum force that the PID controller is able to exert.
        """
    @max_force.setter
    def max_force(self, arg1: float) -> None:
        ...
    @property
    def name(self) -> str:
        """
        The name of the PID controller.
        """
    @property
    def stiffness(self) -> float:
        """
        The stiffness value of the PID controller's spring.
        """
    @stiffness.setter
    def stiffness(self, arg1: float) -> None:
        ...
class PIDProperty:
    """
    Members:
    
      STIFFNESS
    
      DAMPING
    
      MAX_FORCE
    """
    DAMPING: typing.ClassVar[PIDProperty]  # value = <PIDProperty.DAMPING: 1>
    MAX_FORCE: typing.ClassVar[PIDProperty]  # value = <PIDProperty.MAX_FORCE: 2>
    STIFFNESS: typing.ClassVar[PIDProperty]  # value = <PIDProperty.STIFFNESS: 0>
    __members__: typing.ClassVar[dict[str, PIDProperty]]  # value = {'STIFFNESS': <PIDProperty.STIFFNESS: 0>, 'DAMPING': <PIDProperty.DAMPING: 1>, 'MAX_FORCE': <PIDProperty.MAX_FORCE: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PIDPropertyCommand:
    """
    A command to access PID property information of an articulation.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_pid_property_command` instead to create these.
    """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def end_index(self) -> int:
        """
        Exclusive last index of the PID controller to be accessed.
        
        The total range looks like ``[start_index; end_index)``.
        
        E.g. with ``start_index = 2`` and ``end_index = 5`` PID controllers at indices 2, 3, and 4 will be accessed.
        """
    @property
    def pid_control_def_handle(self) -> ArticulationPIDDefHandle:
        """
        :class:`ArticulationPIDDefHandle` of the PID def being accessed.
        """
    @property
    def property(self) -> PIDProperty:
        """
        Selects the property to be accessed.  See :class:`PIDProperty` for options
        """
    @property
    def start_index(self) -> int:
        """
        First index of the PID controller to be accessed.
        """
class PIDPropertyCommandGpuArray:
    """
    A GPU array of :class:`PIDPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_pid_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`PIDPropertyCommand` elements in the GPU array.
        """
class Quat:
    """
    Quaternion that represents a rotation in 3D space.
    """
    __hash__: typing.ClassVar[None] = None
    def __abs__(self) -> float:
        """
        Returns the magnitude, computed from all 4 components of the :class:`Quat`.
        This is not the total angle this :class:`Quat` rotates by.
        """
    def __add__(self, other: Quat) -> Quat:
        """
        Adds the two :class:`Quat`'s components together.
        """
    def __eq__(self, other: Quat) -> bool:
        """
        Checks whether the :class:`Quat`'s are equal. Uses an exact bitwise comparison with no epsilon.
        """
    def __iadd__(self, other: Quat) -> Quat:
        """
        Adds the two :class:`Quat`'s components together in place.
        """
    def __imul__(self, other: Quat) -> Quat:
        """
        Combines the rotation of the right :class:`Quat` into the left :class:`Quat` in place.
        """
    @typing.overload
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        """
        Constructs a Quat from 3 real numbers (*x*, *y*, *z*) and an imaginary number *w*.
        """
    @typing.overload
    def __init__(self, axis: Vec3, angle: float) -> None:
        """
        Constructs a Quat from a normalized axis and a rotation magnitude in radians
        """
    @typing.overload
    def __init__(self, other: Quat) -> None:
        """
        Copy constructor.
        """
    def __isub__(self, other: Quat) -> Quat:
        """
        Subtracts the right :class:`Quat`'s components from the left :class:`Quat` in place.
        """
    @typing.overload
    def __mul__(self, other: Quat) -> Quat:
        """
        Combines the rotations of two operands.
        Using the resulting rotation is equivalent to rotating by both operands from left to right.
        Thus this multiplication is not commutative!
        """
    @typing.overload
    def __mul__(self, scalar: float) -> Quat:
        """
        Multiply this quaternion with a scalar.
        """
    def get_conjugate(self) -> Quat:
        """
        Returns an inverse :class:`Quat` that performs an opposite rotation.
        """
    def get_normalized(self) -> Quat:
        """
        Returns a normalized version of this :class:`Quat` with a magnitude of 1.
        """
    def rotate(self, vector: Vec3) -> Vec3:
        """
        Rotates the supplied vector by this :class:`Quat`.
        """
    def rotate_inv(self, vector: Vec3) -> Vec3:
        """
        Performs an inverse rotation of this :class:`Quat` on the supplied vector.
        """
    @property
    def w(self) -> float:
        """
        The imaginary *W* component of the quaternion.
        """
    @w.setter
    def w(self, arg0: float) -> None:
        ...
    @property
    def x(self) -> float:
        """
        The real *X* component of the quaternion.
        """
    @x.setter
    def x(self, arg0: float) -> None:
        ...
    @property
    def y(self) -> float:
        """
        The real *Y* component of the quaternion.
        """
    @y.setter
    def y(self, arg0: float) -> None:
        ...
    @property
    def z(self) -> float:
        """
        The real *Z* component of the quaternion.
        """
    @z.setter
    def z(self, arg0: float) -> None:
        ...
class QueryMode:
    """
    Mode to use for importing query geometries.
    
    Members:
    
      USE_FILE : Uses collisions and visuals explicitly marked with ``useQueries="true"`` in the model file.
    
      USE_VISUALS : Uses visual geometries as query geometry source, no matter what the file specifies. If no visual geometries are defined in the file, no query geometries will be created.
    
      USE_COLLISIONS : Uses collision geometries as query geometry source, no matter what the file specifies. If no collision geometries are defined in the file, no query geometries will be created.
    
      USE_NONE : Uses no query geometry, no matter what the file specifies.
    """
    USE_COLLISIONS: typing.ClassVar[QueryMode]  # value = <QueryMode.USE_COLLISIONS: 2>
    USE_FILE: typing.ClassVar[QueryMode]  # value = <QueryMode.USE_FILE: 0>
    USE_NONE: typing.ClassVar[QueryMode]  # value = <QueryMode.USE_NONE: 3>
    USE_VISUALS: typing.ClassVar[QueryMode]  # value = <QueryMode.USE_VISUALS: 1>
    __members__: typing.ClassVar[dict[str, QueryMode]]  # value = {'USE_FILE': <QueryMode.USE_FILE: 0>, 'USE_VISUALS': <QueryMode.USE_VISUALS: 1>, 'USE_COLLISIONS': <QueryMode.USE_COLLISIONS: 2>, 'USE_NONE': <QueryMode.USE_NONE: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RGBCamera(Camera):
    """
    A class holding the data for an RGB camera instance.
    """
    @property
    def base_color(self) -> Vec3:
        """
        Color reported for no-hits.
        """
    @base_color.setter
    def base_color(self, arg1: Vec3) -> None:
        ...
class RGBCameraCommand:
    """
    A command for RGB cameras.
    
    The data can be stored either as unsigned 8-byte integers or 32-bit floating-point values.
    
    The data type is derived from the ``GpuBufferWrapper`` type passed to
    :meth:`EnvironmentGroup.create_rgb_camera_command()`, which can be :class:`Uint8GpuBufferWrapper` or
    :class:`Float32GpuBufferWrapper`.
    
    The ``data`` buffer must have size
     - num_envs * resolution_y * resolution_x * 4 * 1 bytes (uint8)
     - num_envs * resolution_y * resolution_x * 4 * 4 bytes (float32)
     
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rgb_camera_handle(self) -> RGBCameraHandle:
        """
        :class:`RGBCameraHandle` of the RGB camera being accessed.
        """
class RGBCameraCommandGpuArray:
    """
    A GPU array of :class:`RGBCameraCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rgb_camera_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RGBCameraCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[RGBCameraCommand]:
        """
        TODO: document
        """
class RGBCameraDef:
    """
    A class holding the data for a camera definition.
    """
    @property
    def far_clip(self) -> float:
        """
        Far clip of camera def.
        """
    @far_clip.setter
    def far_clip(self, arg1: float) -> None:
        ...
    @property
    def field_of_view_x(self) -> float:
        """
        Field of view of camera def along x-axis in radians.
        """
    @field_of_view_x.setter
    def field_of_view_x(self, arg1: float) -> None:
        ...
    @property
    def field_of_view_y(self) -> float:
        """
        Field of view of camera def along y-axis in radians.
        """
    @field_of_view_y.setter
    def field_of_view_y(self, arg1: float) -> None:
        ...
    @property
    def name(self) -> str:
        """
        Name of camera def.
        """
    @property
    def post_filter(self) -> CameraPostFilter:
        """
        Post filter for camera rendering.
        """
    @post_filter.setter
    def post_filter(self, arg1: CameraPostFilter) -> None:
        ...
    @property
    def resolution_x(self) -> int:
        """
        Resolution of camera def along x-axis.
        """
    @resolution_x.setter
    def resolution_x(self, arg1: int) -> None:
        ...
    @property
    def resolution_y(self) -> int:
        """
        Resolution of camera def along y-axis.
        """
    @resolution_y.setter
    def resolution_y(self, arg1: int) -> None:
        ...
class RGBCameraDefHandle:
    """
    A class wrapping a handle for an RGB camera def.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: RGBCameraDefHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the RGB camera def.
        """
class RGBCameraHandle:
    """
    A class wrapping a handle for an RGB camera.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: RGBCameraHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the RGB camera.
        """
class RGBCameraSkybox:
    """
    Members:
    
      TEXTURE
    """
    TEXTURE: typing.ClassVar[RGBCameraSkybox]  # value = <RGBCameraSkybox.TEXTURE: 0>
    __members__: typing.ClassVar[dict[str, RGBCameraSkybox]]  # value = {'TEXTURE': <RGBCameraSkybox.TEXTURE: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RGBCameraSkyboxCommand:
    """
    A command to access rgb camera property information. No constructor defined. Use :meth:`EnvironmentGroup.create_rgb_camera_transform_command` instead to create these.
    """
    @property
    def data_ptr(self) -> TextureHandle:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def property(self) -> RGBCameraSkybox:
        """
        Selects the property to be accessed.  See :class:`RGBCameraTransform` for options
        """
    @property
    def rgb_camera_handle(self) -> RGBCameraDefHandle:
        """
        :class:`RGBCameraHandle` of the rgb camera being accessed.
        """
class RGBCameraSkyboxCommandGpuArray:
    """
    A GPU array of :class:`RGBCameraSkyboxCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rgb_camera_transform_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RGBCameraSkyboxCommand` elements in the GPU array.
        """
class RGBCameraTransformCommand:
    """
    A command to access rgb camera transform information. No constructor defined. Use :meth:`EnvironmentGroup.create_rgb_camera_transform_command` instead to create these.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def rgb_camera_handle(self) -> RGBCameraHandle:
        """
        :class:`RGBCameraHandle` of the rgb camera being accessed.
        """
class RGBCameraTransformCommandGpuArray:
    """
    A GPU array of :class:`RGBCameraTransformCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rgb_camera_transform_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RGBCameraTransformCommand` elements in the GPU array.
        """
class RGBMaterial:
    color: Vec3
    luminescence: float
    spec_intensity: float
    specular: float
    def __init__(self) -> None:
        """
        Default constructor
        """
class RGBMaterialHandle:
    """
    A class wrapping a handle for an RGBMaterial.
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def invalid() -> RGBMaterialHandle:
        """
        Returns an invalid handle.
        """
    def __eq__(self, arg0: RGBMaterialHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the RGBMaterial.
        """
    def is_valid(self) -> bool:
        """
        Returns whether the handle is valid.
        """
class RGBMaterialProperty:
    """
    Members:
    
      COLOR
    
      LUMINESCENE
    
      TEXTURE
    
      SPECULAR_EXPONENT
    
      SPECULAR_INTENSITY
    """
    COLOR: typing.ClassVar[RGBMaterialProperty]  # value = <RGBMaterialProperty.COLOR: 0>
    LUMINESCENE: typing.ClassVar[RGBMaterialProperty]  # value = <RGBMaterialProperty.LUMINESCENE: 1>
    SPECULAR_EXPONENT: typing.ClassVar[RGBMaterialProperty]  # value = <RGBMaterialProperty.SPECULAR_EXPONENT: 3>
    SPECULAR_INTENSITY: typing.ClassVar[RGBMaterialProperty]  # value = <RGBMaterialProperty.SPECULAR_INTENSITY: 4>
    TEXTURE: typing.ClassVar[RGBMaterialProperty]  # value = <RGBMaterialProperty.TEXTURE: 2>
    __members__: typing.ClassVar[dict[str, RGBMaterialProperty]]  # value = {'COLOR': <RGBMaterialProperty.COLOR: 0>, 'LUMINESCENE': <RGBMaterialProperty.LUMINESCENE: 1>, 'TEXTURE': <RGBMaterialProperty.TEXTURE: 2>, 'SPECULAR_EXPONENT': <RGBMaterialProperty.SPECULAR_EXPONENT: 3>, 'SPECULAR_INTENSITY': <RGBMaterialProperty.SPECULAR_INTENSITY: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RGBMaterialPropertyCommand:
    """
    A command to access rgb material property information. No constructor defined. Use :meth:`EnvironmentGroup.create_rgb_material_property_command` instead to create these.
    """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def property(self) -> RGBMaterialProperty:
        """
        Selects the property to be accessed.  See :class:`RGBMaterialProperty` for options
        """
    @property
    def rgb_material_handle(self) -> RGBMaterialHandle:
        """
        :class:`RGBMaterialHandle` of the rgb material being accessed.
        """
class RGBMaterialPropertyCommandGpuArray:
    """
    A GPU array of :class:`RGBMaterialPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rgb_material_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RGBMaterialPropertyCommand` elements in the GPU array.
        """
class RaycastCommand:
    """
    A command for raycasting that takes `origins` and `directions` arrays as input, and return the
    results in the arrays `is_hits`, `normals`, and `distances`.  All arrays are two-dimensional, with
    the first dimension spanning all the environments in the environment group, and the second dimension
    spanning the rays per environments.
    
    - `origins`:    4-dimensional vectors where the first three components specify the (x, y, z)
      coordinates of the raycast origins in the environment frame.  The fourth component is currently
      unused.
    - `directions`: 4-dimensional vectors where the first three components specify the (x, y, z)
      coordinates of the raycast directions in the environment frame.  The direction vector must be
      normalized.  The fourth component is the max distance of the raycast.
    - `is_hits`:    boolean array indicating whether the raycast hit an object.
    - `normals`:    if the raycast hit something, the normal vector of the surface that was hit.
    - `distances`:  if the raycast hit something, the distance of the object from the raycast
      origin.
    
    The command optionally takes a :class:`VsQueryGeometryHandle` to raycast selectively against that
    query handle.
    """
    @property
    def directions_data_ptr(self) -> int:
        ...
    @property
    def distances_data_ptr(self) -> int:
        ...
    @property
    def is_hits_data_ptr(self) -> int:
        ...
    @property
    def normals_data_ptr(self) -> int:
        ...
    @property
    def num_rays_per_environment(self) -> int:
        ...
    @property
    def origins_data_ptr(self) -> int:
        ...
    @property
    def query_geometry_handle(self) -> VsQueryGeometryHandle:
        ...
class RaycastCommandGpuArray:
    """
    A GPU array of :class:`RaycastCommand`.
    
    No constructor defined. Use :meth:`Gym.create_raycast_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        ...
class RigidBody(Model):
    """
    A class holding the data of an instantiated rigid body.
    """
    def get_kinematic_sensor_handle(self, index: int, definition_index: int = 0) -> KinematicSensorHandle:
        """
        Gets the :class:`KinematicSensorHandle` of this rigid body instance.  Use ``definition_index``
        for swappable rigid bodies to select the rigid body definition.
        
        .. note::
        
            Only use handles of kinematic sensors from the rigid body definition that is currently active in
            the rigid body instance (set via :meth:`Gym.set_rigid_body_defs()`). Using the handles from an
            inactive definition results in undefined behaviour.
        
        :param index: index of kinematic sensor
        :type index: int
        
        :param definition_index: index of rigid body definition (default 0)
        :type definition_index: int
        
        :return: handle of kinematic sensor
        :rtype: KinematicSensorHandle
        """
    def get_num_definitions(self) -> int:
        """
        Gets the number of rigid body definitions associated with this :class:`RigidBody`.
        """
    def get_rigid_body_def(self, definition_index: int = 0) -> RigidBodyDef:
        """
        Gets the rigid body def.
        """
    def get_rigid_body_def_handle(self, definition_index: int = 0) -> RigidBodyDefHandle:
        """
        Gets the rigid body definition handle.
        
        It can be used to query :meth:`EnvironmentDef.get_rigid_body_def`.
        """
    def get_transform_handle(self) -> TransformHandle:
        """
        Gets the transform handle of the rigid body.
        """
    @property
    def segmentation(self) -> int:
        """
        Segmentation value of the rigid body.
        
        .. note::
            All segmentation values are initialized to ``0``.  Non-hits always evaluate to the fixed value
            ``0``.
        """
    @segmentation.setter
    def segmentation(self, arg1: int) -> None:
        ...
class RigidBodyDef(ModelDef):
    """
    A class that defines the whole structure of a single rigid body.
    """
    def get_kinematic_sensor_def(self, index: int) -> RigidBodyKinematicSensorDef:
        """
        Gets the kinematic sensor def at ``index``.
        
        Fatal error is raised if the ``index`` is out of bounds.
        """
    def get_num_kinematic_sensor_defs(self) -> int:
        """
        Gets the number of kinematic sensor defs contained in the rigid body definition
        """
    @property
    def angular_damping(self) -> float:
        """
        Angular damping coefficient of rigid body.
        """
    @angular_damping.setter
    def angular_damping(self, arg1: float) -> None:
        ...
    @property
    def body_to_model(self) -> Transform:
        """
        Transform from body frame to model frame.
        
        This transform defines the offset to the COM of the body and additionally the rotational frame of
        the diagonalized inverse inertia matrix
        """
    @property
    def diag_inv_inertia(self) -> Vec3:
        """
        Diagonalized inverse inertia of rigid body
        """
    @property
    def inv_mass(self) -> float:
        """
        Inverse mass of rigid body
        
        Setting the inverse mass also scales the diagonalized inverse inertia tensor by the same
        factor.
        """
    @inv_mass.setter
    def inv_mass(self, arg1: float) -> None:
        ...
    @property
    def linear_damping(self) -> float:
        """
        Linear damping coefficient of rigid body.
        """
    @linear_damping.setter
    def linear_damping(self, arg1: float) -> None:
        ...
    @property
    def max_angular_velocity(self) -> float:
        """
        Maxinum angular velocity of rigid body.
        """
    @max_angular_velocity.setter
    def max_angular_velocity(self, arg1: float) -> None:
        ...
    @property
    def max_linear_velocity(self) -> float:
        """
        Maxinum linear velocity of rigid body.
        """
    @max_linear_velocity.setter
    def max_linear_velocity(self, arg1: float) -> None:
        ...
class RigidBodyDefCommand:
    """
    A command to access definition index information of a rigid body. No constructor defined. Use :meth:`EnvironmentGroup.create_rigid_body_def_command` instead to create these.
    
    .. note::
    
        The render of the rigid body is not updated in the standard viewport camera. Please
        tick `Use Tiled RGB renderer` in the GUI menu in order to switch to the tiled RGB renderer,
        which does update the render of the rigid body.
    """
    @property
    def data_ptr(self) -> int:
        """
        Pointer to the data of this command. It will either be used:
        
        - as a source to write data into rigid bodies in setter methods. The required memory size is then equal to ``sizeof(definition index) * num_environments``.
        - as a destination to store read rigid body data in getter methods. The required memory size is then equal to ``sizeof(definition index) * num_environments``.
        
        ``num_environments`` is equal to the total number of environments in the :class:`EnvironmentGroup`.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rigid_body_handle(self) -> RigidBodyHandle:
        """
        :class:`RigidBodyHandle` of the rigid body being accessed.
        """
class RigidBodyDefCommandGpuArray:
    """
    A GPU array of :class:`RigidBodyDefCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rigid_body_def_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RigidBodyDefCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[RigidBodyDefCommand]:
        """
        TODO: document
        """
class RigidBodyDefHandle:
    """
    A class wrapping a handle for a rigid body definition.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: RigidBodyDefHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the rigid body definition.
        """
class RigidBodyExternalForceCommand:
    """
    A command to access external force information of a rigid body. No constructor defined. Use :meth:`EnvironmentGroup.create_rigid_body_external_force_command` instead to create these.
    
    :class:`SpatialVector` objects are written to and
    read from data buffers as ``(top.x, top.y, top.z, bottom.x, bottom.y, bottom.z)``.
    
    In the case of ``force_type=FORCE_TORQUE``, the force is
    stored in the ``top`` part of the :class:`SpatialVector`, and the torque is stored in the ``bottom``
    part of the :class:`SpatialVector`.
    
    In the case of ``force_type=FORCE_POSITION``, the force is stored in the ``top`` part of the
    :class:`SpatialVector`, and the position is stored in the ``bottom`` part of the
    :class:`SpatialVector`.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def force_type(self) -> ForceType:
        """
        Format in which forces are specified
        """
    @property
    def frame_type(self) -> FrameType:
        """
        Frame in which forces are specified
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rigid_body_handle(self) -> RigidBodyHandle:
        """
        :class:`RigidBodyHandle` of the rigid body being accessed.
        """
class RigidBodyExternalForceCommandGpuArray:
    """
    A GPU array of :class:`RigidBodyExternalForceCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rigid_body_external_force_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RigidBodyExternalForceCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[RigidBodyExternalForceCommand]:
        """
        TODO: document
        """
class RigidBodyHandle:
    """
    A class wrapping a handle for a rigid body instance.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: RigidBodyHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the rigid body instance
        """
    def is_valid(self) -> bool:
        """
        Returns ``True`` if the handle is valid and ``False`` otherwise.
        """
class RigidBodyKinematicSensorDef:
    """
    A class holding the data for a kinematic sensor definition on a rigid body.
    """
    @property
    def flags(self) -> list[str]:
        """
        The flags of this kinematic sensor def.
        """
    @property
    def name(self) -> str:
        """
        The name of the kinematic sensor def.
        """
    @property
    def offset(self) -> Vec3:
        """
        The offset of the kinematic sensor def relative to the rigid body COM.
        """
    @offset.setter
    def offset(self, arg1: Vec3) -> None:
        ...
class RigidBodyKinematicStateCommand:
    """
    A command to access kinematic state information of a rigid body. No constructor defined. Use :meth:`EnvironmentGroup.create_rigid_body_kinematic_state_command` instead to create these. This is the same as using:
    
    - :class:`RigidBodyTransformCommand` command to read transforms
    - :class:`RigidBodyVelocityCommand` command to read  velocities
    
    Combining these in one command is much faster and is the preferred method if more than one of the above commands is needed.
    """
    @property
    def frame_type(self) -> FrameType:
        """
        Frame in which velocities are defined
        """
    @property
    def indices_data_ptr(self) -> int:
        ...
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rigid_body_handle(self) -> RigidBodyHandle:
        """
        :class:`RigidBodyHandle` of the rigid body being accessed.
        """
    @property
    def rigid_body_handle_list(self) -> list[RigidBodyHandle]:
        ...
    @property
    def transform_type(self) -> TransformType:
        """
        Type of transform (only for getting transforms)
        """
    @property
    def transforms_data_ptr(self) -> int:
        ...
    @property
    def velocities_data_ptr(self) -> int:
        ...
class RigidBodyKinematicStateCommandGpuArray:
    """
    A GPU array of :class:`RigidBodyKinematicStateCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rigid_body_kinematic_state_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RigidBodyKinematicStateCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[RigidBodyKinematicStateCommand]:
        """
        TODO: document
        """
class RigidBodyProperty:
    """
    Members:
    
      INV_MASS
    """
    INV_MASS: typing.ClassVar[RigidBodyProperty]  # value = <RigidBodyProperty.INV_MASS: 0>
    __members__: typing.ClassVar[dict[str, RigidBodyProperty]]  # value = {'INV_MASS': <RigidBodyProperty.INV_MASS: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RigidBodyPropertyCommand:
    """
    A command to access rigid body property information of a rigid body. No constructor defined. Use :meth:`EnvironmentGroup.create_rigid_body_property_command` instead to create these.
    """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def property(self) -> RigidBodyProperty:
        """
        Selects the property to be accessed.  See :class:`RigidBodyProperty` for options
        """
    @property
    def rigid_body_def_handle(self) -> RigidBodyDefHandle:
        """
        :class:`RigidBodyDefHandle` of the rigid body def being accessed.
        """
class RigidBodyPropertyCommandGpuArray:
    """
    A GPU array of :class:`RigidBodyPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rigid_body_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RigidBodyPropertyCommand` elements in the GPU array.
        """
class RigidBodyTransformCommand:
    """
    A command to access transform information of a rigid body. No constructor defined. Use :meth:`EnvironmentGroup.create_rigid_body_transform_command` instead to create these.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    """
    @property
    def data_ptr(self) -> int:
        """
        Pointer to the data of this command. It will either be used:
        
        - as a source to write data into rigid bodies in setter methods. The required memory size is then equal to ``sizeof(Transform) * num_environments``.
        - as a destination to store read rigid body data in getter methods. The required memory size is then equal to ``sizeof(Transform) * num_environments``.
        
        ``num_environments`` is equal to the total number of environments in the :class:`EnvironmentGroup`.
        """
    @property
    def frame_type(self) -> FrameType:
        """
        Frame of transform (only for getting transforms)
        """
    @property
    def indices_data_ptr(self) -> int:
        ...
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rigid_body_handle(self) -> RigidBodyHandle:
        """
        :class:`RigidBodyHandle` of the rigid body being accessed.
        """
    @property
    def rigid_body_handle_list(self) -> list[RigidBodyHandle]:
        ...
    @property
    def transform_type(self) -> TransformType:
        """
        Type of transform (only for getting transforms)
        """
class RigidBodyTransformCommandGpuArray:
    """
    A GPU array of :class:`RigidBodyTransformCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rigid_body_transform_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RigidBodyTransformCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[RigidBodyTransformCommand]:
        """
        TODO: document
        """
class RigidBodyVelocityCommand:
    """
    A command to access velocity information of a rigid body. No constructor defined. Use :meth:`EnvironmentGroup.create_rigid_body_velocity_command` instead to create these.
    
    :class:`SpatialVector` objects are written to and
    read from data buffers as ``(top.x, top.y, top.z, bottom.x, bottom.y, bottom.z)``.
    
    The angular velocity is stored in the ``top`` part of the :class:`SpatialVector`, and
    the linear velocity is stored in the ``bottom`` part of the :class:`SpatialVector`.
    """
    @property
    def data_ptr(self) -> int:
        """
        Pointer to the data of this command. It will either be used:
        
        - as a source to write data into rigid bodies in setter methods. The required memory size is then equal to ``sizeof(Velocity) * num_environments``.
        - as a destination to store read rigid body data in getter methods. The required memory size is then equal to ``sizeof(Velocity) * num_environments``.
        
        ``num_environments`` is equal to the total number of environments in the :class:`EnvironmentGroup`.
        """
    @property
    def frame_type(self) -> FrameType:
        """
        Frame in which velocities are defined
        """
    @property
    def indices_data_ptr(self) -> int:
        ...
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rigid_body_handle(self) -> RigidBodyHandle:
        """
        :class:`RigidBodyHandle` of the rigid body being accessed.
        """
    @property
    def rigid_body_handle_list(self) -> list[RigidBodyHandle]:
        ...
class RigidBodyVelocityCommandGpuArray:
    """
    A GPU array of :class:`RigidBodyVelocityCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rigid_body_velocity_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RigidBodyVelocityCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[RigidBodyVelocityCommand]:
        """
        TODO: document
        """
class RigidDistanceJointHandle:
    """
    A class wrapping a handle for an RigidDistanceJoint.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: RigidDistanceJointHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the RigidDistanceJoint.
        """
class RigidDistanceJointProperty:
    """
    Members:
    
      ANCHOR1
    
      ANCHOR2
    """
    ANCHOR1: typing.ClassVar[RigidDistanceJointProperty]  # value = <RigidDistanceJointProperty.ANCHOR1: 0>
    ANCHOR2: typing.ClassVar[RigidDistanceJointProperty]  # value = <RigidDistanceJointProperty.ANCHOR2: 1>
    __members__: typing.ClassVar[dict[str, RigidDistanceJointProperty]]  # value = {'ANCHOR1': <RigidDistanceJointProperty.ANCHOR1: 0>, 'ANCHOR2': <RigidDistanceJointProperty.ANCHOR2: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RigidDistanceJointPropertyCommand:
    """
    A command to access rigid distance joint property information of a rigid body. No constructor defined. Use :meth:`EnvironmentGroup.create_rigid_distance_joint_property_command` instead to create these.
    """
    @property
    def data_ptr(self) -> capsule:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def property(self) -> RigidDistanceJointProperty:
        """
        Selects the property to be accessed.  See :class:`RigidDistanceJointProperty` for options
        """
    @property
    def rigid_body_def_handle(self) -> RigidDistanceJointHandle:
        """
        :class:`RigidDistanceJointHandle` of the rigid distance joint being accessed.
        """
class RigidDistanceJointPropertyCommandGpuArray:
    """
    A GPU array of :class:`RigidDistanceJointPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rigid_distance_joint_property_command` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RigidDistanceJointPropertyCommand` elements in the GPU array.
        """
class RigidMaterial:
    def __init__(self) -> None:
        """
        Default constructor
        """
    @property
    def damping(self) -> float:
        """
        Damping value of the contact implicit spring.
        
        This value is only used if any of the two involved rigid materials is using the implicit spring
        model (has a negative restitution).
        """
    @damping.setter
    def damping(self, arg0: float) -> None:
        ...
    @property
    def dynamic_friction(self) -> float:
        """
        Friction value that's used when the contacting bodies are already in motion relative to each other. Typically a value between 0 and 1.
        """
    @dynamic_friction.setter
    def dynamic_friction(self, arg0: float) -> None:
        ...
    @property
    def restitution(self) -> float:
        """
        Restitution coefficient or stiffness value if negative.
        
        - A value of 0 results in collisions that are perfectly inelastic. All kinetic energy is lost during
          contact.
        - A value of 1 results in collisions that are perfectly elastic. No kinetic energy is lost during
          contact and bodies will rebound from each other with the same relative velocity.
        - Values in between interpolate between the two extremes and result in collisions that lose some
          kinetic energy.
        - Negative values are interpreted as a stiffness value for an implicit spring that's used to model a
          soft contact. If any of the two contacting materials is using this soft contact model, the
          collision will be resolved using it.
        """
    @restitution.setter
    def restitution(self, arg0: float) -> None:
        ...
    @property
    def static_friction(self) -> float:
        """
        Friction value that's used when the contacting bodies are stationary relative to each other. Typically a value between 0 and 1.
        """
    @static_friction.setter
    def static_friction(self, arg0: float) -> None:
        ...
class RigidMaterialHandle:
    """
    A class wrapping a handle for an RigidMaterial.
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def invalid() -> RigidMaterialHandle:
        """
        Returns an invalid handle.
        """
    def __eq__(self, arg0: RigidMaterialHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the RigidMaterial.
        """
    def is_valid(self) -> bool:
        """
        Returns whether the handle is valid.
        """
class RigidMaterialProperty:
    """
    Members:
    
      DYNAMIC_FRICTION
    
      STATIC_FRICTION
    
      RESTITUTION
    
      DAMPING
    """
    DAMPING: typing.ClassVar[RigidMaterialProperty]  # value = <RigidMaterialProperty.DAMPING: 3>
    DYNAMIC_FRICTION: typing.ClassVar[RigidMaterialProperty]  # value = <RigidMaterialProperty.DYNAMIC_FRICTION: 0>
    RESTITUTION: typing.ClassVar[RigidMaterialProperty]  # value = <RigidMaterialProperty.RESTITUTION: 2>
    STATIC_FRICTION: typing.ClassVar[RigidMaterialProperty]  # value = <RigidMaterialProperty.STATIC_FRICTION: 1>
    __members__: typing.ClassVar[dict[str, RigidMaterialProperty]]  # value = {'DYNAMIC_FRICTION': <RigidMaterialProperty.DYNAMIC_FRICTION: 0>, 'STATIC_FRICTION': <RigidMaterialProperty.STATIC_FRICTION: 1>, 'RESTITUTION': <RigidMaterialProperty.RESTITUTION: 2>, 'DAMPING': <RigidMaterialProperty.DAMPING: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RigidMaterialPropertyCommand:
    """
    A command to access rigid material property information. No constructor defined. Use :meth:`EnvironmentGroup.create_rigid_material_property_command` instead to create these.
    """
    @property
    def data_ptr(self) -> float:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def property(self) -> RigidMaterialProperty:
        """
        Selects the property to be accessed.  See :class:`RigidMaterialProperty` for options
        """
    @property
    def rigid_material_handle(self) -> RigidMaterialHandle:
        """
        :class:`RigidMaterialHandle` of the rigid material being accessed.
        """
class RigidMaterialPropertyCommandGpuArray:
    """
    A GPU array of :class:`RigidMaterialPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_rigid_material_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`RigidMaterialPropertyCommand` elements in the GPU array.
        """
class SegmentedDepthCameraCommand:
    """
    A command for getting segmented images from depth cameras.
    
    The data can be stored either as a segmentation map, or as an RGB image.
    
    The maximum number of transform handles is 256.
    
    When stored as a segmentation map, the data is stored as unsigned 8-bit integers. The background is
    assigned a segmentation value of ``0``, the first transform handle in the ``transform_handle_list``
    is assigned the segmentation value of ``1``, the second transform handle is assigned ``2``, and so
    on. Use :meth:`EnvironmentGroup.create_segmented_depth_camera_command()` to create the command,
    :meth:`Gym.create_segmented_depth_camera_command_gpu_array()` to create the command array, and
    :meth:`Gym.get_segmented_depth_camera_images()` to execute the command.
    
    When stored as an RGB image, the data is stored as 8-bit integers or 32-bit floating-point values.
    The data type is derived from the ``GpuBufferWrapper`` type passed to
    :meth:`EnvironmentGroup.create_segmented_depth_camera_command()`, which can be
    :class:`Uint8GpuBufferWrapper` or :class:`Float32GpuBufferWrapper`. Use
    :meth:`EnvironmentGroup.create_segmented_depth_camera_to_rgb_command()` to create the command,
    :meth:`Gym.create_segmented_depth_camera_command_gpu_array()` to create the command array, and
    :meth:`Gym.get_segmented_depth_camera_images_in_rgb()` to execute the command.
    
    The ``data`` buffer must have size
     - num_envs * resolution_y * resolution_x * 1 * 1 bytes (uint8)
     - num_envs * resolution_y * resolution_x * 4 * 1 bytes (uint8)
     - num_envs * resolution_y * resolution_x * 4 * 4 bytes (float32)
     
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def depth_camera_handle(self) -> DepthCameraHandle:
        """
        :class:`DepthCameraHandle` of the RGB camera being accessed.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
class SegmentedDepthCameraCommandGpuArray:
    """
    A GPU array of :class:`SegmentedDepthCameraCommand`.
    
    No constructor defined. Use :meth:`Gym.create_segmented_depth_camera_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`SegmentedDepthCameraCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[SegmentedDepthCameraCommand]:
        """
        TODO: document
        """
class SegmentedRGBCameraCommand:
    """
    A command for getting segmented images from RGB cameras.
    
    The data can be stored either as a segmentation map, or as an RGB image.
    
    The maximum number of transform handles is 256.
    
    When stored as a segmentation map, the data is stored as unsigned 8-bit integers. The background is
    assigned a segmentation value of ``0``, the first transform handle in the ``transform_handle_list``
    is assigned the segmentation value of ``1``, the second transform handle is assigned ``2``, and so
    on. Use :meth:`EnvironmentGroup.create_segmented_rgb_camera_command()` to create the command,
    :meth:`Gym.create_segmented_rgb_camera_command_gpu_array()` to create the command array, and
    :meth:`Gym.get_segmented_rgb_camera_images()` to execute the command.
    
    When stored as an RGB image, the data is stored as 8-bit integers or 32-bit floating-point values.
    The data type is derived from the ``GpuBufferWrapper`` type passed to
    :meth:`EnvironmentGroup.create_segmented_rgb_camera_command()`, which can be
    :class:`Uint8GpuBufferWrapper` or :class:`Float32GpuBufferWrapper`. Use
    :meth:`EnvironmentGroup.create_segmented_rgb_camera_to_rgb_command()` to create the command,
    :meth:`Gym.create_segmented_rgb_camera_command_gpu_array()` to create the command array, and
    :meth:`Gym.get_segmented_rgb_camera_images_in_rgb()` to execute the command.
    
    The ``data`` buffer must have size
     - num_envs * resolution_y * resolution_x * 1 * 1 bytes (uint8)
     - num_envs * resolution_y * resolution_x * 4 * 1 bytes (uint8)
     - num_envs * resolution_y * resolution_x * 4 * 4 bytes (float32)
     
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def rgb_camera_handle(self) -> RGBCameraHandle:
        """
        :class:`RGBCameraHandle` of the RGB camera being accessed.
        """
class SegmentedRGBCameraCommandGpuArray:
    """
    A GPU array of :class:`SegmentedRGBCameraCommand`.
    
    No constructor defined. Use :meth:`Gym.create_segmented_rgb_camera_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`SegmentedRGBCameraCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[SegmentedRGBCameraCommand]:
        """
        TODO: document
        """
class SpatialTendonControlCommand:
    """
    A command for setting the control values of spatial tendons.
    
    When used with :meth:`Gym.set_spatial_tendon_controls()` and an elastic material, the control values
    are interpreted as the ratio of the control offset to the rest offset.
    
    When used with :meth:`Gym.set_spatial_tendon_controls()` and a Hill material, the control values are
    interpreted as the activation, which ranges from 0 to 1.  Any values outside of this interval will
    be clamped to this interval.
    
    When used with :meth:`Gym.set_spatial_tendon_forces()`, the control values are interpreted as the
    force applied on the tendon.
    
    No constructor defined. Use :meth:`EnvironmentGroup.create_spatial_tendon_control_command` instead to create these.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def spatial_tendon_control_handle(self) -> ArticulationSpatialTendonHandle:
        """
        :class:`ArticulationSpatialTendonHandle` of the Spatial tendon control being accessed.
        """
class SpatialTendonControlCommandGpuArray:
    """
    A GPU array of :class:`SpatialTendonControlCommand`.
    
    No constructor defined. Use :meth:`Gym.create_spatial_tendon_control_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`SpatialTendonControlCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[SpatialTendonControlCommand]:
        """
        TODO: document
        """
class SpatialTendonDef(TendonDef):
    """
    A class holding the data for a spatial tendon definition.
    """
    def assign_hill_material(self, arg0: HillMaterialHandle) -> None:
        """
        Assign Hill material to spatial tendon.
        
        :param handle: handle of Hill material
        :type handle: HillMaterialHandle
        """
class SpatialTendonProperty:
    """
    Members:
    
      LOW_LIMIT
    
      HIGH_LIMIT
    """
    HIGH_LIMIT: typing.ClassVar[SpatialTendonProperty]  # value = <SpatialTendonProperty.HIGH_LIMIT: 1>
    LOW_LIMIT: typing.ClassVar[SpatialTendonProperty]  # value = <SpatialTendonProperty.LOW_LIMIT: 0>
    __members__: typing.ClassVar[dict[str, SpatialTendonProperty]]  # value = {'LOW_LIMIT': <SpatialTendonProperty.LOW_LIMIT: 0>, 'HIGH_LIMIT': <SpatialTendonProperty.HIGH_LIMIT: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SpatialTendonPropertyCommand:
    """
    TODO: document
    """
    @property
    def data_ptr(self) -> float:
        """
        TODO: document
        """
    @property
    def property(self) -> SpatialTendonProperty:
        """
        TODO: document
        """
    @property
    def spatial_tendon_def_handle(self) -> ...:
        """
        TODO: document
        """
class SpatialTendonPropertyCommandGpuArray:
    """
    A GPU array of :class:`SpatialTendonPropertyCommand`.
    
    No constructor defined. Use :meth:`Gym.create_spatial_tendon_property_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`SpatialTendonPropertyCommand` elements in the GPU array.
        """
class SpatialTendonState:
    """
    Members:
    
      LENGTH
    
      VELOCITY
    
      FORCE
    """
    FORCE: typing.ClassVar[SpatialTendonState]  # value = <SpatialTendonState.FORCE: 2>
    LENGTH: typing.ClassVar[SpatialTendonState]  # value = <SpatialTendonState.LENGTH: 0>
    VELOCITY: typing.ClassVar[SpatialTendonState]  # value = <SpatialTendonState.VELOCITY: 1>
    __members__: typing.ClassVar[dict[str, SpatialTendonState]]  # value = {'LENGTH': <SpatialTendonState.LENGTH: 0>, 'VELOCITY': <SpatialTendonState.VELOCITY: 1>, 'FORCE': <SpatialTendonState.FORCE: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class SpatialTendonStateCommand:
    """
    A command for getting the state of spatial tendons.
    No constructor defined. Use :meth:`EnvironmentGroup.create_spatial_tendon_state_command` instead to create these.
    """
    @property
    def data_ptr(self) -> int:
        """
        Gets the ``data`` cast to a number type.
        """
    @property
    def masks_data_ptr(self) -> int:
        ...
    @property
    def spatial_tendon_control_handle(self) -> ArticulationSpatialTendonHandle:
        """
        :class:`ArticulationSpatialTendonHandle` of the Spatial tendon state being accessed.
        """
    @property
    def state(self) -> SpatialTendonState:
        """
        TODO: document
        """
class SpatialTendonStateCommandGpuArray:
    """
    A GPU array of :class:`SpatialTendonStateCommandGpuArray`.
    
    No constructor defined. Use :meth:`Gym.create_spatial_tendon_state_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of :class:`SpatialTendonStateCommand` elements in the GPU array.
        """
    def get_commands(self) -> list[SpatialTendonStateCommand]:
        """
        TODO: document
        """
class SpatialVector:
    """
    A vector holding two 3D vectors with the top one usually representing a linear part and the
    bottom one representing an angular part.  E.g. a linear velocity and an angular velocity.
    
    :class:`SpatialVector` objects are written to and
    read from data buffers as ``(top.x, top.y, top.z, bottom.x, bottom.y, bottom.z)``.
    """
    def __abs__(self) -> float:
        """
        Returns the magnitude sum of the :attr:`top` and :attr:`bottom` vectors.
        """
    def __add__(self, other: SpatialVector) -> SpatialVector:
        """
        Performs componentwise addition of the two operands.
        """
    def __iadd__(self, other: SpatialVector) -> None:
        """
        Performs componentwise addition of the two operands in place.
        """
    @typing.overload
    def __init__(self, top: Vec3, bottom: Vec3) -> None:
        """
        Constructs a spatial vector with from the provided ``top`` and ``bottom`` 3D vectors.
        """
    @typing.overload
    def __init__(self, value: float) -> None:
        """
        Constructs a spatial vector filled with the provided ``value``.
        """
    @typing.overload
    def __init__(self, value: Vec3) -> None:
        """
        Constructs a spatial vector with top and bottom components equal to ``value``.
        """
    def __isub__(self, other: SpatialVector) -> None:
        """
        Performs componentwise subtraction of the two operands in place.
        """
    def __sub__(self, other: SpatialVector) -> SpatialVector:
        """
        Performs componentwise subtraction of the two operands.
        """
    def abs(self) -> SpatialVector:
        """
        Returns a spatial vector with absolute component values of this spatial vector.
        """
    def dot(self, other: SpatialVector) -> float:
        """
        Returns the dot product between this spatial vector and the provided one. The dot product is equal to self.top · other.top + self.bottom · other.bottom.
        """
    def transform(self, arg0: ...) -> SpatialVector:
        """
        Transforms spatial vector using the given transform.
        """
    @property
    def bottom(self) -> Vec3:
        """
        The bottom (or angular) part of this spatial vector.
        """
    @bottom.setter
    def bottom(self, arg0: Vec3) -> None:
        ...
    @property
    def top(self) -> Vec3:
        """
        The top (or linear) part of this spatial vector.
        """
    @top.setter
    def top(self, arg0: Vec3) -> None:
        ...
class TendonDef:
    """
    A class holding the data for a tendon definition.
    """
    def assign_elastic_material(self, arg0: ElasticMaterialHandle) -> None:
        """
        Assign elastic material to tendon.
        
        :param handle: handle of elastic material
        :type handle: ElasticMaterialHandle
        """
    @property
    def high_limit(self) -> float:
        """
        High limit of tendon.
        """
    @high_limit.setter
    def high_limit(self, arg1: float) -> None:
        ...
    @property
    def low_limit(self) -> float:
        """
        Low limit of tendon.
        """
    @low_limit.setter
    def low_limit(self, arg1: float) -> None:
        ...
    @property
    def name(self) -> str:
        """
        The name of the tendon.
        """
    @property
    def rest_offset(self) -> float:
        """
        Rest offset of tendon.
        """
    @rest_offset.setter
    def rest_offset(self, arg1: float) -> None:
        ...
class TextureHandle:
    """
    A class wrapping a handle for a loaded texture.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: TextureHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the texture.
        """
class Transform:
    """
    A transform composed of a rotation, expressed by a :class:`Quat`, and a translation, expressed
    by a :class:`Vec3`.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: Transform) -> bool:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, translation: Vec3) -> None:
        """
        Constructs a :class:`Transform` from a :class:`Vec3` translation/position with identity
        rotation
        """
    @typing.overload
    def __init__(self, rotation: Quat, translation: Vec3) -> None:
        """
        Constructs a :class:`Transform` from a :class:`Quat` rotation and a :class:`Vec3`
        translation/position
        """
    @typing.overload
    def __init__(self, other: Transform) -> None:
        """
        Copy constructor
        """
    def get_inverse(self) -> Transform:
        """
        Returns the inverse transform of this :class:`Transform`.
        """
    def get_normalized(self) -> Transform:
        """
        Returns a normalized version of this :class:`Transform`, where the :class:`Quat` has a
        magnitude of 1.
        """
    def rotate(self, vector: Vec3) -> Vec3:
        """
        Rotates the supplied vector into the world space of this :class:`Transform`
        """
    def rotate_inv(self, vector: Vec3) -> Vec3:
        """
        Rotates the supplied vector into the local space of this :class:`Transform`
        """
    @typing.overload
    def transform(self, local_transform: Transform) -> Transform:
        """
        Transforms the supplied :class:`Transform` value into the world space of this :class:`Transform`
        """
    @typing.overload
    def transform(self, local_point: Vec3) -> Vec3:
        """
        Transforms the supplied point into the world sapce of this :class:`Transform`
        """
    @typing.overload
    def transform_inv(self, world_transform: Transform) -> Transform:
        """
        Transforms the supplied :class:`Transform` value into the local space of this :class:`Transform`
        """
    @typing.overload
    def transform_inv(self, world_point: Vec3) -> Vec3:
        """
        Transforms the supplied point into the world space of this :class:`Transform`
        """
    @property
    def p(self) -> Vec3:
        """
        Position of the :class:`Transform`, expressed as a :class:`Vec3`
        """
    @p.setter
    def p(self, arg0: Vec3) -> None:
        ...
    @property
    def q(self) -> Quat:
        """
        Rotation of the :class:`Transform`, expressed as a :class:`Quat`
        """
    @q.setter
    def q(self, arg0: Quat) -> None:
        ...
class TransformCommand:
    """
    A command to access transform information. No constructor defined. Use :meth:`EnvironmentGroup.create_transform_command` instead to create these.
    
    :class:`Transform` objects are written to and read from
    data buffers as ``(q.x, q.y, q.z, q.w, p.x, p.y, p.z)``.
    """
    @property
    def data_ptr(self) -> int:
        ...
    @property
    def transform_handle(self) -> TransformHandle:
        """
        :class:`TransformHandle` of the transform being accessed.
        """
class TransformCommandGpuArray:
    """
    A GPU array of :class:`TransformCommand`.
    
    No constructor defined. Use :meth:`Gym.create_transform_command_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        ...
    def get_commands(self) -> list[TransformCommand]:
        """
        TODO: document
        """
class TransformHandle:
    """
    A class wrapping a handle for a transform.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: TransformHandle) -> bool:
        """
        Returns ``True`` if the two handles are equal and ``False`` otherwise.
        """
    def env_def_index(self) -> int:
        """
        Gets the underlying index of the :class:`EnvironmentDef` that this :class:`TransformHandle`
        belongs to.
        """
    def index(self) -> int:
        """
        Gets the underlying index of the Transform.
        """
    def type(self) -> int:
        """
        Gets the type of the transform.
        
        * ``0``: transform belongs to global environment
        * ``1``: transform does not belong to global environment
        """
class TransformHandleGpuBufferWrapper:
    """
    Wrapper class for user-provided GPU buffer of TransformHandle bools
    """
    def __init__(self, data_ptr: int, size: int) -> None:
        """
        Wraps a GPU buffer using its data pointer and size in bytes
        """
    @property
    def data_ptr(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
class TransformType:
    """
    The type of transform to return.
    
    Specify the frame in which to represent transforms using :class:`FrameType`.  Valid frames are
    `WORLD` and `ENVIRONMENT`.
    
    
    Members:
    
      MODEL : Return the model transform.
    
      COM : Return the center of mass transform.
    """
    COM: typing.ClassVar[TransformType]  # value = <TransformType.COM: 1>
    MODEL: typing.ClassVar[TransformType]  # value = <TransformType.MODEL: 0>
    __members__: typing.ClassVar[dict[str, TransformType]]  # value = {'MODEL': <TransformType.MODEL: 0>, 'COM': <TransformType.COM: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Uint16GpuArray:
    """
    A GPU array of `uint16`.
    
    No constructor defined. Use :meth:`Gym.create_uint16_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of `uint16` elements in the GPU array.
        """
    def data_ptr(self, offset: int = 0) -> int:
        """
        Gets the underlying GPU pointer cast to an integer type big enough to hold the pointer.
        
        You can specify an offset in the array using the optional ``offset`` argument.
        """
    def get(self) -> list[int]:
        """
        Returns the GPU array data as a vector of `uint16` elements on the CPU.
        """
    def set(self, host_array: list[int]) -> None:
        """
        Fills the GPU array with data from the supplied ``host_array`` vector of `uint16` elements on
        the CPU.
        """
class Uint16GpuBufferWrapper:
    """
    Wrapper class for user-provided GPU buffer of 16-bit unsigned integers
    """
    def __init__(self, data_ptr: int, size: int) -> None:
        """
        Wraps a GPU buffer using its data pointer and size in bytes
        """
    @property
    def data_ptr(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
class Uint32GpuArray:
    """
    A GPU array of `uint32`.
    
    No constructor defined. Use :meth:`Gym.create_uint32_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of `uint32` elements in the GPU array.
        """
    def data_ptr(self, offset: int = 0) -> int:
        """
        Gets the underlying GPU pointer cast to an integer type big enough to hold the pointer.
        
        You can specify an offset in the array using the optional ``offset`` argument.
        """
    def get(self) -> list[int]:
        """
        Returns the GPU array data as a vector of `uint32` elements on the CPU.
        """
    def set(self, host_array: list[int]) -> None:
        """
        Fills the GPU array with data from the supplied ``host_array`` vector of `uint32` elements on
        the CPU.
        """
class Uint32GpuBufferWrapper:
    """
    Wrapper class for user-provided GPU buffer of 32-bit unsigned integers
    """
    def __init__(self, data_ptr: int, size: int) -> None:
        """
        Wraps a GPU buffer using its data pointer and size in bytes
        """
    @property
    def data_ptr(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
class Uint8GpuArray:
    """
    A GPU array of `uint8`.
    
    No constructor defined. Use :meth:`Gym.create_uint8_gpu_array` instead to create these.
    """
    def __len__(self) -> int:
        """
        Gets the number of `uint8` elements in the GPU array.
        """
    def data_ptr(self, offset: int = 0) -> int:
        """
        Gets the underlying GPU pointer cast to an integer type big enough to hold the pointer.
        
        You can specify an offset in the array using the optional ``offset`` argument.
        """
    def get(self) -> list[int]:
        """
        Returns the GPU array data as a vector of `uint8` elements on the CPU.
        """
    def set(self, host_array: list[int]) -> None:
        """
        Fills the GPU array with data from the supplied ``host_array`` vector of `uint8` elements on
        the CPU.
        """
class Uint8GpuBufferWrapper:
    """
    Wrapper class for user-provided GPU buffer of 8-bit unsigned integers
    """
    def __init__(self, data_ptr: int, size: int) -> None:
        """
        Wraps a GPU buffer using its data pointer and size in bytes
        """
    @property
    def data_ptr(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
class UserCheckbox(UserMenuItem):
    def __init__(self, name: str, value: bool) -> None:
        """
        Constructs a :class:`UserCheckbox` with a ``name`` and a ``value``.
        """
    def get_name(self) -> str:
        """
        Gets the name of the checkbox.
        """
    def get_value(self) -> bool:
        """
        Gets the value of the checkbox.
        """
    def set_value(self, value: bool) -> None:
        """
        Sets the value of the checkbox.
        """
class UserCombo(UserMenuItem):
    """
    A combo dropdown menu.
    """
    def __init__(self, name: str, item_values: list[str], current_index: int) -> None:
        """
        Constructs a combo menu with the provided ``name``, ``item_values``, and ``current_index``
        """
    def get_current_index(self) -> int:
        """
        Gets the currently selected item index.
        """
    def get_current_item(self) -> str:
        """
        Gets the currently selected item name.
        """
    def get_name(self) -> str:
        """
        Gets the name of this combo menu.
        """
    def get_num_items(self) -> int:
        """
        Gets the total number of items in the dropdown.
        """
    def set_current_index(self, index: int) -> None:
        """
        Sets the current selection index to ``index``.
        """
class UserLine(UserLineShape):
    """
    A line that does not interact with environments.
    """
    def get_color(self) -> Vec3:
        """
        "Gets the color of this line.
        """
    def get_points(self) -> list[Vec3]:
        """
        "Gets the points of this line.
        """
    def set_color(self, color: Vec3) -> None:
        """
        "Sets the color of this line to ``color``.
        """
    def set_points(self, points: list[Vec3]) -> None:
        """
        "Sets the points of this line to ``points``.
        """
class UserLineCube(UserLineShape):
    """
    A cube drawn using lines that does not interact with environments.
    """
    def get_color(self) -> Vec3:
        """
        Gets the color of this line cube.
        """
    def get_size(self) -> float:
        """
        Gets the size of this line cube.
        """
    def get_transform(self) -> Transform:
        """
        Gets the transform of this line cube.
        """
    def set_color(self, color: Vec3) -> None:
        """
        Sets the color of this line cube to ``color``.
        """
    def set_size(self, size: float) -> None:
        """
        Sets the size of this line cube to ``size``.
        """
    def set_transform(self, transform: Transform) -> None:
        """
        Sets the transform of this line cube to ``transform``.
        """
class UserLineShape:
    """
    Custom line shape for the Vlearn render.
    
    Register using :meth:`GymRender.register_line_shape()`.
    
    Unregister using :meth:`GymRender.unregister_line_shape()`.
    """
class UserMenuItem:
    """
    Custom menu item for the Vlearn GUI menu.
    
    Register using :meth:`GymRender.register_menu_item()`.
    
    Unregister using :meth:`GymRender.unregister_menu_item()`.
    """
class UserSlider(UserMenuItem):
    def __init__(self, name: str, low: float, high: float, value: float) -> None:
        """
        Constructs a :class:`UserSlider` with a name, minimum, maximum, and initial values.
        """
    def get_high(self) -> float:
        """
        Gets the upper limit of the slider
        """
    def get_low(self) -> float:
        """
        Gets the low limit of the slider
        """
    def get_name(self) -> str:
        """
        Gets the name of the slider.
        """
    def get_value(self) -> float:
        """
        Gets the current value of the slider.
        """
    def set_value(self, value: float) -> None:
        """
        Sets the current value of the slider.
        """
class Vec3:
    """
    Represents a vector in 3D space.
    """
    __hash__: typing.ClassVar[None] = None
    def __abs__(self) -> float:
        """
        Returns the magnitude of this vector.
        """
    def __add__(self, arg0: Vec3) -> Vec3:
        """
        Adds the two vectors componentwise
        """
    def __eq__(self, arg0: Vec3) -> bool:
        """
        Performs a bitwise equality check on the vectors.
        """
    def __iadd__(self, arg0: Vec3) -> Vec3:
        """
        Adds the two vectors componentwise in place
        """
    def __imul__(self, arg0: Vec3) -> Vec3:
        """
        Multiplies the two vectors componentwise in place.
        """
    @typing.overload
    def __init__(self, x: float, y: float, z: float) -> None:
        """
        Constructs a vector from the provided x, y, and z values.
        """
    @typing.overload
    def __init__(self, value: float) -> None:
        """
        Constructs a vector filled with the provided value.
        """
    def __isub__(self, arg0: Vec3) -> Vec3:
        """
        Subtracts the two vectors componentwise in place
        """
    @typing.overload
    def __mul__(self, arg0: Vec3) -> Vec3:
        """
        Multiplies the two vectors componentwise.
        """
    @typing.overload
    def __mul__(self, arg0: float) -> Vec3:
        """
        Scales this vector by a scalar value.
        """
    def __neg__(self) -> Vec3:
        """
        Multiply this vector by -1.
        """
    def __rmul__(self, arg0: float) -> Vec3:
        """
        Scales this vector by a scalar value.
        """
    def __sub__(self, arg0: Vec3) -> Vec3:
        """
        Subtracts the two vectors componentwise
        """
    def __truediv__(self, arg0: float) -> Vec3:
        """
        Divides this vector by a scalar value.
        """
    def abs(self) -> Vec3:
        """
        Returns a vector with absolute component values of this vector.
        """
    def cross(self, other: Vec3) -> Vec3:
        """
        Performs the cross product operation between this and the supplied vector.
        """
    def dot(self, other: Vec3) -> float:
        """
        Performs the dot product operation between this and the supplied vector.
        """
    def get_normalized(self) -> Vec3:
        """
        Get a normalized copy of this vector. If the magnitude of this vector is 0, this returns a
        null vector.
        """
    def normalize(self) -> float:
        """
        Normalizes this vector in place. If the magnitude of this vector is 0, division by 0 will
        occur.
        """
    @property
    def x(self) -> float:
        """
        The *X* component of the 3D vector
        """
    @x.setter
    def x(self, arg0: float) -> None:
        ...
    @property
    def y(self) -> float:
        """
        The *Y* component of the 3D vector
        """
    @y.setter
    def y(self, arg0: float) -> None:
        ...
    @property
    def z(self) -> float:
        """
        The *Z* component of the 3D vector
        """
    @z.setter
    def z(self, arg0: float) -> None:
        ...
class VsMaterialHandle:
    def __init__(self) -> None:
        ...
    def index(self) -> int:
        """
        The index of this material handle
        """
    def type(self) -> int:
        """
        The type index of this material handle
        """
class VsQueryGeometryHandle:
    """
    Vsim handle for query geometry.
    """
    def index(self) -> int:
        ...
    def is_valid(self) -> bool:
        ...
class VsRigidBodyHandle:
    """
    A handle identifying a rigid body
    """
class VsRigidBodySetHandle:
    """
    A handle identifying a set of rigid bodies
    """
    def __init__(self) -> None:
        """
        Default constructor
        """
    def u16(self) -> int:
        """
        Converts the handle to its underlying unsigned 16 bit representation
        """
class VsTransformHandle:
    """
    Vsim handle for transform.
    """
    def index(self) -> int:
        ...
    def is_valid(self) -> bool:
        ...
def axis_angle_from_quat(quat: Quat) -> Vec3:
    """
    :param quat: rotation as quaternion
    :type quat: Quat
    
    :return: rotation as axis angle unit vector multiplied by angle
    :rtype: Vec3
    """
def matrix_from_quat(quat: Quat) -> list[list[float]]:
    """
    :param quat: rotation as quaternion
    :type quat: Quat
    
    :return: rotation as rotation matrix
    :rtype: List[List[VsReal]]
    """
def quat_from_rpy(rpy: Vec3) -> Quat:
    """
    :param rpy: rotation as roll, pitch, and yaw
    :type rpy: Vec3
    
    :return: rotation as quaternion
    :rtype: Quat
    """
def rotate_inertia_matrix(mass: float, diag_inertia: Vec3, transform: Quat) -> list[list[float]]:
    """
    Rotates a principal inertia matrix and returns a spatial inertia matrix
    
    The spatial inertia matrix follows the angular-linear ordering.
    
    :param mass: mass of rigid body
    :type mass: VsReal
    :param diag_inertia: diagonal entries of principal inertia matrix
    :type diag_inertia: Vec3
    :param rotation: rotation
    :type transform: Transform
    
    :return: rotated spatial inertia matrix in row-major format
    :rtype: List[List[VsReal]]
    """
def rpy_from_quat(quat: Quat) -> Vec3:
    """
    :param quat: rotation as quaternion
    :type quat: Quat
    
    :return: rotation as roll, pitch, and yaw
    :rtype: Vec3
    """
def shortest_rotation(vector0: Vec3, vector1: Vec3) -> Quat:
    """
    :param vector0:
    :type vector0: Vec3
    
    :param vector1:
    :type vector1: Vec3
    
    :return: shortest rotation from ``vector0`` to ``vector``
    :rtype: Quat
    """
A: HillMaterialProperty  # value = <HillMaterialProperty.A: 1>
ANCHOR1: RigidDistanceJointProperty  # value = <RigidDistanceJointProperty.ANCHOR1: 0>
ANCHOR2: RigidDistanceJointProperty  # value = <RigidDistanceJointProperty.ANCHOR2: 1>
ANTI_ALIASING: CameraPostFilter  # value = <CameraPostFilter.ANTI_ALIASING: 0>
ARMATURE: JointDofProperty  # value = <JointDofProperty.ARMATURE: 0>
ARTICULATION: KinematicSensorType  # value = <KinematicSensorType.ARTICULATION: 0>
B: HillMaterialProperty  # value = <HillMaterialProperty.B: 2>
CHILD_FRAME: JointProperty  # value = <JointProperty.CHILD_FRAME: 3>
COLOR: RGBMaterialProperty  # value = <RGBMaterialProperty.COLOR: 0>
COM: TransformType  # value = <TransformType.COM: 1>
DAMPING: PIDProperty  # value = <PIDProperty.DAMPING: 1>
DYNAMIC_FRICTION: RigidMaterialProperty  # value = <RigidMaterialProperty.DYNAMIC_FRICTION: 0>
ENVIRONMENT: BroadPhaseType  # value = <BroadPhaseType.ENVIRONMENT: 0>
FORCE: SpatialTendonState  # value = <SpatialTendonState.FORCE: 2>
FORCE_POSITION: ForceType  # value = <ForceType.FORCE_POSITION: 1>
FORCE_TORQUE: ForceType  # value = <ForceType.FORCE_TORQUE: 0>
FRICTION: DeformableMaterialProperty  # value = <DeformableMaterialProperty.FRICTION: 3>
GEAR_RATIO: MotorProperty  # value = <MotorProperty.GEAR_RATIO: 0>
GLOBAL_ENV_DEF_HANDLE: EnvironmentDefHandle  # value = <vlearn_bindings.EnvironmentDefHandle object>
GLOBAL_ENV_HANDLE: EnvironmentHandle  # value = <vlearn_bindings.EnvironmentHandle object>
HIGH_LIMIT: SpatialTendonProperty  # value = <SpatialTendonProperty.HIGH_LIMIT: 1>
INV_MASS: RigidBodyProperty  # value = <RigidBodyProperty.INV_MASS: 0>
JOINT_FRICTION: JointProperty  # value = <JointProperty.JOINT_FRICTION: 0>
LENGTH: SpatialTendonState  # value = <SpatialTendonState.LENGTH: 0>
LIMIT_DAMPING: ElasticMaterialProperty  # value = <ElasticMaterialProperty.LIMIT_DAMPING: 3>
LIMIT_STIFFNESS: ElasticMaterialProperty  # value = <ElasticMaterialProperty.LIMIT_STIFFNESS: 2>
LOCAL: FrameType  # value = <FrameType.LOCAL: 2>
LOW_LIMIT: SpatialTendonProperty  # value = <SpatialTendonProperty.LOW_LIMIT: 0>
LSE0: HillMaterialProperty  # value = <HillMaterialProperty.LSE0: 0>
LUMINESCENE: RGBMaterialProperty  # value = <RGBMaterialProperty.LUMINESCENE: 1>
MASS: LinkProperty  # value = <LinkProperty.MASS: 0>
MAX_FORCE: PIDProperty  # value = <PIDProperty.MAX_FORCE: 2>
MAX_JOINT_VELOCITY: JointProperty  # value = <JointProperty.MAX_JOINT_VELOCITY: 2>
MODEL: TransformType  # value = <TransformType.MODEL: 0>
MOTOR: ArticulationControlType  # value = <ArticulationControlType.MOTOR: 1>
NONE: CameraPostFilter  # value = <CameraPostFilter.NONE: 1>
PINHOLE: CameraType  # value = <CameraType.PINHOLE: 0>
POISSON_RATIO: DeformableMaterialProperty  # value = <DeformableMaterialProperty.POISSON_RATIO: 1>
RADIAL: CameraType  # value = <CameraType.RADIAL: 1>
RESTITUTION: RigidMaterialProperty  # value = <RigidMaterialProperty.RESTITUTION: 2>
RIGID_BODY: KinematicSensorType  # value = <KinematicSensorType.RIGID_BODY: 1>
SCENE: BroadPhaseType  # value = <BroadPhaseType.SCENE: 1>
SPECULAR_EXPONENT: RGBMaterialProperty  # value = <RGBMaterialProperty.SPECULAR_EXPONENT: 3>
SPECULAR_INTENSITY: RGBMaterialProperty  # value = <RGBMaterialProperty.SPECULAR_INTENSITY: 4>
STATIC_FRICTION: RigidMaterialProperty  # value = <RigidMaterialProperty.STATIC_FRICTION: 1>
STIFFNESS: PIDProperty  # value = <PIDProperty.STIFFNESS: 0>
TEXTURE: RGBCameraSkybox  # value = <RGBCameraSkybox.TEXTURE: 0>
USE_COLLISIONS: QueryMode  # value = <QueryMode.USE_COLLISIONS: 2>
USE_FILE: QueryMode  # value = <QueryMode.USE_FILE: 0>
USE_NONE: QueryMode  # value = <QueryMode.USE_NONE: 3>
USE_VISUALS: QueryMode  # value = <QueryMode.USE_VISUALS: 1>
VELOCITY: SpatialTendonState  # value = <SpatialTendonState.VELOCITY: 1>
WORLD: FrameType  # value = <FrameType.WORLD: 0>
YOUNGS_MODULUS: DeformableMaterialProperty  # value = <DeformableMaterialProperty.YOUNGS_MODULUS: 0>

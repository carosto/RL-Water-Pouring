import os

import pysplishsplash as sph
import pyvista
import vtk

import pysplishsplash.Utilities.SceneLoaderStructs as Scenes

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)  # TODO check vtkMath::Jacobi warning for collision check

from scipy.spatial.transform import Rotation as R

import numpy as np


class Simulation:
    def __init__(self, use_gui, output_directory, jug_start_position, cup_position, scene_file):
        self.use_gui = use_gui
        self.output_directory = output_directory
        self.jug_start_position = jug_start_position
        self.cup_position = cup_position
        self.scene_file = scene_file

        self.particles_ids_cup = []
        self.particles_ids_jug = []
        self.collector_dict = []

        self.cup_name = "wine_glass_test"
        self.cup_path = os.path.join(os.path.dirname(__file__), f"ObjectFiles/{self.cup_name}.obj")
        self.jug_name = "beakernew"
        self.jug_path = os.path.join(os.path.dirname(__file__), f"ObjectFiles/{self.jug_name}.obj")

        # calculate the bounds of the cup (it does not move, so this only has to be done once)
        self.cup_obj = pyvista.read(self.cup_path)  # f"../ObjectFiles/{self.cup_name}.obj")
        cup_position = self.cup_position[:3]
        cup_rotation = self.cup_position[3:]

        transformed_cup = self.cup_obj.transform(
            self.__transform_matrix_func(cup_position, cup_rotation), inplace=False
        )
        self.cup_bounds = transformed_cup.bounds

        self.jug_obj = pyvista.read(
            self.jug_path
        )  # f"water_pouring/water_pouring/envs/ObjectFiles/{self.jug_name}.obj")
        self.spilled_collector_grid = pyvista.read(
            os.path.join(os.path.dirname(__file__), "ObjectFiles/UnitBox_open.obj")
        )  # "water_pouring/water_pouring/envs/ObjectFiles/UnitBox_open.obj")

        self.n_particles_cup = 0
        self.n_particles_jug = 0
        self.n_particles_spilled = 0
        self.n_particles_pouring = 0

        # self.collision = False

        self.is_initialized = False

    def init_simulation(self):
        base = sph.Exec.SimulatorBase()
        base.init(
            useGui=self.use_gui,
            sceneFile=os.path.abspath(
                os.path.join(os.path.dirname(__file__), self.scene_file)
            ),  #'water_pouring/water_pouring/envs/scene.json'),
            stopAt=5,
            outputDir=self.output_directory,
        )

        base.setTimeStepCB(self.__time_step_callback)

        # Get the scene and add objects
        scene = sph.Exec.SceneConfiguration.getCurrent().getScene()

        # Add the jug
        jug_file = f"{self.jug_name}.obj"
        rots = self.jug_start_position[3:]
        vec = rots[:3]
        s = rots[3]
        # formula to transform quaternions into axis angle format: see wikipedia
        theta = 2 * np.arctan2(np.linalg.norm(vec), s)
        w = (vec / np.sin(theta / 2)) if theta != 0 else 0
        scene.boundaryModels.append(
            Scenes.BoundaryData(
                meshFile=self.jug_path,  # f'ObjectFiles/{jug_file}',
                translation=self.jug_start_position[:3],
                scale=[1, 1, 1],
                color=[0.5, 0.5, 0.5, 1.0],
                isAnimated=True,
                axis=w,
                angle=theta,
                isWall=False,
                mapInvert=False,
                mapResolution=[25, 25, 25],
            )
        )

        # Add the cup
        cup_file = f"{self.cup_name}.obj"
        rots = self.cup_position[3:]
        vec = rots[:3]
        s = rots[3]
        # formula to transform quaternions into axis angle format: see wikipedia
        theta = 2 * np.arctan2(np.linalg.norm(vec), s)
        w = (vec / np.sin(theta / 2)) if theta != 0 else 0
        scene.boundaryModels.append(
            Scenes.BoundaryData(
                meshFile=self.cup_path,  # f'ObjectFiles/{cup_file}',
                translation=self.cup_position[:3],
                scale=[1, 1, 1],
                color=[0.5, 0.5, 0.5, 1.0],
                isAnimated=True,
                axis=w,
                angle=theta,
                isWall=False,
                mapInvert=False,
                mapResolution=[25, 25, 25],
            )
        )

        # scene.camPosition = [1, 2, 2.]

        fluid_init_pos = np.array(self.jug_start_position[:3])
        """fluid_init_pos_upper = fluid_init_pos + [0.07, 0.175, 0.07]

        scene.fluidBlocks.append(
                Scenes.FluidBlock(
                    id='Fluid',
                    boxMin=fluid_init_pos,
                    boxMax=fluid_init_pos_upper,
                    mode=0,
                    initialVelocity=[0.0, 0.0, 0.0],
                )
            )"""

        base.initSimulation()

        sim = sph.Simulation.getCurrent()
        # sim.setValueInt(sim.BOUNDARY_HANDLING_METHOD, 0)

        # base.readFluidParticlesState('water_pouring/water_pouring/envs/LiquidParticles/beakernew_fluid_block_p006_v5.bgeo', sim.getFluidModel(0))

        # move particles to the start position of the jug
        fluid_model = sim.getFluidModel(0)

        self._number_particles = fluid_model.getNumActiveParticles0()

        self.time_step_size = sph.TimeManager.getCurrent().getTimeStepSize()

        for i in range(fluid_model.getNumActiveParticles0()):
            curr_pos = fluid_model.getPosition(i)
            fluid_model.setPosition(i, curr_pos + fluid_init_pos)
            fluid_model.setVelocity(i, [0.0, 0.0, 0.0])
            fluid_model.setAcceleration(i, [0.0, 0.0, 0.0])

        self.base = base
        self.base.finishInitialization()
        self.is_initialized = True

    def cleanup(self):
        self.base.cleanup()
        self.base = None
        self.is_initialized = False

        self.n_particles_cup = 0
        self.n_particles_jug = 0
        self.n_particles_spilled = 0
        self.n_particles_pouring = 0

        # self.collision = False

        self.particles_ids_cup = []
        self.particles_ids_jug = []
        self.collector_dict = []

    def __time_step_callback(self):
        sim = sph.Simulation.getCurrent()

        position_change = self.next_position[0]
        rotation_change = self.next_position[1]

        # update the position of the jug
        boundary = sim.getBoundaryModel(0)  # first object is jug
        animated_body = boundary.getRigidBodyObject()
        new_position = animated_body.getPosition() + position_change

        old_rotation = R.from_quat(animated_body.getRotation()).as_euler("XYZ", degrees=True)
        new_rotation = R.from_euler("XYZ", old_rotation + rotation_change, degrees=True)

        collision = self.__check_collision(
            self.cup_position[:3], self.cup_position[3:], new_position, new_rotation.as_quat()
        )

        # move the jug only if it does not collide with the cup
        if not collision:
            animated_body.setPosition(new_position)
            animated_body.setRotation(new_rotation.as_quat())
            animated_body.animate()
        else:  # keep the jug at the previous position if it would collide otherwise
            animated_body.setPosition(animated_body.getPosition())
            animated_body.setRotation(R.from_euler("XYZ", old_rotation, degrees=True).as_quat())
            animated_body.animate()
        # keep the cup at a constant position
        # TODO ist das nötig? (ansonsten macht das return oben schwierigkeiten!)
        boundary = sim.getBoundaryModel(1)
        animated_body = boundary.getRigidBodyObject()  # second object is cup
        animated_body.setPosition(self.cup_position[:3])
        animated_body.setRotation(self.cup_position[3:])
        animated_body.animate()

        self.__count_particles()

        # keep the simulation running until the process is stopped
        self.base.setValueFloat(self.base.STOP_AT, sph.TimeManager.getCurrent().getTime() + 5)

    def __get_bounds(self, bounds):
        x_bounds = bounds[:2]  # xmin, xmax
        y_bounds = bounds[2:4]  # ymin, ymax
        z_bounds = bounds[4:]  # zmin, zmax
        return x_bounds, y_bounds, z_bounds

    def __get_liquid_inside_obj(self, fluid_model, cup_bounds, jug_bounds):
        ids_inside_cup = []
        ids_inside_jug = []
        ids_spilled = []
        cup_x_bounds, cup_y_bounds, cup_z_bounds = self.__get_bounds(cup_bounds)
        jug_x_bounds, jug_y_bounds, jug_z_bounds = self.__get_bounds(jug_bounds)
        for i in range(fluid_model.getNumActiveParticles0()):
            p = fluid_model.getPosition(i)
            if p[0] > cup_x_bounds[0] and p[0] < cup_x_bounds[1]:
                if p[1] > cup_y_bounds[0] and p[1] < cup_y_bounds[1]:
                    if p[2] > cup_z_bounds[0] and p[2] < cup_z_bounds[1]:
                        ids_inside_cup.append(i)
            if p[0] > jug_x_bounds[0] and p[0] < jug_x_bounds[1]:
                if p[1] > jug_y_bounds[0] and p[1] < jug_y_bounds[1]:
                    if p[2] > jug_z_bounds[0] and p[2] < jug_z_bounds[1]:
                        ids_inside_jug.append(i)
            if (
                p[1] < cup_y_bounds[1] and i not in ids_inside_cup and i not in ids_inside_jug
            ):  # particles is spilled if it is below the cup
                ids_spilled.append(i)

        return len(ids_inside_cup), ids_inside_cup, len(ids_inside_jug), ids_inside_jug, len(ids_spilled), ids_spilled

    def __transform_matrix_func(self, position, quat):
        # construct matrix for transformation
        try:
            rot = R.from_quat(quat)
        except:
            print(quat)
            sim = sph.Simulation.getCurrent()
            boundary = sim.getBoundaryModel(0)
            animated_body = boundary.getRigidBodyObject()
            print(animated_body.getPosition())
            print(animated_body.getRotation())
            print(self.next_position)
            sys.exit()
        body_matrix = np.zeros((4, 4))
        body_matrix[:3, :3] = rot.as_matrix()
        body_matrix[:3, 3] = position
        body_matrix[3, 3] = 1
        body_matrix = np.matrix(body_matrix)
        return body_matrix

    def __count_particles(self):
        sim = sph.Simulation.getCurrent()
        fluid_model = sim.getFluidModel(0)

        # particles in jug
        boundary = sim.getBoundaryModel(0)
        animated_body = boundary.getRigidBodyObject()
        jug_position = animated_body.getPosition()
        jug_rotation = animated_body.getRotation()

        transformed_jug = self.jug_obj.transform(
            self.__transform_matrix_func(jug_position, jug_rotation), inplace=False
        )
        jug_bounds = transformed_jug.bounds

        """#spilled particles in base of cup.. 
        spilled_collector_bounds = self.spilled_collector_grid.bounds
        
        spilled_collector_bounds = self.__scale_bounds(spilled_collector_bounds, self.cup_scale)
        
        boundary = sim.getBoundaryModel(2) 
        animated_body = boundary.getRigidBodyObject()
        spilled_collector_position = animated_body.getPosition()
        spilled_collector_rotation = animated_body.getRotation()
        
        spilled_collector_bounds = self.__move_bounds(spilled_collector_bounds, spilled_collector_position)
        
        spilled_collector_rotation = R.from_quat(spilled_collector_rotation)
        spilled_collector_bounds = self.__rotate_bounds(spilled_collector_bounds, spilled_collector_rotation)"""

        # collecting the particles in each area
        (
            liq_count_cup,
            liq_ids_cup,
            liq_count_jug,
            liq_ids_jug,
            ids_count_spilled,
            liq_ids_spilled,
        ) = self.__get_liquid_inside_obj(fluid_model, self.cup_bounds, jug_bounds)

        self.particles_ids_jug = liq_ids_jug

        # considering the bounding box of cup includes the feet as well, particles shouldnt be considered FOR WINECUP!
        overlap_cup_jug = list(set(liq_ids_cup) & set(liq_ids_jug))
        liq_ids_cup = [e for e in liq_ids_cup if e not in overlap_cup_jug]

        self.particles_ids_cup = liq_ids_cup

        # all particles within cup, jug and collector
        liq_ids_in_objs = liq_ids_jug + liq_ids_cup

        # print('Number particles: ', fluid_model.numberOfParticles())
        # print('Particles in objects: ', len(liq_ids_in_objs))

        self.n_particles_cup = len(liq_ids_cup)
        self.n_particles_jug = len(liq_ids_jug)
        self.n_particles_spilled = len(liq_ids_spilled)  # fluid_model.numberOfParticles() - len(liq_ids_in_objs)
        self.n_particles_pouring = (
            fluid_model.getNumActiveParticles0() - self.n_particles_spilled - len(liq_ids_in_objs)
        )

        # print('Cup: ', self.n_particles_cup, ' Jug: ', self.n_particles_jug, ' Spilled: ', self.n_particles_spilled, ' Pouring: ', self.n_particles_pouring)

    def __check_collision(self, cup_position, cup_rotation, jug_position, jug_rotation):
        # transform the jug obj to the current pose and check for collisions with the cup obj
        """sim = sph.Simulation.getCurrent()
        boundary = sim.getBoundaryModel(0)
        animated_body = boundary.getRigidBodyObject()
        jug_position = animated_body.getPosition()
        jug_rotation = animated_body.getRotation()

        boundary = sim.getBoundaryModel(1)
        animated_body = boundary.getRigidBodyObject()
        cup_position = animated_body.getPosition()
        cup_rotation = animated_body.getRotation()"""

        transformed_jug = self.jug_obj.transform(
            self.__transform_matrix_func(jug_position, jug_rotation), inplace=False
        )
        transformed_cup = self.cup_obj.transform(
            self.__transform_matrix_func(cup_position, cup_rotation), inplace=False
        )
        _, n_collisions = transformed_cup.collision(
            transformed_jug, contact_mode=1
        )  # self.cup_obj.collision(transformed_jug, contact_mode=1)

        if n_collisions > 0:
            # self.collision = True
            return True
        else:
            # self.collision = False
            return False

    def get_object_position(self, object_number):
        # 0 = jug, 1 = cup
        sim = sph.Simulation.getCurrent()
        boundary = sim.getBoundaryModel(object_number)
        animated_body = boundary.getRigidBodyObject()
        position = animated_body.getPosition()
        position = self.change_of_coordinates(self.cup_position, position)
        rotation = animated_body.getRotation()

        return np.append(position, rotation)

    def get_particle_positions_velocities(self):
        sim = sph.Simulation.getCurrent()
        fluid_model = sim.getFluidModel(0)
        positions = np.zeros((fluid_model.getNumActiveParticles0(), 3))
        velocities = np.zeros((fluid_model.getNumActiveParticles0(), 3))
        for i in range(fluid_model.getNumActiveParticles0()):
            # make sure the particles are always ordered by ids
            id = fluid_model.getParticleId(i)
            p = fluid_model.getPosition(i)
            p = self.change_of_coordinates(self.cup_position, p)
            v = fluid_model.getVelocity(i)
            # temp = [-100 if np.isnan(x) else x for x in np.append(p, v)] # remove nan values
            # positions.append(np.nan_to_num(np.append(p, v), nan=-1))
            positions[id] = np.nan_to_num(p, nan=-100)
            velocities[id] = np.nan_to_num(v, nan=-100)
        return positions, velocities

    def get_particle_accelerations(self):
        sim = sph.Simulation.getCurrent()
        fluid_model = sim.getFluidModel(0)
        accelerations = np.zeros((fluid_model.getNumActiveParticles0(), 3))
        for i in range(fluid_model.getNumActiveParticles0()):
            id = fluid_model.getParticleId(i)
            accelerations[id] = fluid_model.getAcceleration(i)
        return accelerations

    def get_number_of_particles(self):
        return self._number_particles

    def save_particles(self):
        self.base.writeFluidParticlesState(
            "beakernew_fluid_block_rotated.bgeo", sph.Simulation.getCurrent().getFluidModel(0)
        )
        print("saved")

    def change_of_coordinates(self, base, positions):
        pos = base[:3]
        rot = base[3:]
        body_matrix = self.__transform_matrix_func(pos, rot)

        def coordinates_changeRefFrame(pt, body_matrix):
            point = np.copy(pt)
            body_matrix_inverse = body_matrix.I
            coords = body_matrix_inverse.dot(np.append(point, [1])).A1[0:3]
            return coords

        postions_new = coordinates_changeRefFrame(positions, body_matrix)
        return postions_new

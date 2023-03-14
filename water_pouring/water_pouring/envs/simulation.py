import os

import pysplishsplash as sph
import pyvista
import vtk
import pysplishsplash.Utilities.SceneLoaderStructs as Scenes

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)# TODO check vtkMath::Jacobi warning for collision check

from scipy.spatial.transform import Rotation as R

import numpy as np

class Simulation():

    def __init__(self, use_gui, output_directory, jug_start_position, cup_position):
        self.use_gui = use_gui
        self.output_directory = output_directory
        self.jug_start_position = jug_start_position
        self.cup_position = cup_position
        
        self.particles_ids_cup = []
        self.particles_ids_jug = []
        self.collector_dict = []

        self.cup_name = 'wine_glass_test'
        self.jug_name = 'beakernew'

        # calculate the bounds of the cup (it does not move, so this only has to be done once)
        self.cup_obj = pyvista.read(f"water_pouring/water_pouring/envs/ObjectFiles/{self.cup_name}.obj")
        cup_position = self.cup_position[:3]
        cup_rotation = self.cup_position[3:]

        transformed_cup = self.cup_obj.transform(self.__transform_matrix_func(cup_position, cup_rotation), inplace=False)
        self.cup_bounds = transformed_cup.bounds

        self.jug_obj = pyvista.read(f"water_pouring/water_pouring/envs/ObjectFiles/{self.jug_name}.obj")
        self.spilled_collector_grid = pyvista.read("water_pouring/water_pouring/envs/ObjectFiles/UnitBox_open.obj")

        self.__init_simulation()

    def __init_simulation(self):
        base = sph.Exec.SimulatorBase()
        base.init(
                useGui=self.use_gui, sceneFile=os.path.abspath('water_pouring/water_pouring/envs/scene.json'),
                stopAt=5,
                outputDir= self.output_directory
            )
        
        base.setTimeStepCB(self.__time_step_callback)

        # Get the scene and add objects
        scene = sph.Exec.SceneConfiguration.getCurrent().getScene()

        # Add the jug
        jug_file = f'{self.jug_name}.obj'
        rots = self.jug_start_position[3:]
        vec = rots[:3]
        s = rots[3]
        # formula to transform quaternions into axis angle format: see wikipedia
        theta = 2 * np.arctan2(np.linalg.norm(vec), s)
        w = (vec / np.sin(theta/2)) if theta != 0 else 0
        scene.boundaryModels.append(
                Scenes.BoundaryData(
                    meshFile=f'ObjectFiles/{jug_file}',
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
        cup_file = f'{self.cup_name}.obj'
        rots = self.cup_position[3:]
        vec = rots[:3]
        s = rots[3]
        # formula to transform quaternions into axis angle format: see wikipedia
        theta = 2 * np.arctan2(np.linalg.norm(vec), s)
        w = (vec / np.sin(theta/2)) if theta != 0 else 0
        scene.boundaryModels.append(
                Scenes.BoundaryData(
                    meshFile=f'ObjectFiles/{cup_file}',
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

        #scene.camPosition = [1, 2, 2.]

        fluid_init_pos = np.array(self.jug_start_position[:3])
        fluid_init_pos_upper = fluid_init_pos + [0.07, 0.175, 0.07]

        scene.fluidBlocks.append(
                Scenes.FluidBlock(
                    id='Fluid',
                    boxMin=fluid_init_pos,
                    boxMax=fluid_init_pos_upper,
                    mode=0,
                    initialVelocity=[0.0, 0.0, 0.0],
                )
            )

        base.initSimulation()

        sim = sph.Simulation.getCurrent()
        
        sim.setValueInt(sim.BOUNDARY_HANDLING_METHOD, 0)

        base.readFluidParticlesState('water_pouring/water_pouring/envs/LiquidParticles/beakernew_fluid_block_p006_v5.bgeo', sim.getFluidModel(0))
            
        # move particles to the start position of the jug
        fluid_model = sim.getFluidModel(0)

        for i in range(fluid_model.numberOfParticles()):
            curr_pos = fluid_model.getPosition(i)
            fluid_model.setPosition(i, curr_pos + fluid_init_pos)
            fluid_model.setVelocity(i, [0.0, 0.0, 0.0])
            fluid_model.setAcceleration(i, [0.0, 0.0, 0.0])
        
        self.base = base
        self.base.finishInitialization()

    def __time_step_callback(self):  
        sim = sph.Simulation.getCurrent()

        position_change = self.next_position[0]
        rotation_change = self.next_position[1]

        # update the position of the jug
        boundary = sim.getBoundaryModel(0)  # first object is jug 
        animated_body = boundary.getRigidBodyObject()
        new_position = animated_body.getPosition() + position_change
        animated_body.setPosition(new_position)  

        old_rotation = R.from_quat(animated_body.getRotation()).as_euler('XYZ', degrees=True)
        new_rotation = R.from_euler('XYZ', old_rotation + rotation_change, degrees=True)
        animated_body.setRotation(new_rotation.as_quat())
        animated_body.animate()

        # keep the cup at a constant position
        # TODO ist das nötig? (ansonsten macht das return oben schwierigkeiten!)
        boundary = sim.getBoundaryModel(1) 
        animated_body = boundary.getRigidBodyObject()  # second object is cup 
        animated_body.setPosition(self.cup_position[:3])
        animated_body.setRotation(self.cup_position[3:])
        animated_body.animate()   

        self.__count_particles() 

        self.__check_collision()

    def __get_bounds(self, bounds):
        x_bounds = bounds[:2] # xmin, xmax
        y_bounds = bounds[2:4] # ymin, ymax
        z_bounds = bounds[4:] # zmin, zmax
        return x_bounds, y_bounds, z_bounds

    def __get_liquid_inside_obj(self, fluid_model, cup_bounds, jug_bounds):
        ids_inside_cup = []
        ids_inside_jug = []
        cup_x_bounds, cup_y_bounds, cup_z_bounds = self.__get_bounds(cup_bounds)
        jug_x_bounds, jug_y_bounds, jug_z_bounds = self.__get_bounds(jug_bounds)
        for i in range(fluid_model.numberOfParticles()):
            p = fluid_model.getPosition(i)
            if p[0] > cup_x_bounds[0] and p[0] < cup_x_bounds[1]:
                if p[1] > cup_y_bounds[0] and p[1] < cup_y_bounds[1]:
                    if p[2] > cup_z_bounds[0] and p[2] < cup_z_bounds[1]:
                        ids_inside_cup.append(i)
            if p[0] > jug_x_bounds[0] and p[0] < jug_x_bounds[1]:
                if p[1] > jug_y_bounds[0] and p[1] < jug_y_bounds[1]:
                    if p[2] > jug_z_bounds[0] and p[2] < jug_z_bounds[1]:
                        ids_inside_jug.append(i)

        return len(ids_inside_cup), ids_inside_cup, len(ids_inside_jug), ids_inside_jug

    def __transform_matrix_func(self, position, quat):
            #construct matrix for transformation            
            rot = R.from_quat(quat)
            body_matrix = np.zeros((4,4))
            body_matrix[:3,:3] = rot.as_matrix()
            body_matrix[:3,3] = position
            body_matrix[3,3] = 1    
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

        transformed_jug = self.jug_obj.transform(self.__transform_matrix_func(jug_position, jug_rotation), inplace=False)
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
        liq_count_cup, liq_ids_cup, liq_count_jug, liq_ids_jug = self.__get_liquid_inside_obj(fluid_model, self.cup_bounds, jug_bounds)

        self.particles_ids_jug = liq_ids_jug
        
        #considering the bounding box of cup includes the feet as well, particles shouldnt be considered FOR WINECUP!
        overlap_cup_jug = list(set(liq_ids_cup) & set(liq_ids_jug))
        liq_ids_cup = [e for e in liq_ids_cup if e not in overlap_cup_jug]
        
        self.particles_ids_cup = liq_ids_cup
        
        #all particles within cup, jug and collector
        liq_ids_in_objs = liq_ids_jug + liq_ids_cup 

        print('Number particles: ', fluid_model.numberOfParticles())
        print('Particles in objects: ', len(liq_ids_in_objs))

        self.n_particles_cup = len(liq_ids_cup)
        self.n_particles_jug = len(liq_ids_jug)
        self.n_particles_spilled = fluid_model.numberOfParticles() - len(liq_ids_in_objs) # TODO change: flowing particles are currently counted as spilled
    
    def __check_collision(self):
        # transform the jug obj to the current pose and check for collisions with the cup obj
        sim = sph.Simulation.getCurrent()
        boundary = sim.getBoundaryModel(0) 
        animated_body = boundary.getRigidBodyObject()
        jug_position = animated_body.getPosition()
        jug_rotation = animated_body.getRotation()

        transformed_jug = self.jug_obj.transform(self.__transform_matrix_func(jug_position, jug_rotation), inplace=False)

        _, n_collisions = self.cup_obj.collision(transformed_jug, contact_mode=1)
        
        if n_collisions > 0:
            self.collision = True
            return True
        else:
            self.collision = False
            return False

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
        
        cup_bounds = self.cup_obj.bounds
        
        #TODO kann die berechnung von cup_scale auf pyvista übertragen werden?
        reader = vtk.vtkOBJReader()
        reader.SetFileName(f"water_pouring/water_pouring/envs/ObjectFiles/{self.cup_name}.obj")
        reader.Update()

        bounds = reader.GetOutput().GetBounds()
        center = reader.GetOutput().GetCenter()

        xmin, xmax, zmin, zmax, self.ymin, self.ymax = bounds[0]-center[0],bounds[1]-center[0],bounds[2]-center[1],bounds[3]-center[1],bounds[4]-center[2],bounds[5]-center[2], 

        self.cup_scale = [xmax-xmin + .13, ((self.ymax-self.ymin)/3) , zmax-zmin + .13]

        cup_rotation = R.from_quat(self.cup_position[3:])
        
        self.cup_bounds = self.__rotate_bounds(cup_bounds, cup_rotation)

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

        # Create an imgui simulator
        gui = sph.GUI.Simulator_GUI_imgui(base)
        base.setGui(gui)
        
        base.setTimeStepCB(self.__time_step_callback)

        # Get the scene and add objects
        scene = sph.Exec.SceneConfiguration.getCurrent().getScene()

        # Add the jug
        jug_file = f'{self.jug_name}.obj'
        rots = R.from_quat(self.jug_start_position[3:])
        r = np.linalg.norm(rots.as_rotvec())
        scene.boundaryModels.append(
                Scenes.BoundaryData(
                    meshFile=f'ObjectFiles/{jug_file}',
                    translation=self.jug_start_position[:3],
                    scale=[1, 1, 1],
                    color=[0.5, 0.5, 0.5, 1.0],
                    isAnimated=True,
                    axis=[1,0,0],
                    angle=r,
                    isWall=False,
                    mapInvert=False,
                    mapResolution=[25, 25, 25],
                )
            )

        # Add the cup
        cup_file = f'{self.cup_name}.obj'
        rots = R.from_quat(self.cup_position[3:])
        r = np.linalg.norm(rots.as_rotvec())
        scene.boundaryModels.append(
                Scenes.BoundaryData(
                    meshFile=f'ObjectFiles/{cup_file}',
                    translation=self.cup_position[:3],
                    scale=[1, 1, 1],
                    color=[0.5, 0.5, 0.5, 1.0],
                    isAnimated=True,
                    axis=[1,0,0],
                    angle=r,
                    isWall=False,
                    mapInvert=False,
                    mapResolution=[25, 25, 25],
                )
            )

        scene.boundaryModels.append(Scenes.BoundaryData(meshFile="ObjectFiles/UnitBox_open.obj", 
            translation=[self.cup_position[0], self.cup_position[1] - (self.ymax-self.ymin)/2, self.cup_position[2]],#+ 0.03], # 0.03 included so the collector doesnt collide with jug
            scale=[2.1, self.cup_scale[1] * 5, 2.1],#[2.5, 1., 2.5], 
            #scale=cup_scale, # different scale possible for different objects
            color=[0.1, 0.4, 0.5, 1.0], isWall=True, mapInvert=True, mapResolution=[25, 25, 25]))
        
        scene.boundaryModels.append(Scenes.BoundaryData(meshFile="ObjectFiles/UnitBox_open.obj", 
            translation=[self.cup_position[0], self.cup_position[1] - (self.ymax-self.ymin)/2, self.cup_position[2] + 0.03], # 0.03 included so the collector doesnt collide with jug
            #scale=[2.5, cup_scale[1] * 5, 2.5],#[2.5, 1., 2.5], 
            scale=self.cup_scale, # different scale possible for different objects
            color=[0.1, 0.4, 0.5, 1.0], isWall=False, mapInvert=True, mapResolution=[25, 25, 25]))

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
        self.base.deferredInit()

    def __time_step_callback(self):  
        sim = sph.Simulation.getCurrent()

        # sets make_next_step event to false -> agent does not continue making steps
        # otherwise there would be issues with evaluating the correct particle positions for the agent's reward
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
        #self.communications_manager['current_jug_pose'] = [new_position, new_rotation]

        # keep the cup at a constant position
        # TODO ist das nötig? (ansonsten macht das return oben schwierigkeiten!)
        boundary = sim.getBoundaryModel(1) 
        animated_body = boundary.getRigidBodyObject()  # second object is cup 
        animated_body.setPosition(self.cup_position[:3])
        animated_body.setRotation(self.cup_position[3:])
        animated_body.animate()    

        #self.__count_particles() 

        #self.__check_collision()

        """elif command == 'STOP':
            # change stopAt value of the simulation to cause the end of the simulation
            current_time = sph.TimeManager.getCurrent().getTime()
            self.base.setValueFloat(self.base.STOP_AT, 0.01 if current_time - 1 < 0 else current_time - 1) # StopAt has to be higher than 0
            return"""

        # keep the simulation running until the process is stopped
        #self.base.setValueFloat(self.base.STOP_AT, sph.TimeManager.getCurrent().getTime() + 5)

    def __move_bounds(self, bounds, new_position):
        #assumes that the object was originally positioned at [0,0,0]
        min_vector = np.array([bounds[0], bounds[2], bounds[4]])
        max_vector = np.array([bounds[1], bounds[3], bounds[5]])
        
        min_vector += new_position
        max_vector += new_position
        
        return [min_vector[0], max_vector[0], min_vector[1], max_vector[1], min_vector[2], max_vector[2]]
        
    def __rotate_bounds(self, bounds, rotation):
        min_vector = np.array([bounds[0], bounds[2], bounds[4]])
        max_vector = np.array([bounds[1], bounds[3], bounds[5]])
        
        vectors = np.array([min_vector, max_vector])
        rotated_bounds = rotation.apply(vectors)
        
        # during testing the z values were switched? -> manually assigning the min and max values TODO check
        xmin = min(rotated_bounds[:,0])
        xmax = max(rotated_bounds[:,0])
        ymin = min(rotated_bounds[:,1])
        ymax = max(rotated_bounds[:,1])
        zmin = min(rotated_bounds[:,2])
        zmax = max(rotated_bounds[:,2])
        
        return [xmin, xmax, ymin, ymax, zmin, zmax]

    def __scale_bounds(self, bounds, scale):
        xmi,xma,ymi,yma,zmi,zma = bounds
        xmi1 = xmi - scale[0]*abs(xmi)
        xma1 = xma + scale[0]*abs(xma)
        ymi1 = ymi - scale[1]*abs(ymi)
        yma1 = yma + scale[1]*abs(yma)   
        zmi1 = zmi - scale[2]*abs(zmi)
        zma1 = zma + scale[2]*abs(zma)
        return xmi1,xma1,ymi1,yma1,zmi1,zma1

    def __get_bounds(self, bounds):
        x_bounds = bounds[:2] # xmin, xmax
        y_bounds = bounds[2:4] # ymin, ymax
        z_bounds = bounds[4:] # zmin, zmax
        return x_bounds, y_bounds, z_bounds

    def __get_liquid_inside_obj(self, fluid_model, cup_bounds, jug_bounds, collector_bounds):
        ids_inside_cup = []
        ids_inside_jug = []
        ids_inside_collector = []
        cup_x_bounds, cup_y_bounds, cup_z_bounds = self.__get_bounds(cup_bounds)
        jug_x_bounds, jug_y_bounds, jug_z_bounds = self.__get_bounds(jug_bounds)
        collector_x_bounds, collector_y_bounds, collector_z_bounds = self.__get_bounds(collector_bounds)
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
            if p[0] > collector_x_bounds[0] and p[0] < collector_x_bounds[1]:
                if p[1] > collector_y_bounds[0] and p[1] < collector_y_bounds[1]:
                    if p[2] > collector_z_bounds[0] and p[2] < collector_z_bounds[1]:
                        ids_inside_collector.append(i)

        return len(ids_inside_cup), ids_inside_cup, len(ids_inside_jug), ids_inside_jug, len(ids_inside_collector), ids_inside_collector

    def __count_particles(self):
        # TODO maybe check new analysis_new.py file on cluster
        sim = sph.Simulation.getCurrent()
        fluid_model = sim.getFluidModel(0)
                
        # particles in jug         
        jug_bounds = self.jug_obj.bounds
        
        boundary = sim.getBoundaryModel(0) 
        animated_body = boundary.getRigidBodyObject()
        jug_position = animated_body.getPosition()
        jug_rotation = animated_body.getRotation()
        
        jug_bounds = self.__move_bounds(jug_bounds, jug_position)
        
        jug_rotation = R.from_quat(jug_rotation)
        jug_bounds = self.__rotate_bounds(jug_bounds, jug_rotation)
        
        #spilled particles in base of cup.. 
        spilled_collector_bounds = self.spilled_collector_grid.bounds
        
        spilled_collector_bounds = self.__scale_bounds(spilled_collector_bounds, self.cup_scale)
        
        boundary = sim.getBoundaryModel(2) 
        animated_body = boundary.getRigidBodyObject()
        spilled_collector_position = animated_body.getPosition()
        spilled_collector_rotation = animated_body.getRotation()
        
        spilled_collector_bounds = self.__move_bounds(spilled_collector_bounds, spilled_collector_position)
        
        spilled_collector_rotation = R.from_quat(spilled_collector_rotation)
        spilled_collector_bounds = self.__rotate_bounds(spilled_collector_bounds, spilled_collector_rotation)
        
        # collecting the particles in each area
        liq_count_cup, liq_ids_cup, liq_count_jug, liq_ids_jug, liq_count_collector, liq_ids_collector = self.__get_liquid_inside_obj(fluid_model, self.cup_bounds, jug_bounds, spilled_collector_bounds)
        
        self.particles_ids_jug = liq_ids_jug

        self.collector_dict = liq_ids_collector
        
        #considering the bounding box of cup includes the feet as well, particles shouldnt be considered FOR WINECUP!
        overlap_cup_collector = list(set(liq_ids_cup) & set(liq_ids_collector))
        liq_ids_cup = [e for e in liq_ids_cup if e not in overlap_cup_collector]
        overlap_cup_jug = list(set(liq_ids_cup) & set(liq_ids_jug))
        liq_ids_cup = [e for e in liq_ids_cup if e not in overlap_cup_jug]
        
        self.particles_ids_cup = liq_ids_cup
        
        #all particles within cup, jug and collector
        liq_ids_in_objs = liq_ids_jug + liq_ids_cup 

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

        def transform_matrix_func(position, quat):
            #construct matrix for transformation            
            rot = R.from_quat(quat)
            body_matrix = np.zeros((4,4))
            body_matrix[:3,:3] = rot.as_matrix()
            body_matrix[:3,3] = position
            body_matrix[3,3] = 1    
            body_matrix = np.matrix(body_matrix)
            return body_matrix

        transformed_jug = self.jug_obj.transform(transform_matrix_func(jug_position, jug_rotation), inplace=False)

        _, n_collisions = self.cup_obj.collision(transformed_jug, contact_mode=1)
        
        if n_collisions > 0:
            self.collision = True
            return True
        else:
            self.collision = False
            return False

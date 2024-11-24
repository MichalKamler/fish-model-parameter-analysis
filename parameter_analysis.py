import numpy as np
import numpy.typing as npt
import random
import math
import logging as log
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ALP0 = 1.0
ALP1_LR = 0.08
ALP1_UD = 0.08
BET0 = 0.5
BET1_LR = 0.08
BET1_UD = 0.08
LAM0 = 1.0
LAM1_LR = 0.08
LAM1_UD = 0.08

SIM_RATE = 10 #Hz

# PHI_SIZE = 1024
PHI_SIZE = 256
THETA_SIZE = 128

GAM = 0.1 #relaxation rate
V0 = 1.0 #prefered vel
R = 1.0 #radius of drone 

N_DRONES = 10
SIM_TIME = 600.0 #s
SIM_START_COLLECT_DATA = 200.0 #s 
SPAWN_DIST = 5 #m

PI = math.pi
PI_2 = PI / 2

class drone: 
    def __init__(self, x: float, y: float, z: float, psi: float =None, 
                    PHI_SIZE: int =PHI_SIZE, THETA_SIZE: int =THETA_SIZE,
                    GAM: float =GAM, V0: float =V0, R: float =R, SIM_RATE: int=SIM_RATE,
                    ALP0: float =ALP0, ALP1_LR: float =ALP1_LR, ALP1_UD: float =ALP1_UD, 
                    BET0: float =BET0, BET1_LR: float =BET1_LR, BET1_UD: float =BET1_UD, 
                    LAM0: float =LAM0, LAM1_LR: float =LAM1_LR, LAM1_UD: float =LAM1_UD) -> None:
        self.x = x #global coords
        self.y = y 
        self.z = z

        self.PHI_SIZE, self.THETA_SIZE = PHI_SIZE, THETA_SIZE
        self.d_phi = 2*PI/PHI_SIZE
        self.d_theta = PI/THETA_SIZE

        self.GAM, self.V0, self.R = GAM, V0, R

        self.SIM_RATE = SIM_RATE
        self.d_t = 1/SIM_RATE

        self.ALP0, self.ALP1_LR, self.ALP1_UD = ALP0, ALP1_LR, ALP1_UD
        self.BET0, self.BET1_LR, self.BET1_UD = BET0, BET1_LR, BET1_UD
        self.LAM0, self.LAM1_LR, self.LAM1_UD = LAM0, LAM1_LR, LAM1_UD

        self.V = VisualField(PHI_SIZE, THETA_SIZE) 

        phi_lin_spaced = np.linspace(-np.pi, np.pi, PHI_SIZE)
        theta_lin_spaced = np.linspace(-np.pi / 2, np.pi / 2, THETA_SIZE)

        self.cos_phi = np.cos(phi_lin_spaced)
        self.sin_phi = np.sin(phi_lin_spaced)
        self.cos_theta = np.cos(theta_lin_spaced)
        self.sin_theta = np.sin(theta_lin_spaced)

        if psi==None: #psi is direction of velocity in global coords
            self.psi = 0.0
        else:
            self.psi = psi
        self.velocity = np.array([0.0, 0.0, 0.0]) #vx, vy, vz in global coords
        self.velocity_norm = 0.0

    def dPhi_V_of(self, V: npt.ArrayLike) -> npt.ArrayLike:
        # circular padding for edge cases
        padV = np.pad(V, (1, 1), 'wrap')
        dPhi_V_raw = np.diff(padV)

        # we want to include non-zero value if it is on the edge
        if dPhi_V_raw[0] > 0 and dPhi_V_raw[-1] > 0:
            dPhi_V_raw = dPhi_V_raw[0:-1]

        else:
            dPhi_V_raw = dPhi_V_raw[1:, ...]

        dPhi_V = dPhi_V_raw
        return dPhi_V
    
    def dTheta_V_of(self, row_idx: int) -> npt.ArrayLike:
        dTheta_V = np.zeros(self.PHI_SIZE)
        if row_idx<=0 or row_idx>=self.PHI_SIZE:
            return dTheta_V
        
        for col_idx in range(self.PHI_SIZE):
            dTheta_V[col_idx] = self.V[row_idx, col_idx] - self.V[row_idx-1, col_idx]
        return dTheta_V

    def compute_state_variables_3d(self):
        v_integral_by_dphi = np.zeros(self.THETA_SIZE)
        psi_integral_by_dphi = np.zeros(self.THETA_SIZE)
        v_z_integral_by_dphi = np.zeros(self.THETA_SIZE)

        for i in range(self.THETA_SIZE):
            angle_z_to_theta = np.pi - i * (np.pi / (self.THETA_SIZE - 1))
            V_row = self.V.field[i, :].astype(float)
            dPhi_V = self.dPhi_V_of(V_row)
            dTheta_V = self.dTheta_V_of(i)

            G = -V_row
            G_spike = dPhi_V**2

            integrand_dvel = G * self.cos_phi
            integrand_dpsi = G * self.sin_phi

            integral_dvel = self.d_phi * (0.5 * integrand_dvel[0] + integrand_dvel[1:self.PHI_SIZE - 1].sum() + 0.5 * integrand_dvel[self.PHI_SIZE - 1])
            integral_dpsi = self.d_phi * (0.5 * integrand_dpsi[0] + integrand_dpsi[1:self.PHI_SIZE - 1].sum() + 0.5 * integrand_dpsi[self.PHI_SIZE - 1])
            integral_dv_z = self.d_phi * (0.5 * G[0] + G[1:self.PHI_SIZE - 1].sum() + 0.5 * G[self.PHI_SIZE - 1])

            v_integral_by_dphi[i] = self.ALP0 * (integral_dvel + self.ALP1_LR * (self.cos_phi * G_spike).sum() + self.ALP1_UD * (self.cos_phi * (dTheta_V**2)).sum()) * np.sin(angle_z_to_theta)
            psi_integral_by_dphi[i] = self.BET0 * (integral_dpsi + self.BET1_LR * (self.sin_phi * G_spike).sum() + self.BET1_UD * (self.sin_phi * (dTheta_V**2)).sum()) * np.sin(angle_z_to_theta)
            v_z_integral_by_dphi[i] = self.LAM0 * (integral_dv_z + self.LAM1_LR * G_spike.sum() + self.LAM1_UD * (dTheta_V**2).sum()) * np.sin(angle_z_to_theta)

        v_integral_by_dphi *= self.cos_theta
        psi_integral_by_dphi *= self.cos_theta
        v_z_integral_by_dphi *= self.sin_theta

        dvel = self.d_theta * (0.5 * v_integral_by_dphi[0] + v_integral_by_dphi[1:self.THETA_SIZE - 1].sum() + 0.5 * v_integral_by_dphi[self.THETA_SIZE - 1])
        dpsi = self.d_theta * (0.5 * psi_integral_by_dphi[0] + psi_integral_by_dphi[1:self.THETA_SIZE - 1].sum() + 0.5 * psi_integral_by_dphi[self.THETA_SIZE - 1])
        dv_z = self.d_theta * (0.5 * v_z_integral_by_dphi[0] + v_z_integral_by_dphi[1:self.THETA_SIZE - 1].sum() + 0.5 * v_z_integral_by_dphi[self.THETA_SIZE - 1])

        dvel += self.GAM * (self.V0 - self.velocity_norm)

        return dvel, dpsi, dv_z

    def updateVelocity(self, dvel: float, dpsi: float, dvz: float) -> None:
        self.velocity_norm += dvel * self.d_t
        self.psi += dpsi * self.d_t
        self.velocity[0] = self.velocity_norm * np.cos(self.psi)
        self.velocity[1] = self.velocity_norm * np.sin(self.psi)
        self.velocity[2] += dvz * self.d_t
        return

    def updatePositon(self) -> None:
        self.x += self.velocity[0] * self.d_t
        self.y += self.velocity[1] * self.d_t
        self.z += self.velocity[2] * self.d_t
        return

    def setZeroVisualField(self) -> None:
        self.V.setZero()
        return

    def updateVFieldBasedOnDroneCoords(self, x: float, y: float, z:float) -> None:
        """
        input xyz are global coords of newly observed drone that I want to push into visual field  
        """
        relative_xy = np.array([self.x - x, self.y -y])
        relative_z = self.z - z
        dist = np.sqrt(pow(relative_xy[0],2) + pow(relative_xy[1],2) + pow(relative_z,2))
        Rot_psi = np.array([[np.cos(self.psi), -np.sin(self.psi)],
                            [np.sin(self.psi), np.cos(self.psi)]])
        xy_local = Rot_psi @ relative_xy
        phi = np.atan2(xy_local[1], xy_local[0])
        theta = np.arcsin(abs(relative_z) / dist)
        alpha = np.atan(R/dist)
        self.V.setSphereCap(phi, theta, alpha)
        return

        
class simulation:
    def __init__(self, SIM_RATE: int, N_DRONES: int =N_DRONES, SIM_TIME: float =SIM_TIME, SIM_START_COLLECT_DATA: float =SIM_START_COLLECT_DATA, SHOW: bool =False) -> None:
        self.SIM_RATE = SIM_RATE
        self.N_DRONES = N_DRONES
        self.SIM_TIME = SIM_TIME
        self.SIM_START_COLLECT_DATA = SIM_START_COLLECT_DATA
        self.SHOW = SHOW
        self.drones = self.spawn_drones(n_drones=N_DRONES)
        self.minDist = float('inf')
        self.avgMinDist = 0
        self.cntMinDist = 0
    
    def spawn_drones(self, n_drones) -> list:
        drones_list = []
        for i in range(n_drones):
            x = i*3
            y = random.uniform(-3, 3)
            z = random.uniform(0, 5)
            drones_list.append(drone(x, y, z, 0, 
                    PHI_SIZE, THETA_SIZE,
                    GAM, V0, R, SIM_RATE,
                    ALP0, ALP1_LR, ALP1_UD, 
                    BET0, BET1_LR, BET1_UD, 
                    LAM0, LAM1_LR, LAM1_UD))

        return drones_list
    
    def getDronesPositions(self):
        x = np.zeros(self.N_DRONES)
        y = np.zeros(self.N_DRONES)
        z = np.zeros(self.N_DRONES)

        for i, drone in enumerate(self.drones):
            x[i] = drone.x
            y[i] = drone.y
            z[i] = drone.z
        return x, y, z
    
    def startPlotDrones(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = self.getDronesPositions()
        self.sc = ax.scatter(x, y, z, c='blue', marker='o')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])

        plt.ion()  # Turn on interactive mode
        plt.show()
    
    def updatePlotDrones(self):
        x, y, z = self.getDronesPositions()
        self.sc._offsets3d = (x, y, z)
        plt.pause(0.1) 
    
    def updateVFieldForDrones(self):
        x, y, z = self.getDronesPositions()
        for drone in self.drones:
            drone.setZeroVisualField()
            x_i, y_i, z_i = drone.x, drone.y, drone.z

            for j in range(self.N_DRONES):
                if x[j]!=x_i and y[j]!=y_i and z[j]!=z_i:
                    drone.updateVFieldBasedOnDroneCoords(x[j], y[j], z[j])
    
    def updateDronesPosition(self):
        for drone in self.drones:
            drone.updatePositon()

    def updateDronesStateVar(self):
        for drone in self.drones:
            dvel, dpsi, dv_z = drone.compute_state_variables_3d()
            drone.updateVelocity(dvel, dpsi, dv_z)

    def getMinDistInfo(self):
        x, y, z = self.getDronesPositions()
        n = len(x)

        min_distances = np.array([])

        now_observed_min_dist = float('inf')

        for i in range(n):
            min_dist = float('inf')  # Start with a large number
            for j in range(n):
                if i != j:  # Skip the same point
                    dist = math.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)
                    min_dist = min(min_dist, dist)
            now_observed_min_dist = min(now_observed_min_dist, min_dist)
            min_distances.append(min_dist)
        self.minDist = min(self.minDist, now_observed_min_dist)
        self.avgMinDist = (self.avgMinDist*self.cntMinDist + min_distances.sum())/(self.cntMinDist+n)


    def simulate(self):
        dt = 1/self.SIM_RATE
        t = 0
        if self.SHOW:
            self.startPlotDrones()
        while t<self.SIM_TIME:
            t += dt

            self.updateVFieldForDrones()
            self.updateDronesStateVar()
            self.updateDronesPosition()

            if self.SHOW:
                self.updatePlotDrones()
            if t>=self.SIM_START_COLLECT_DATA:
                self.getMinDistInfo()
        print(self.minDist, self.avgMinDist)
                
                

    

class VisualField: 
    """
    spherical representation of the visual field as a rows x cols -> theta x phi discretization
    
    """
    def __init__(self, phi_size: int, theta_size: int) -> None:
        self.field = np.zeros((theta_size, phi_size))
        self.theta_size = theta_size #rows
        self.phi_size = phi_size #cols
        self.d_phi = 2*PI/phi_size
        self.d_theta = PI/theta_size
    
    def __getitem__(self, indices):
        return self.field[indices]  

    def __setitem__(self, indices, value):
        self.field[indices] = value  

    def setZero(self) -> None:
        self.field.fill(0)
    
    def phiToRange(self, phi: float) -> float: #TESTED
        """
        phi is periodical
        converts phi so it is in interval [-pi, pi)
        """
        while phi<-PI:
            phi += 2*PI
        while phi>=PI:
            phi -= 2*PI
        return phi

    def thetaToRange(self, theta: float) -> float:        
        return max(min(theta, PI_2 - self.d_theta), -PI_2 + self.d_theta)

    def phiToCol(self, phi: float) -> int: #TESTED
        """
        converts angle phi to its corresponding column in field representation - from 0 to phi_sieze-1
        """
        if phi<-PI or phi>=PI:
            phi = self.phiToRange(phi)
        col = int((phi + PI)*(self.phi_size)/(2*PI))
        if col<0 or col>=self.phi_size: #DEBUG
            log.error("Calculated col outside range of field")
            return 0
        return col

    def colToPhi(self, col: int) -> float: #TESTED
        """
        converts column to its corresponding phi angle - from [-pi, pi)
        """
        if col<0 or col>=self.phi_size:
            log.error("Unexpected index when converting col index to angle phi")
            col = 0
        phi = col * (2*PI)/(self.phi_size) - PI
        if phi<-PI or phi>=PI: #DEBUG
            log.error("Calculated wrong phi that is not in [-pi, pi)")
        return phi
    
    def thetaToRow(self, theta: float) -> int: #TESTED
        """
        converts angle theta to its corresponding row in field representation - from 0 to theta_size-1
        """
        theta = self.thetaToRange(theta) #d_theta to make sure that it falls into correct interval
        row = int((theta + PI_2)*(self.theta_size)/(PI))
        if row<0 or row>=self.theta_size: #DEBUG
            log.error("Calculated row outside range of field")
            return 0
        return row

    def rowToTheta(self, row: int) -> float: #TESTED
        """
        converts row to its corresponding theta angel - from [-pi/2, pi/2]
        """
        if row<0 or row >=self.theta_size:
            log.error("Unexpected index when converting row index to angle theta")
            row = 0
        theta = row * (PI)/(self.theta_size-1) - PI_2
        if theta<-PI_2 or theta >PI_2: #DEBUG
            log.error("Calculated wrong theta taht is not in [-pi/2, pi/2]")
            return 0
        return theta
    
    def sphericalToCartesian(self, phi: float, theta: float, radius:float =1.0) -> tuple:
        """
        phi is from [0 to 2pi) - angle in the xy-plane from the x-axis
        theta is from [0, pi] - angle from the positive z-axis
        radius - distance from origin
        """
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return x, y, z
    
    def phiShift(self, phi: float) -> float:
        """
        shifts phi that is usually from [-pi, pi) to [0, 2pi)
        """
        if phi<-PI or phi>=PI:
            phi = self.phiToRange(phi)
        if phi < 0:
            phi += 2*PI
        return phi

    def thetaShift(self, theta: float) -> float:
        """
        shifts theta to interval [0, pi], angle represents angle from the positive z-axis
        """
        theta = self.thetaToRange(theta)
        theta = PI_2 - theta
        return theta
    
    def fillFieldWithinAlpha(self, row_start: int, row_end: int, col_start: int, col_end: int, x_c: float, y_c: float, z_c: float, alpha: float) -> None:
        """
        helper function for setSphereCap
        """
        self.field[row_start:row_end, col_start:col_end] = 1
        # for row in range(row_start, row_end):
        #         for col in range(col_start, col_end):
        #             phi = self.colToPhi(col)
        #             theta  =self.rowToTheta(row)
        #             x, y, z = self.sphericalToCartesian(self.phiShift(phi), self.thetaShift(theta))
        #             dot_product = x*x_c + y*y_c + z*z_c
        #             distance = np.acos(dot_product)
        #             if distance <= alpha:
        #                 self.field[row, col] = 1
        return

    def setSphereCap(self, phi_center: float, theta_center: float, alpha: float) -> None: #TESTED
        """
        phi_center - angle in xy plane in drone frame of reference
        theta_center - angle from xy to the center of the cap
        alpha - angle from the center of the cap to the edge of the cap
        """
        x_center, y_center, z_center = self.sphericalToCartesian(self.phiShift(phi_center), self.thetaShift(theta_center))
        theta_min = self.thetaToRow(self.thetaToRange(theta_center-alpha))
        theta_max = self.thetaToRow(self.thetaToRange(theta_center+alpha))
        if PI_2-theta_center<=alpha or -PI_2-theta_center<=alpha:
            phi_min = 0
            phi_max = self.phi_size-1
        else: 
            phi_min = self.phiToCol(phi_center-alpha)
            phi_max = self.phiToCol(phi_center+alpha)

        if phi_min <= phi_max: #normal case
            self.fillFieldWithinAlpha(theta_min, theta_max+1, phi_min, phi_max+1, x_center, y_center, z_center, alpha)
        else: #sad case :(
            self.fillFieldWithinAlpha(theta_min, theta_max+1, phi_min, self.phi_size, x_center, y_center, z_center, alpha)

            self.fillFieldWithinAlpha(theta_min, theta_max+1, 0, phi_max, x_center, y_center, z_center, alpha)
        return

    def plotVisualField(self) -> None:
        # Parameters
        r = 1  # Radius of the sphere

        # Create spherical coordinate grid
        theta = np.linspace(-np.pi/2, np.pi/2, self.theta_size)  # Elevation angle (-π/2 to π/2)
        phi = np.linspace(-np.pi, np.pi, self.phi_size)          # Azimuthal angle (-π to π)
        phi, theta = np.meshgrid(phi, theta)  # Create a 2D grid

        # Set the upper half of the field to 1
        # self.field[theta <= np.pi / 2] = 1

        # Convert spherical to Cartesian coordinates
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.cos(theta) * np.sin(phi)
        z = r * np.sin(theta)

        # Plot the field on the sphere
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Use a custom colormap for dark blue (1) and lighter blue (0)
        ax.plot_surface(x, y, z, facecolors=plt.cm.Blues(self.field), rstride=1, cstride=1)

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Optional: Style the plot
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.set_title("Visual Field on Sphere", fontsize=16)
        ax.axis("on")  # Show the axes
        ax.grid(True)  # Turn on the grid

        plt.show()

        # Optional: Style the plot
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.set_title("Visual Field on Sphere", fontsize=16)
        ax.axis("off")

        plt.show()




if __name__ == "__main__":
    # log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # V = VisualField(PHI_SIZE, THETA_SIZE)
    # V.setSphereCap(0, -PI_2, PI/4)
    # V.plotVisualField()

    # drone1 = drone(0, 0, 0)
    # for i in range(30):
    #     drone1.updateVelocity(1, 0, 0)
    #     print("time: ", i*1/30, "position: ", drone1.velocity[0], drone1.velocity[1], drone1.velocity[2])
    sim = simulation(SIM_RATE=SIM_RATE, N_DRONES=10, SIM_TIME=SIM_TIME, SIM_START_COLLECT_DATA=SIM_START_COLLECT_DATA, SHOW=True)
    sim.simulate()



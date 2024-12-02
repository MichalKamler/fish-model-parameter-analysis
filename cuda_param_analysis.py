import numpy as np
import numpy.typing as npt
import random
import math
import logging as log
import time
import itertools
from tqdm import tqdm
import csv
from numba import cuda, float32


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ALP0 = 1.0
ALP1_LR = 0.16
ALP1_UD = 0.
BET0 = 0.5
BET1_LR = 0.16
BET1_UD = 0.
LAM0 = 1.0
LAM1_LR = 0.0
LAM1_UD = 0.16

SIM_RATE = 5 #Hz

# PHI_SIZE = 1024
PHI_SIZE = 256
THETA_SIZE = 128
# PHI_SIZE = 16384
# THETA_SIZE = 8192

GAM = 0.1 #relaxation rate
V0 = 1.0 #prefered vel
R = 1.0 #radius of drone 
RELAXATION_RATE_VZ = 0.75
RELAXATION_RATE_VZ = RELAXATION_RATE_VZ ** (1/SIM_RATE)

N_DRONES = 10
SIM_TIME = 300.0 #s
SIM_START_COLLECT_DATA = 100.0 #s 
SPAWN_DIST = 5 #m

USE_BOUNDARY_BOX = True
BOX_WIDTH = 20. #m
BOX_LENGTH = 20. #m
BOX_HEIGHT = 10. #m
BOX_DIST_FROM_GROUND = 1. #m

MAX_XY_VEL = 1.5 # m/s
MAX_Z_VEL = 0.5 # m/s
MAX_ROT_VEL = 1. # rad/s

BLACKEN_V_FIELD = np.pi/4

UPDATE_FIELD_ONLY_CLOSER_THAN_10 = False

PI = math.pi
PI_2 = PI / 2

USE_GPU = True

if USE_GPU:
    phi_lin_spaced = np.linspace(-np.pi, np.pi, PHI_SIZE).astype(np.float32)
    theta_lin_spaced = np.linspace(-np.pi / 2, np.pi / 2, THETA_SIZE).astype(np.float32)

    cos_phi = np.cos(phi_lin_spaced).astype(np.float32)
    sin_phi = np.sin(phi_lin_spaced).astype(np.float32)
    cos_theta = np.cos(theta_lin_spaced).astype(np.float32)
    sin_theta = np.sin(theta_lin_spaced).astype(np.float32)

    theta_indices = np.arange(THETA_SIZE)
    angle_z_to_theta = np.pi - theta_indices * (np.pi / (THETA_SIZE - 1))
    sin_angle_z_to_theta = np.sin(angle_z_to_theta)

    cos_phi_device = cuda.to_device(cos_phi.astype(np.float32))
    sin_phi_device = cuda.to_device(sin_phi.astype(np.float32))
    sin_angle_z_to_theta_device = cuda.to_device(sin_angle_z_to_theta.astype(np.float32))
    cos_theta_device = cuda.to_device(cos_theta.astype(np.float32))
    sin_theta_device = cuda.to_device(sin_theta.astype(np.float32))

    v_integral_by_dphi_device = cuda.to_device(np.zeros(THETA_SIZE, dtype=np.float32))
    psi_integral_by_dphi_device = cuda.to_device(np.zeros(THETA_SIZE, dtype=np.float32))
    vz_integral_by_dphi_device = cuda.to_device(np.zeros(THETA_SIZE, dtype=np.float32))


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

        self.V = VisualField(PHI_SIZE, THETA_SIZE, True) 

        phi_lin_spaced = np.linspace(-np.pi, np.pi, PHI_SIZE).astype(np.float32)
        theta_lin_spaced = np.linspace(-np.pi / 2, np.pi / 2, THETA_SIZE).astype(np.float32)

        self.cos_phi = np.cos(phi_lin_spaced).astype(np.float32)
        self.sin_phi = np.sin(phi_lin_spaced).astype(np.float32)
        self.cos_theta = np.cos(theta_lin_spaced).astype(np.float32)
        self.sin_theta = np.sin(theta_lin_spaced).astype(np.float32)

        if psi==None: #psi is direction of velocity in global coords
            self.psi = 0.0
        else:
            self.psi = psi
        self.velocity = np.array([0.0, 0.0, 0.0]) #vx, vy, vz in global coords
        self.velocity_norm = 0.0

        # Precompute common factors to avoid repeated calculations
        theta_indices = np.arange(self.THETA_SIZE)
        angle_z_to_theta = np.pi - theta_indices * (np.pi / (self.THETA_SIZE - 1))
        self.sin_angle_z_to_theta = np.sin(angle_z_to_theta)

        # if USE_GPU:
            

            # self.ALP0_device, self.ALP1_LR_device, self.ALP1_UD_device = cuda.to_device(np.array([ALP0], dtype=np.float32)), cuda.to_device(np.array([ALP1_LR], dtype=np.float32)), cuda.to_device(np.array([ALP1_UD], dtype=np.float32))
            # self.BET0_device, self.BET1_LR_device, self.BET1_UD_device = cuda.to_device(np.array([BET0], dtype=np.float32)), cuda.to_device(np.array([BET1_LR], dtype=np.float32)), cuda.to_device(np.array([BET1_UD], dtype=np.float32))
            # self.LAM0_device, self.LAM1_LR_device, self.LAM1_UD_device = cuda.to_device(np.array([LAM0], dtype=np.float32)), cuda.to_device(np.array([LAM1_LR], dtype=np.float32)), cuda.to_device(np.array([LAM1_UD], dtype=np.float32))

            # self.d_phi_device, self.d_theta_device = cuda.to_device(np.array([self.d_phi], dtype=np.float32)), cuda.to_device(np.array([self.d_theta], dtype=np.float32))
            # self.GAM_device, self.V0_device = cuda.to_device(np.array([GAM], dtype=np.float32)), cuda.to_device(np.array([V0], dtype=np.float32))

            # self.PHI_SIZE_device, self.THETA_SIZE_device = cuda.to_device(np.array([PHI_SIZE], dtype=np.int32)), cuda.to_device(np.array([PHI_SIZE], dtype=np.int32))

            # Allocate device memory for output arrays
            
    
    def dPhi_V_of(self, V: npt.ArrayLike) -> npt.ArrayLike:
        # Compute circular differences directly without padding
        dPhi_V_raw = np.diff(V, prepend=V[-1], append=V[0])

        # Handle the edge condition efficiently
        if dPhi_V_raw[0] > 0 and dPhi_V_raw[-1] > 0:
            return dPhi_V_raw[:-1]
        else:
            return dPhi_V_raw[1:]
    
    def dTheta_V_of(self, row_idx: int) -> npt.ArrayLike:
        # Check for edge cases
        if row_idx <= 0 or row_idx >= self.PHI_SIZE:
            return np.zeros(self.PHI_SIZE)
        
        # Use slicing to compute the difference directly
        return self.V[row_idx, :] - self.V[row_idx - 1, :]

    @staticmethod
    @cuda.jit
    def compute_state_kernel(V_field, cos_phi, sin_phi, sin_angle_z_to_theta, cos_theta, sin_theta,
                            ALP0, ALP1_LR, ALP1_UD, BET0, BET1_LR, BET1_UD, LAM0, LAM1_LR, LAM1_UD,
                            d_phi, d_theta, GAM, V0, velocity_norm,
                            v_integral_by_dphi_device, psi_integral_by_dphi_device, vz_integral_by_dphi_device, PHI_SIZE, THETA_SIZE):
        # row, col = cuda.grid(2)  # 2D grid indices: row (theta), col (phi)
        col = cuda.threadIdx.x
        row = cuda.blockIdx.x

        # row, col = cuda.grid(2)

        # Shared memory for reduction across the row (phi axis)
        smem_dvel = cuda.shared.array(256, dtype=float32)
        smem_dpsi = cuda.shared.array(256, dtype=float32)
        smem_dvz = cuda.shared.array(256, dtype=float32)

        if row < THETA_SIZE and col < PHI_SIZE:

            V_rc_val = V_field[row, col] #Value of Visual field at index row and col

            dPhi_V_rc = V_field[row, (col+1)%PHI_SIZE] - V_rc_val

            if row > 0:
                dTheta_rc = V_rc_val - V_field[row-1, col]
            else: 
                dTheta_rc = 0.
            G_rc = -V_rc_val
            G_spike_rc = dPhi_V_rc ** 2
            G_spike_UD_rc = dTheta_rc ** 2

            integrand_dvel = G_rc * cos_phi[col]
            integrand_dpsi = G_rc * sin_phi[col]
            integrand_dvz = G_rc

            smem_dvel[col] = integrand_dvel
            smem_dpsi[col] = integrand_dpsi
            smem_dvz[col] = integrand_dvz
            cuda.syncthreads()

            integral_dvel = 0.0
            integral_dpsi = 0.0
            integral_dvz = 0.0
            if col == 0:
                integral_dvel += 0.5 * smem_dvel[0]  # First term
                integral_dpsi += 0.5 * smem_dpsi[0]
                integral_dvz += 0.5 * smem_dvz[0]
            elif col == PHI_SIZE - 1:
                integral_dvel += 0.5 * smem_dvel[col]  # Last term
                integral_dpsi += 0.5 * smem_dpsi[col]
                integral_dvz += 0.5 * smem_dvz[col]
            else:
                integral_dvel += smem_dvel[col]  # Middle terms
                integral_dpsi += smem_dpsi[col]
                integral_dvz += smem_dvz[col]

            integral_dvel *= d_phi
            integral_dpsi *= d_phi
            integral_dvz *= d_phi

            cuda.atomic.add(v_integral_by_dphi_device, row, ALP0 * (integral_dvel + ALP1_LR * G_spike_rc * cos_phi[col] + ALP1_UD * G_spike_UD_rc * cos_phi[col]) * sin_angle_z_to_theta[row])

            cuda.atomic.add(psi_integral_by_dphi_device, row, BET0 * (integral_dpsi + BET1_LR * G_spike_rc * sin_phi[col] + BET1_UD * G_spike_UD_rc * sin_phi[col]) * sin_angle_z_to_theta[row])

            cuda.atomic.add(vz_integral_by_dphi_device, row, LAM0 * (integral_dvz + LAM1_LR * G_spike_rc + LAM1_UD * G_spike_UD_rc) * sin_angle_z_to_theta[row])

    def compute_state_variables_3d_on_gpu(self):
        # Allocate GPU memory for input arrays

        # cos_phi_device = cuda.to_device(self.cos_phi.astype(np.float32))
        # sin_phi_device = cuda.to_device(self.sin_phi.astype(np.float32))
        # sin_angle_z_to_theta_device = cuda.to_device(self.sin_angle_z_to_theta.astype(np.float32))
        # cos_theta_device = cuda.to_device(self.cos_theta.astype(np.float32))
        # sin_theta_device = cuda.to_device(self.sin_theta.astype(np.float32))

        V_field_device = cuda.to_device(self.V.field.astype(np.float32))
        
        velocity_norm_device = cuda.to_device(np.array([self.velocity_norm]).astype(np.float32))

        v_integral_by_dphi_device[:] = 0.
        psi_integral_by_dphi_device[:] = 0.
        vz_integral_by_dphi_device[:] = 0.

        # self.v_integral_by_dphi_device = cuda.to_device(np.zeros(self.THETA_SIZE, dtype=np.float32))
        # self.psi_integral_by_dphi_device = cuda.to_device(np.zeros(self.THETA_SIZE, dtype=np.float32))
        # self.vz_integral_by_dphi_device = cuda.to_device(np.zeros(self.THETA_SIZE, dtype=np.float32))
        
        # Kernel launch configuration
        threads_per_block = (256, 1)  # 256 threads in the x-direction (columns), 1 thread in the y-direction (rows)
        blockspergrid_x = 128  # One block for each row
        blockspergrid_y = 1    # Only one block in the y-direction (since each block covers one row)
        blocks_per_grid = (blockspergrid_x, blockspergrid_y)
        # Launch the kernel
        cuda.synchronize()
        self.compute_state_kernel[blocks_per_grid, threads_per_block](
            V_field_device, cos_phi_device, sin_phi_device, sin_angle_z_to_theta_device,
            cos_theta_device, sin_theta_device, self.ALP0, self.ALP1_LR, self.ALP1_UD,
            self.BET0, self.BET1_LR, self.BET1_UD, self.LAM0, self.LAM1_LR, self.LAM1_UD,
            self.d_phi, self.d_theta, self.GAM, self.V0, velocity_norm_device,
            v_integral_by_dphi_device, psi_integral_by_dphi_device, vz_integral_by_dphi_device, self.PHI_SIZE, self.THETA_SIZE)
        cuda.synchronize()

        # Copy results back to host
        v_integral_by_dphi = v_integral_by_dphi_device.copy_to_host()
        psi_integral_by_dphi = psi_integral_by_dphi_device.copy_to_host()
        vz_integral_by_dphi = vz_integral_by_dphi_device.copy_to_host()

        # Compute final results on the host
        v_integral_by_dphi *= self.cos_theta
        psi_integral_by_dphi *= self.cos_theta
        vz_integral_by_dphi *= self.sin_theta

        dvel = self.d_theta * (0.5 * v_integral_by_dphi[0] + v_integral_by_dphi[1:-1].sum() + 0.5 * v_integral_by_dphi[-1])
        dpsi = self.d_theta * (0.5 * psi_integral_by_dphi[0] + psi_integral_by_dphi[1:-1].sum() + 0.5 * psi_integral_by_dphi[-1])
        dv_z = self.d_theta * (0.5 * vz_integral_by_dphi[0] + vz_integral_by_dphi[1:-1].sum() + 0.5 * vz_integral_by_dphi[-1])

        dvel += self.GAM * (self.V0 - self.velocity_norm)

        return dvel, dpsi, dv_z
    
    def compute_state_variables_3d_on_cpu(self):
        v_integral_by_dphi = np.zeros(self.THETA_SIZE)
        psi_integral_by_dphi = np.zeros(self.THETA_SIZE)
        v_z_integral_by_dphi = np.zeros(self.THETA_SIZE)

        for i in range(self.THETA_SIZE):
            V_row = self.V.field[i, :].astype(float)
            dPhi_V = self.dPhi_V_of(V_row)
            dTheta_V = self.dTheta_V_of(i)

            G = -V_row
            G_spike = dPhi_V**2
            G_spike_UD = dTheta_V**2

            integrand_dvel = G * self.cos_phi
            integrand_dpsi = G * self.sin_phi

            integral_dvel = self.d_phi * (0.5 * integrand_dvel[0] + integrand_dvel[1:self.PHI_SIZE - 1].sum() + 0.5 * integrand_dvel[self.PHI_SIZE - 1])
            integral_dpsi = self.d_phi * (0.5 * integrand_dpsi[0] + integrand_dpsi[1:self.PHI_SIZE - 1].sum() + 0.5 * integrand_dpsi[self.PHI_SIZE - 1])
            integral_dv_z = self.d_phi * (0.5 * G[0] + G[1:self.PHI_SIZE - 1].sum() + 0.5 * G[self.PHI_SIZE - 1])

            v_integral_by_dphi[i] = self.ALP0 * (integral_dvel + self.ALP1_LR * (self.cos_phi * G_spike).sum() + self.ALP1_UD * (self.cos_phi * (G_spike_UD)).sum()) * self.sin_angle_z_to_theta[i]
            psi_integral_by_dphi[i] = self.BET0 * (integral_dpsi + self.BET1_LR * (self.sin_phi * G_spike).sum() + self.BET1_UD * (self.sin_phi * (G_spike_UD)).sum()) * self.sin_angle_z_to_theta[i]
            v_z_integral_by_dphi[i] = self.LAM0 * (integral_dv_z + self.LAM1_LR * G_spike.sum() + self.LAM1_UD * (G_spike_UD).sum()) * self.sin_angle_z_to_theta[i]

        v_integral_by_dphi *= self.cos_theta
        psi_integral_by_dphi *= self.cos_theta
        v_z_integral_by_dphi *= self.sin_theta

        dvel = self.d_theta * (0.5 * v_integral_by_dphi[0] + v_integral_by_dphi[1:self.THETA_SIZE - 1].sum() + 0.5 * v_integral_by_dphi[self.THETA_SIZE - 1])
        dpsi = self.d_theta * (0.5 * psi_integral_by_dphi[0] + psi_integral_by_dphi[1:self.THETA_SIZE - 1].sum() + 0.5 * psi_integral_by_dphi[self.THETA_SIZE - 1])
        dv_z = self.d_theta * (0.5 * v_z_integral_by_dphi[0] + v_z_integral_by_dphi[1:self.THETA_SIZE - 1].sum() + 0.5 * v_z_integral_by_dphi[self.THETA_SIZE - 1])

        dvel += self.GAM * (self.V0 - self.velocity_norm)

        return dvel, dpsi, dv_z

    def psi_to_bounds(self, psi):
        while psi>np.pi:
            psi -= 2*np.pi
        while psi<=-np.pi:
            psi += 2*np.pi
        return psi

    def updateVelocity(self, dvel: float, dpsi: float, dvz: float) -> None:
        self.velocity_norm += min(dvel * self.d_t, MAX_XY_VEL)
        self.psi = max(-MAX_ROT_VEL, min(MAX_ROT_VEL, self.psi_to_bounds(self.psi + dpsi * self.d_t)))
        # print("velocity before", self.velocity, self.psi)
        self.velocity[0] = self.velocity_norm * np.cos(self.psi)
        self.velocity[1] = self.velocity_norm * np.sin(self.psi)
        self.velocity[2] = max(-MAX_Z_VEL, min(MAX_Z_VEL, self.velocity[2] + dvz * self.d_t))
        # print("velocity after", self.velocity)
        return

    def updatePositon(self) -> None:
        self.x += self.velocity[0] * self.d_t
        self.y += self.velocity[1] * self.d_t
        self.z += self.velocity[2] * self.d_t
        # print(f"x: {self.x:.5f}, y: {self.y:.5f}, z: {self.z:.5f}")
        return

    def setZeroVisualField(self) -> None:
        self.V.setZero()
        return

    def updateVFieldBasedOnDroneCoords(self, x: float, y: float, z:float) -> None:
        """
        input xyz are global coords of newly observed drone that I want to push into visual field  
        """
        relative_xy = np.array([x - self.x, y - self.y])
        relative_z = z - self.z
        dist = np.sqrt(pow(relative_xy[0],2) + pow(relative_xy[1],2) + pow(relative_z,2))
        Rot_psi = np.array([[np.cos(self.psi), -np.sin(self.psi)],
                            [np.sin(self.psi), np.cos(self.psi)]])
        xy_local = Rot_psi @ relative_xy
        phi = np.arctan2(xy_local[1], xy_local[0])
        theta = np.arcsin(relative_z / dist)
        alpha = np.arctan(R/dist)
        if UPDATE_FIELD_ONLY_CLOSER_THAN_10 and dist<=10.:
            self.V.setSphereCap(phi, theta, alpha)
        else:
            self.V.setSphereCap(phi, theta, alpha)
        return

class simulation:
    def __init__(self, SIM_RATE: int, N_DRONES: int =N_DRONES, SIM_TIME: float =SIM_TIME, SIM_START_COLLECT_DATA: float =SIM_START_COLLECT_DATA, SHOW: bool =False, alp0: float=ALP0, bet0: float=BET0, lam0: float=LAM0) -> None:
        self.SIM_RATE = SIM_RATE
        self.N_DRONES = N_DRONES
        self.SIM_TIME = SIM_TIME
        self.SIM_START_COLLECT_DATA = SIM_START_COLLECT_DATA
        self.SHOW = SHOW
        self.ALP0 = alp0
        self.BET0 = bet0
        self.LAM0 = lam0
        self.drones = self.spawn_drones(n_drones=N_DRONES)
        self.minDist = float('inf')
        self.avgMinDist = 0.
        self.cntMinDist = 0
        self.polarization = 0.
        self.avgDist = 0.
        self.counter = 0
    
    def spawn_drones(self, n_drones) -> list:
        drones_list = []
        for i in range(n_drones):
            x = i*3
            y = random.uniform(-3, 3)
            z = random.uniform(0, 5)
            drones_list.append(drone(x, y, z, np.pi/2, 
                    PHI_SIZE, THETA_SIZE,
                    GAM, V0, R, SIM_RATE,
                    self.ALP0, ALP1_LR, ALP1_UD, 
                    self.BET0, BET1_LR, BET1_UD, 
                    self.LAM0, LAM1_LR, LAM1_UD))

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
        ax.set_xlim([-250, 250])
        ax.set_ylim([-250, 250])
        ax.set_zlim([-20, 20])

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
            
            if USE_BOUNDARY_BOX:

                if x_i >= BOX_WIDTH/2:
                    drone.V.setSphereCap(phi_center=0., theta_center=0., alpha=BLACKEN_V_FIELD)
                elif x_i <= -BOX_WIDTH/2:
                    drone.V.setSphereCap(phi_center=np.pi, theta_center=0., alpha=BLACKEN_V_FIELD)

                if y_i >= BOX_LENGTH/2:
                    drone.V.setSphereCap(phi_center=np.pi/2, theta_center=0., alpha=BLACKEN_V_FIELD)
                elif y_i <= -BOX_LENGTH/2:
                    drone.V.setSphereCap(phi_center=-np.pi/2, theta_center=0., alpha=BLACKEN_V_FIELD)
                
                if z_i >= BOX_HEIGHT+BOX_DIST_FROM_GROUND:
                    drone.V.setSphereCap(phi_center=0., theta_center=np.pi/2, alpha=BLACKEN_V_FIELD)
                elif z_i <= BOX_DIST_FROM_GROUND:
                    drone.V.setSphereCap(phi_center=0., theta_center=-np.pi/2, alpha=BLACKEN_V_FIELD)
    
    def updateDronesPosition(self):
        for drone in self.drones:
            drone.updatePositon()

    def updateDronesStateVar(self):
        for drone in self.drones:

            if USE_GPU:
                dvel, dpsi, dv_z = drone.compute_state_variables_3d_on_gpu()
            else:
                dvel, dpsi, dv_z = drone.compute_state_variables_3d_on_cpu()
            # print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}")

            # start_time = time.time()
            # dvel, dpsi, dv_z = drone.compute_state_variables_3d_on_gpu()
            # end_time = time.time()
            # old_ver_time = end_time-start_time
            # print(f"compute_state_variables_3d_gpu executed in {old_ver_time:.5f} seconds. dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}")

            # start_time = time.time()
            # dvel, dpsi, dv_z = drone.compute_state_variables_3d_on_cpu()
            # end_time = time.time()
            # new_ver_time = end_time-start_time
            # print(f"compute_state_variables_3d_cpu executed in {new_ver_time:.5f} seconds. dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}")
            # print(f"cpu version is this much worse: {new_ver_time-old_ver_time:.5f}")



            drone.updateVelocity(dvel, dpsi, dv_z)

    def getAllVel(self):
        velocities = np.empty((self.N_DRONES, 3)) 

        for i, drone in enumerate(self.drones):
            velocities[i] = drone.velocity
        
        return velocities
    
    def calcPol(self):
        velocities = self.getAllVel()
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)  # Compute norm for each row
        unit_velocities = velocities / norms 
        mean_heading = np.mean(unit_velocities, axis=0)
        pol = np.linalg.norm(mean_heading)
        return pol

    def getMinDistInfoAndPol(self):
        x, y, z = self.getDronesPositions()
        n = len(x)

        min_distances = np.zeros(n)

        now_observed_min_dist = float('inf')

        avgDist = 0.

        for i in range(n):
            min_dist = float('inf')  # Start with a large number
            for j in range(n):
                if i != j:  # Skip the same point
                    dist = math.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)
                    avgDist += dist
                    min_dist = min(min_dist, dist)
            now_observed_min_dist = min(now_observed_min_dist, min_dist)
            min_distances[i] = min_dist
        self.minDist = min(self.minDist, now_observed_min_dist)
        self.avgMinDist = (self.avgMinDist*self.cntMinDist + min_distances.sum())/(self.cntMinDist+n)
        self.cntMinDist += n

        avgDist /= n

        self.polarization = (self.polarization*self.counter + self.calcPol())/(self.counter+1)
        self.avgDist = (self.avgDist*self.counter + avgDist)/(self.counter+1)
        # print(self.avgDist, avgDist)
        self.counter += 1

        if(self.polarization>1):
            print(self.polarization, "EXTREM")

    def writeCsv(self):
        drone1 = self.drones[0]
        data = [drone1.ALP0, drone1.ALP1_LR, drone1.ALP1_UD, drone1.BET0, drone1.BET1_LR, drone1.BET1_UD, drone1.LAM0, drone1.LAM1_LR, drone1.LAM1_UD, self.minDist, self.avgMinDist, self.polarization, self.avgDist]
        with open('data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def simulate(self):
        dt = 1/self.SIM_RATE
        t = 0
        if self.SHOW:
            self.startPlotDrones()
        while t<self.SIM_TIME:
            t += dt
            # print("Simulation time right now is: ", t, " s")

            # start_time = time.time()
            self.updateVFieldForDrones()
            # end_time = time.time()
            # print(f"updateVFieldForDrones executed in {end_time-start_time:.6f} seconds.")

            # start_time = time.time()q
            self.updateDronesStateVar()
            # end_time = time.time()
            # print(f"updateDronesStateVar executed in {end_time-start_time:.6f} seconds.")

            # start_time = time.time()
            self.updateDronesPosition()
            # end_time = time.time()
            # print(f"updateDronesPosition executed in {end_time-start_time:.6f} seconds.")

            # for i, drone in enumerate(self.drones):
            #     print("V_field of drone number: ", i)
            #     drone.V.plotVisualField()

            if self.SHOW:
                self.updatePlotDrones()
            if t>=self.SIM_START_COLLECT_DATA:
                self.getMinDistInfoAndPol()
        self.writeCsv()
                              
class VisualField: 
    """
    spherical representation of the visual field as a rows x cols -> theta x phi discretization
    
    """
    def __init__(self, phi_size: int, theta_size: int, simplet_V_field: bool =False) -> None:
        self.field = np.zeros((theta_size, phi_size), dtype=np.float32)
        self.theta_size = theta_size #rows
        self.phi_size = phi_size #cols
        self.d_phi = 2*PI/phi_size
        self.d_theta = PI/theta_size
        self.simple_V_Field = simplet_V_field
    
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
        # self.field[row_start:row_end, col_start:col_end] = 1
        for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    phi = self.colToPhi(col)
                    theta  =self.rowToTheta(row)
                    x, y, z = self.sphericalToCartesian(self.phiShift(phi), self.thetaShift(theta))
                    dot_product = x*x_c + y*y_c + z*z_c
                    distance = np.acos(dot_product)
                    if distance <= alpha:
                        self.field[row, col] = 1
        return

    def setSphereCap(self, phi_center: float, theta_center: float, alpha: float) -> None: #TESTED
        """
        phi_center - angle in xy plane in drone frame of reference
        theta_center - angle from xy to the center of the cap
        alpha - angle from the center of the cap to the edge of the cap
        """
        if self.simple_V_Field:
            theta_min = self.thetaToRow(self.thetaToRange(theta_center-alpha))
            theta_max = self.thetaToRow(self.thetaToRange(theta_center+alpha))
            if theta_min==0 or theta_max==self.theta_size-1:
                phi_min = 0
                phi_max = self.phi_size-1
            else:
                phi_min = self.phiToCol(phi_center-alpha)
                phi_max = self.phiToCol(phi_center+alpha)

            if phi_min<phi_max:
                self.field[theta_min:theta_max, phi_min:phi_max] = 1
            else:
                self.field[theta_min:theta_max, phi_min:self.phi_size-1] = 1
                self.field[theta_min:theta_max, 0:phi_max] = 1
            return

        x_center, y_center, z_center = self.sphericalToCartesian(self.phiShift(phi_center), self.thetaShift(theta_center))
        theta_min = self.thetaToRow(self.thetaToRange(theta_center-alpha))
        theta_max = self.thetaToRow(self.thetaToRange(theta_center+alpha))
        if PI_2-theta_center<=alpha or -PI_2-theta_center<=alpha:
            phi_min = 0
            phi_max = self.phi_size-1
        else: 
            phi_min = self.phiToCol(phi_center-alpha)
            phi_max = self.phiToCol(phi_center+alpha)
        # print(phi_min, phi_max)
        # print(theta_min, theta_max)
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


def test(x, y, z):
    # phisize = 1024
    # thetasize = 256

    phisize = 256
    thetasize = 128
    drone1 = drone(0, 0, 0, 0, 
                    PHI_SIZE=phisize, THETA_SIZE=thetasize,
                    GAM=GAM, V0=V0, R=R, SIM_RATE=SIM_RATE,
                    ALP0=1, ALP1_LR=0.08, ALP1_UD=0., 
                    BET0=1, BET1_LR=0.08, BET1_UD=0., 
                    LAM0=1, LAM1_LR=0.08, LAM1_UD=0.08)
    drone1.updateVFieldBasedOnDroneCoords(x*3, y*3, z*3)
    dvel, dpsi, dvz = drone1.compute_state_variables_3d()
    # drone1.setZeroVisualField()
    # drone1.updateVFieldBasedOnDroneCoords(x*10, y*10, z*10)
    # dvel, dpsi, dvz = drone1.compute_state_variables_3d()

    drone1.V.plotVisualField()

def test_wat_dronedoin():
    sim = simulation(SIM_RATE=SIM_RATE, N_DRONES=1, SIM_TIME=SIM_TIME, SIM_START_COLLECT_DATA=SIM_START_COLLECT_DATA, SHOW=True, alp0=2., bet0=5, lam0=1)

    sim.drones[0].V.setSphereCap(phi_center=0., theta_center=0., alpha=BLACKEN_V_FIELD)
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_cpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the front of the drone")
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_gpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the front of the drone gpu")
    sim.drones[0].V.plotVisualField()
    input()
    sim.drones[0].V.setZero()

    sim.drones[0].V.setSphereCap(phi_center=np.pi, theta_center=0., alpha=BLACKEN_V_FIELD)
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_cpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the back of the drone")
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_gpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the front of the drone gpu")
    sim.drones[0].V.plotVisualField()
    input()
    sim.drones[0].V.setZero()

    sim.drones[0].V.setSphereCap(phi_center=np.pi/2, theta_center=0., alpha=BLACKEN_V_FIELD)
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_cpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the left of the drone")
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_gpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the front of the drone gpu")
    sim.drones[0].V.plotVisualField()
    input()
    sim.drones[0].V.setZero()

    sim.drones[0].V.setSphereCap(phi_center=-np.pi/2, theta_center=0., alpha=BLACKEN_V_FIELD)
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_cpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the right of the drone")
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_gpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the front of the drone gpu")
    sim.drones[0].V.plotVisualField()
    input()
    sim.drones[0].V.setZero()

    sim.drones[0].V.setSphereCap(phi_center=0., theta_center=np.pi/2, alpha=BLACKEN_V_FIELD)
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_cpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the up of the drone")
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_gpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the front of the drone gpu")
    sim.drones[0].V.plotVisualField()
    input()
    sim.drones[0].V.setZero()

    sim.drones[0].V.setSphereCap(phi_center=0., theta_center=-np.pi/2, alpha=BLACKEN_V_FIELD)
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_cpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the down of the drone")
    dvel, dpsi, dv_z = sim.drones[0].compute_state_variables_3d_on_gpu()
    print(f"dvel: {dvel:.5f}, dpsi: {dpsi:.5f}, dvz: {dv_z:.5f}, for the front of the drone gpu")
    sim.drones[0].V.plotVisualField()
    input()
    sim.drones[0].V.setZero()


if __name__ == "__main__":
    # test_wat_dronedoin()
    # print(BOX_WIDTH/2, BOX_LENGTH/2, BOX_HEIGHT+BOX_DIST_FROM_GROUND, BOX_DIST_FROM_GROUND)
    # exit()


    alp0 = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    bet0 = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    lam0 = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    # sim = simulation(SIM_RATE=SIM_RATE, N_DRONES=10, SIM_TIME=SIM_TIME, SIM_START_COLLECT_DATA=SIM_START_COLLECT_DATA, SHOW=True, alp0=2., bet0=5, lam0=1)
    # sim.simulate()
    # exit()

    combinations = list(itertools.product(alp0, bet0, lam0))
    for alpha, beta, lam in tqdm(combinations):
        sim = simulation(SIM_RATE=SIM_RATE, N_DRONES=10, SIM_TIME=SIM_TIME, SIM_START_COLLECT_DATA=SIM_START_COLLECT_DATA, SHOW=False, alp0=alpha, bet0=beta, lam0=lam)
        sim.simulate()
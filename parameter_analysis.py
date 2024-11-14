import numpy as np
import math
import logging as log

ALP0 = 1
ALP1_LR = 0.08
ALP1_UD = 0.08
BET0 = 0.5
BET1_LR = 0.08
BET1_UD = 0.08
LAM0 = 1
LAM1_LR = 0.08
LAM1_UD = 0.08

PHI_SIZE: 1024
THETA_SIZE: 128

GAM = 0.1
V0 = 1

PI = math.pi

class drone: 
    def __init__(self, x: float, y: float, z: float, phi:float =None) -> None:
        self.x = x #global coords
        self.y = y 
        self.z = z

        self.V = VisualField(PHI_SIZE, THETA_SIZE) 

        if phi==None:
            self.phi = 0.0
        else:
            self.phi = phi
        


        


class VisualField: 
    """
    spherical representation of the visual field as a rows x cols -> theta x phi discretization
    
    """
    def __init__(self, phi_size: int, theta_size: int) -> None:
        self.field = np.zeros((theta_size, phi_size))
        self.rows = theta_size
        self.cols = phi_size

    def setZero(self) -> None:
        self.field.fill(0)
    
    def phiToRange(phi:float):
        while phi<-PI:
            phi += 2*PI
        while phi>=PI:
            phi -= 2*PI
        return phi

    def phiToCol(self, phi:float) -> int:
        if phi<-PI or phi>=PI:
            
        col = 
        if(col<0 or col>=self.cols): #DEBUG
            log.error("Calculated col outside range of field")
            return 0
        return col

    def colToPhi() -> float:
        pass
    
    def thetaToRow() -> int:
        pass

    def rowToTheta() -> float:
        pass

    
    def set_sphere_cap(self, phi_center: float, theta_center: float, alpha: float) -> None:
        """
        phi_center - angle in xy plane in drone frame of reference
        theta_center - angle from xy to the center of the cap
        alpha - angle from the center of the cap to the edge of the cap
        """
        int 


        




if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

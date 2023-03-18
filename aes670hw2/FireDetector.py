
import numpy as np

class FireDetector:
    """
    Executes fire detection algorithm from (Flasse and Ceccato, 1996)
    on arbitrary data grids of brightness temperatures in LWIR and SWIR bands
    and calibrated reflectances (bi-directional reflectance factor) in NIR

    Band : Wavelengths  : Unit
    ------------------------------------------------
    LWIR : 10.3-11.3 um : Brightness temperature (K)
    SWIR : 3.55-3.93 um : Brightness temperature (K)
    NIR  : 0.73-1.00 um : Calibrated reflectance
    """
    def __init__(self):
        #
        self.swir_thresh_lbound = 311 # K
        # Pixels are valid if the LWIR and SWIR are different by 8K;
        # ~3.7um is near the Wein's law peak for fire temperatures.
        self.swir_lwir_diff_lbound = 8 # K
        # Reflectance threshold above which a pixel is just considered
        # a highly reflective surface
        self.nir_ref_ubound = .2
        self._candidates = None

    @property
    def candidates(self):
        return self._candidates

    def _get_candidates(self, LWIR, SWIR, NIR):
        """
        """
        data = [LWIR, SWIR, NIR]
        conditions = [
            SWIR > self.swir_thresh_lbound,
            SWIR-LWIR > self.swir_lwir_diff_lbound,
            NIR < self.nir_ref_ubound
            ]
        print(conditions[0])
        print(np.count_nonzero(conditions[0]),
              np.count_nonzero(conditions[1]),
              np.count_nonzero(conditions[2]))
        self._candidates = np.where(
                np.logical_and(conditions[0], np.logical_and(
                    *conditions[1:])))

    def get_candidates(self, LWIR:np.ndarray, SWIR:np.ndarray, NIR:np.ndarray,
                   debug=False):
        """
        Returns True if candidate pixels are found, and stores any candidate
        pixels to the self.candidates property.
        """
        self._get_candidates(LWIR, SWIR, NIR)
        if not len(self._candidates):
            if debug: print("No fires detected.")
            return False
        if debug: print(f"{len(self._candidates[0])} candidate pixels found")
        return True

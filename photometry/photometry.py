'''
@author: S. N. Hasler

Photometry functions for use with deconfuser.
'''
import numpy as np
import astropy.constants as const
import astropy.units as u
import ast
import os
import re

'''
TODO:
- Add equation to compute leakage 
- Add equation to compute zodiacal & exozodiacal light 
'''

class System:
    def __init__(self, n_exozodi=0.0, n_leakage=0.0, n_zodi=0.0, name=None):
        '''
        Holds system parameters.

        Parameters
        ----------
        n_exozodi : optional, float
            Number of counts per second contributed to the background counts
            due to exozodi source. Default does not account for exozodi.
        n_leakage : optional, float
            Number of counts per second contributed to the background counts
            due to leakage. Default does not account for leakage.
        n_zodi : optional, float
            Number of counts per second contributed to the background counts
            due to zodiacal light. Default does not account for zodi.
        name : optional, int or string
            Name or number corresponding to planetary system
        '''
        self.n_exozodi = n_exozodi
        self.n_leakage = n_leakage
        self.n_zodi = n_zodi

        self.name = name 
        self.planets = []

    def add_planet(self, Planet):
        '''
        Add planet to the system. 
        Each planet contains the info for radius, albedo, etc.

        Parameters
        ----------
        Planet : photometry.Planet
            Planet object (see below)
        '''
        self.planets.append(Planet)

    def add_star(self, Star):
        '''
        Add star to the system.

        Parameters
        ----------
        Star : photometry.Star
            Star object (see below)
        '''
        self.star = Star

class Star:
    def __init__(self, T, R_star, d_system, mu):
        '''
        Holds system's star parameters.

        Parameters
        ----------
        T : float
            Stellar effective temperature [units: K]
        R_star : float
            Stellar radius [units: m]
        d_system : float
            Distance of system from observer [units: parsecs]
        mu : float
            Stellar gravitational parameter in units of AU^3 / yr^2 (for consistency with deconfuser)
        '''
        self.T = T * u.K
        self.R_star = R_star * u.m 
        self.d_system = d_system * u.parsec
        self.mu = mu * (u.AU**3 / u.yr**2)

    def blackbody_spec(self, wavelength):
        '''
        Calculate blackbody spectrum value of star at given wavelength. 

        Parameters
        ----------
        wavelength : numpy.ndarray or numpy.float64
            Wavelength or wavelength range to consider [m].

        Returns
        -------
        B_lambda_star : numpy.ndarray or numpy.float64
            Blackbody spectrum value of star at wavelength of interest.

        '''
        # Make sure wavelength has astropy units and make sure the units are in meters
        if not isinstance(wavelength, u.quantity.Quantity):
            raise ValueError("Input wavelength must be an astropy Quantity with units of meters.")
        if wavelength.unit != u.m:
            wavelength = wavelength.to(u.m)

        # Constants
        h = const.h          # Planck's constant
        c = const.c          # speed of light
        k = const.k_B        # boltzmann constant

        num = (2 * h * c**2)
        denom = wavelength**5 * (np.exp((h * c) / (wavelength * k * self.T)) - 1)

        B_lambda_star = (( num / (denom) )).to(u.Watt / u.m**2 / u.um) # blackbody spectrum of star [ units: W / m^2 / um ] 

        self.B_star = B_lambda_star
        
        return B_lambda_star
    
    def stellar_flux(self, B_lambda_star):
        '''
        Calculate flux contribution from star. 

        Parameters
        ----------
        B_lambda_star : numpy.ndarray or numpy.float64
            Blackbody spectrum of star. Generated with blackbody_spec()

        Returns
        -------
        F_star : numpy.ndarray or numpy.float64
            Stellar flux density.

        '''
        system_distance = self.d_system.to(u.m) # convert to meters
        F_star = ((np.pi * B_lambda_star * ( self.R_star / system_distance )**2)) 
        self.F_star = F_star

        return F_star
    
    # def read_spectrum(self, spectrum_file):
    #     '''
    #     Function to read in stellar spectrum from a file.
    #     File should be in two column format with wavelength in first 
    #     column and spectrum in second. 

    #     Parameters
    #     ----------
    #     spectrum_file : str
    #         Full path to file with stellar spectrum.
        
    #     Returns
    #     -------
    #     wavelength : np.array
    #         Array of wavelength values for spectrum
    #     spectrum : np.array
    #         Array of stellar spectrum values
    #     '''

    #     try:
    #         wavelength, spectrum = np.loadtxt(spectrum_file, unpack=True, usecols=(0,1))

    #         self.wavelength = wavelength
    #         self.spectrum = spectrum
    #         return wavelength, spectrum

    #     except (IndexError, ValueError, SyntaxError) as e:
    #         raise ValueError("Error processing file: ensure it is in the correct format.") from e     
    
    def flux_in_filter(self, filter_lambda, filter_transmission, lambda_min, lambda_max):
        '''
        Convolve stellar spectrum with filter bandpass to get flux in filter bandpass.

        Parameters
        ----------
        filter_lambda : np.ndarray
            Wavelength array of filter bandpass in same units
            as stellar spectrum
        filter_transmission : np.ndarray
            Filter transmission array with same length as lambda_filter
        lambda_min : float
            Minimum wavelength of filter bandpass
        lambda_max : float
            Maximum wavelength of filter bandpass

        Returns
        -------
        flux : np.ndarray
            Integrated band flux of star in filter bandpass in same units as stellar spectrum
        '''
        # Interpolate solar spectrum to filter wavelength grid
        spectrum_in_filter = self.interp_spectrum(filter_lambda)

        # Calculate filter flux
        filter_flux = np.trapz(spectrum_in_filter * filter_transmission, filter_lambda) / np.trapz(filter_transmission, filter_lambda)

        return filter_flux
        
class Planet:
    def __init__(self, R_p=6378e3, Ag=0.3):
        '''
        Planet object 

        Parameters
        ----------
        R_p : float, optional
            Planet's radius [m], default = Earth radius
            Can also assign in choose_random_Rp() or random_planet_params()
        Ag : float, optional
            geometric albedo, default = Earth Ag
            Can also assign in random_planet_params()
        '''
        self.R_p = R_p * u.m
        self.Ag = Ag

        self.type = None            # unassigned planet type
        self.phase_angles = []      # list to append phase angles of detections to
        self.band_fluxes = []       # empty list to append filter dict info to

    def choose_random_Rp(self, R_min, R_max, n_planets):
        '''
        Choose random planet radii from a uniform distribution
        Radii in units of R_Earth

        Parameters
        ----------
        R_min : float
            Minimum radius to sample from in units of m
        R_max : float
            Maximum radius to sample from in units of m
        n_planets : int
            Number of planets to sample radii for
        Returns 
        -------
        Rp : np.ndarray
            Array of length n_planets with planet radii in R_Earth
        '''
        Rp = np.random.uniform(R_min, R_max, n_planets)
        self.Rp = Rp * u.m
        return Rp
    
    def random_planet_params(self, planet_types=['rocky', 'gas_giant', 'ice_giant'],
                             assign_type_by_sep=True):
        '''
        Choose a random planet type from a list and assign it a radius value based on type.
        Based on planet type, pull the best matching albedo spectrum (based on type and separation)

        Parameters
        ---------
        planet_types : list of string
            List of planet types to choose frome
        assign_type_by_sep : bool
            Whether or not to assign planet type based on its orbital separation.
            Set to true if you want to limit rocky planets to within ~0.5-2.0 AU, giant planets between 0.5-7.0 AU,
            and ice giants between 0.5 - 11 AU. 
            These limits are set based on available spectra at time of writing.
        
        TODO: update this function to choose planet type based on orbital separation
        '''
        
        R_Earth = 6378e3 * u.m # Earth radius in m

        # Choose a random planet type to assign 
        if assign_type_by_sep:
            if self.a > 2.0 and 'rocky' in planet_types: # don't assign rocky to planets outside 2 AU
                planet_types.remove("rocky")
        type = np.random.choice(planet_types)
        self.type = type

        # Assign radius based on planet type
        if type == 'rocky':
            self.R_p = 1.0 * R_Earth 
            self.Ag = 0.3 
        elif type == 'ice_giant':
            self.R_p = 3.9 * R_Earth      # ~R_Uranus
            self.Ag = 0.49
        elif type == 'gas_giant':
            self.R_p = 11.0 * R_Earth     # ~R_Jup
            self.Ag = 0.53
            
    def random_Ag(self, Ag_min, Ag_max, n_planets):
        '''
        Choose random geometric albedo values from a uniform distribution

        Parameters
        ----------
        Ag_min : float
            Minimum Ag value to sample from
        Ag_max : float
            Maximum Ag value to sample from
        n_planets : int
            Number of planets to sample Ag for
        '''
        Ag = np.random.uniform(Ag_min, Ag_max, n_planets)
        self.Ag = Ag

        return Ag
    
    def read_albedo_spectrum(self, spectrum_file, lambda_units=u.um, save_to_object=False):
        '''
        Function to read in planet spectrum from a file.
        Spectrum should be in some text file with first column
        giving wavelength values and second giving albedo spectrum

        Parameters
        ----------
        spectrum_file : str
            Full path to file with planet spectrum.
        lambda_units : astropy.units.core.PrefixUnit
            Astropy units for wavelength of planet spectrum.

        Returns
        -------
        wavelength : np.array
            Array of wavelength values for spectrum
        spectrum : np.array
            Array of planet spectrum values
        '''
        try:
            wavelength, spectrum = np.loadtxt(spectrum_file, unpack=True)

            if save_to_object:
                self.wavelength = wavelength * lambda_units
                self.Ag = spectrum
            
            return wavelength * lambda_units, spectrum

        except (IndexError, ValueError, SyntaxError) as e:
            raise ValueError("Error processing file: ensure it is in the correct format.") from e     

    def add_orb_params(self, a, e, i, o, O, M0):
        '''
        Add orbital parameters to planet object.

        Parameters
        ----------
        a : float
            Semi-major axis [AU]
        e : float
            Eccentricity
        i : float 
            Inclination [rad]
        o : float
            Argument of periapsis [rad] 
        O : float 
            Argument of ascending node [rad]
        M0 : float
            mean anomaly [rad]
        '''
        self.a = a
        self.e = e
        self.i = i
        self.o = o
        self.O = O
        self.M0 = M0

    def reset_spectra_arrays(self):
        '''
        Resets the arrays in Planet object which hold the spectra and wavelength arrays.
        '''
        self.wavelength = []
        self.Ag = []

def get_alb_spectra(Planet, spectrum_dir): # TODO: move to System class structure?
        '''
        Get albedo spectrum based on the assigned planet type and its
        orbital separation. Albedo spectrum files must contain a 
        substring of the format "1.5AU" or similar to find match.
        Planets must have assigned type and radius for this to work.

        Parameters
        ----------
        System : photometry.System
            System object containing planets 
        spectrum_dir : string
            Path to directory containing folders with planet spectra
        '''
        # n_planets = len(System.planets) # Get number of planets in system

        # # For each planet, pull the spectrum and add it to the planet object
        # for i in range(n_planets):
        #     # Get planet object from system object
        #     planet = System.planets[i]
        a = Planet.a 
        phase_angles = np.abs(Planet.phase_angles)

        # Go to the directory of the matching planet type
        filepath = os.path.join(spectrum_dir + f"{Planet.type}/")
        files = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))] # get files in folder

        # If more than one file, find the one with the nearest semimajor axis and phase angle by
        # searching the file names which should contain "XAU" and "Ydeg"
        a_pattern = re.compile(r"_([\d.]+)AU_")
        phase_pattern = re.compile(r"_([0-9]+)deg")
        usable_file_info = []
        for f in files:
            a_match = a_pattern.search(f)
            phase_match = phase_pattern.search(f)
            if a_match and phase_match:
                a_val = float(a_match.group(1))
                phase_val = float(phase_match.group(1))
                usable_file_info.append((f, a_val, phase_val))
        # For each phase angle, find best matching file
        spectrum_filenames = []
        for angle in phase_angles:
            try:
                best_file = min(usable_file_info, key=lambda x: (abs(x[1] - a), abs(x[2] - angle)))[0]
                spectrum_filenames.append(best_file)
            except ValueError:
                print("No matching files...")
                spectrum_filenames.append(None)
        
        # Save list of selected files and read in each spectrum
        Planet.spectrum_file = spectrum_filenames
        # Reset planet albedo and wavelength to save multiple spectra to
        Planet.Ag = []
        Planet.wavelength = []
        for f in spectrum_filenames:
            if f is not None:
                wavelength, spectrum = Planet.read_albedo_spectrum(os.path.join(filepath, f), lambda_units=u.um, 
                                                                    save_to_object=False)
                Planet.Ag.append(spectrum)
                Planet.wavelength.append(wavelength)
            else:
                print(f"No matching albedo spectrum to use. Add corresponding albedo spectrum to {filepath}")

class Filters:
    def __init__(self, filter_file):
        '''
        Class for the filter curves of the detector.

        Parameters
        ----------  
        filter_file : str
            Path to filter file.

        '''
        self.filter_file = filter_file

    def read_filter_curves(self):
        '''
        Function to read in filter curves for the detector. 
        Filters should have min/max wavelength for bandpass and throughput: e.g., 
            roman_bands = {"575":{"min_lambda": 0.546, "max_lambda": 0.603, "thru": 0.01},
                           "659":{"min_lambda": 0.604, "max_lambda": 0.715, "thru": 0.01}}

        Parameters
        ----------
        filter_file : str
            Path to txt file with filter info.
        
        Returns
        -------
        filters : dict
            Dictionary of filter curves.

        '''
        with open(self.filter_file, 'r') as file:
            data = file.read()
            
        try:
            dict_str = data.split('=', 1)[1].strip()
            filters = ast.literal_eval(dict_str)
        except (IndexError, ValueError, SyntaxError) as e:
            raise ValueError("Error processing file: ensure it is in the correct format.") from e

        self.filters = filters # Add filters dict to detector object

        return filters
    
    def lambda_range(self, filter_name, filters):
        '''
        Function to get the min/max wavelengths of a filter from the filter dictionary.
        
        Parameters
        ----------
        filter_name : str
            Name of filter to get min/max wavelengths for.

        filters : dict
            Dictionary of filter curves.

        Returns
        -------
        min_lambda : float
            Minimum wavelength of passband

        max_lambda : float
            Maximum wavelength of passband
        
        '''
        # Get min/max wavelengths from passbands
        min_lambda = np.min([filters[filter_name]['min_lambda']])
        max_lambda = np.max([filters[filter_name]['max_lambda']])

        return min_lambda, max_lambda

    def filter_throughput(self, filter_name, wavelength):
        '''
        Set throughput of filter on wavelength region of interest.
        Use if you do not have a filter curve to read in.

        Parameters
        ----------
        filter_name : str
            Name of filter to get throughput for.
        wavelength : np.ndarray
            Wavelength of object spectrum
        '''
        f = wavelength * 0
        f[(wavelength > self.filters[filter_name]['min_lambda']) & (wavelength < self.filters[filter_name]['max_lambda'])] = self.filters[filter_name]['thru']

        return f

def spectrum_in_filter(wavelength, spectrum, min_lambda, max_lambda):
    '''
    Get the spectrum of the object in the filter bandpass

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength of spectrum [TODO: add units]
    spectrum : np.ndarray
        Spectrum of object [TODO: add units]

    Returns
    -------
    filter_spec : np.ndarray
        Spectrum of object in filter bandpass [TODO: add units]
    filter_wavel : np.ndarray   
        Wavelength of spectrum in filter bandpass [TODO: add units]
    '''

    # Select spectrum region within filter band
    filter_spec = spectrum[(wavelength >= min_lambda) & (wavelength <= max_lambda)] 
    filter_wavel = wavelength[(wavelength >= min_lambda) & (wavelength <= max_lambda)]

    return filter_spec, filter_wavel

def compute_color(filter_arr, albedo_spectrum, wavelength, stellar_flux):
    '''
    Compute the integral to calculate the color in one filter.

    Parameters
    ----------
    filter_arr : np.ndarray
        Filter throughput array
    albedo_spectrum : np.ndarray
        Albedo spectrum of planet in filter
    wavelength : np.ndarray
        Wavelength array of planet spectrum in filter
    stellar_flux : np.ndarray
        Stellar flux interpolated to planet wavelength grid

    Returns
    -------
    f_int : float
        "Color" in one filter
    '''
    f_int = np.sum(filter_arr[:-1] * albedo_spectrum[:-1] * np.diff(wavelength) * stellar_flux[:-1])

    return f_int

# def compare_bands(f1_int, f2_int):
#     '''
#     Compare color in two filters

#     Parameters
#     ----------
#     f1_int : float
#         Color in first filter
#     f2_int : float
#         Color in second filter

#     Returns
#     -------
#     f1_f2 : float
#         Color comparison
#     '''
#     f1_f2 = -2.5 * np.log10( f1_int / f2_int)
#     return f1_f2

class Detector:
    '''
    Class for the detector parameters. 
    '''
    def __init__(self, qe, cic, dark_current, read_noise, gain, fwc, conversion_gain, t,
                 D, throughput, f_pa, wavelength, bandwidth):
        '''
        Detector parameters

        Parameters
        ----------
        qe : float
            Quantum efficiency of detector. 
        cic : float
            Clock-induced charge of detector [e-]
        dark_current : float
            Detector dark current [e-/sec]
        read_noise : int
            Read noise of detector [e-/sec]
        gain : int
            Gain of emccd detector
        fwc : int
            Fell-well capacity of detector
        conversion_gain : float
            Conversion value for e- to ADU
        t : int
            Integration time. [s]
        D : float
            Main aperture diameter [m]
        throughput : float
            Total throughput (generally wavelength dependent). For calculating planet photon rate at detector.
        f_pa : float
            Describes the fraction of light from the planet that falls within the photometric aperture (see Robinson+2016, Eqn 12)
            For calculating planet photon rate at detector.
        wavelength : float
            Wavelength of observation [m]
        bandwidth : float
            bandwidth of wavelength band [m]
        '''
        self.qe = qe
        self.cic = cic
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.gain = gain
        self.fwc = fwc
        self.conversion_gain = conversion_gain
        self.t = t 
        self.D = D * u.m
        self.throughput = throughput
        self.f_pa = f_pa
        self.wavelength = wavelength * u.m
        self.bandwidth = bandwidth * u.m

        self.filters = [] # empty list for appending filter info to

        self.calc_FWHM(self.wavelength, self.D)

    def add_filter_info(self, filter_file, filter_name, lambda_units=u.um):
        '''
        Function to add filter information to the detector object. 
        Reads in file with filter wavelength and transmission. 

        Parameters
        ----------
        filter_file : string
            Full path to filter file that contains wavelength array and
            filter transmission array
        filter_name : string
            Name of filter for reference
        lambda_units : astropy.units.core.PrefixUnit
            Units of the wavelength array in the filter transmission file

        '''
        filt_lambda, filt_transmission = np.loadtxt(filter_file, unpack=True)
        filt_lambda *= lambda_units
        # If wavelength not in um, convert
        if lambda_units != u.um:
            filt_lambda = filt_lambda.to(u.um)
            
        # Get filter wavelength range
        min_lambda, max_lambda = np.min(filt_lambda.value), np.max(filt_lambda.value)
        # Save filter info
        filt_dict = {"filter_name": filter_name, "min_lambda": min_lambda, "max_lambda": max_lambda,
                     "filter_lambda": filt_lambda.value, "filter_transmission": filt_transmission}
        # Add filter info to detector object
        self.filters.append(filt_dict)

    def get_filter_info(self, filter_name):
        '''
        Retrieve filter info from filters list of dicts.

        Parameters
        ----------
        filter_name : string
            Name of filter contained in filter 
        '''
        # Retrieve info for filter of observation
        filter_data = [f for f in self.filters if f['filter_name'] == filter_name][0]

        # Update bandwidth and wavelength accordingly
        self.bandwidth = ((filter_data['max_lambda'] - filter_data['min_lambda'])*u.um).to(u.m) # units: m
        self.wavelength = (((filter_data['max_lambda'] + filter_data['min_lambda']) / 2)*u.um).to(u.m) # central wavelength

        max_lambda = filter_data['max_lambda']
        min_lambda = filter_data['min_lambda']

        return filter_data, min_lambda, max_lambda

    def calc_FWHM(self, wavelength, D):
        '''
        Estimate the FWHM of a telescope system given the observing
        wavelength and aperutre diamter

        Parameters
        ----------
        wavelength : float
            Wavelength of observation [m]
        D : float
            aperture diameter [m]

        Returns
        -------
        float
            estimated FWHM in arcseconds
        '''
        FWHM_rad = (wavelength.value / D.value) * u.rad
        FWHM_as = FWHM_rad.to(u.arcsecond).value 
        self.FWHM = FWHM_as 

    def add_noise(self, detected_rate):
        '''
        Function to add detector noise to the count rate that reaches the detector.

        Parameters
        ----------
        detected_rate : float
            Photon count rate from object reaching the detector. [photons/s]
        
        Returns
        -------
        numpy.ndarray
            Output electron count from detector 
        '''
        N = detected_rate * self.t # [photons]
        input_photons = np.array([N])

        # Add shot noise
        with_shot_noise = np.random.poisson(input_photons)
        expected_counts = with_shot_noise * self.qe # [e-]

        # Add dark current
        base_dark_current = self.dark_current * self.t + self.cic # [e-/pxl]
        expected_counts = expected_counts + base_dark_current
        # add gain for emccd -- reference: https://github.com/nasa-jpl/lowfssim/blob/a76d89e3e6c5286674da490492ccc59f5b754965/lowfsc/emccd.py#L201
        # and https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0053671#pone.0053671-Basden1 (Eqn 16)
        k = expected_counts
        zero_mask = k == 0 # temporarily remove 0 to avoid divide by 0 errors
        k[zero_mask] = 1
        theta = self.gain - 1 + 1/k # gamma dist. scale parameter
        count_add_gain = np.random.gamma(k, theta) + k - 1 # gamma dist. for gain
        count_add_gain[zero_mask] = 0 # set values back to 0

        # read noise
        rn = np.random.normal(0, self.read_noise)
        with_noise = count_add_gain + rn

        # if noise > fwc, set to fwc
        with_noise[with_noise > self.fwc] = self.fwc
        with_noise /= self.conversion_gain  # [ADU]

        return with_noise, N # noisy counts, planet counts without noise
    
    def noise_distribution(self, detected_rate, dist_size=1e5):
        '''
        Function to generate noise distribution from input source count

        Parameters
        ----------
        detected_rate : float
            Photon count rate from object reaching the detector. [photons/s]
        dist_size : int
            Size of distribution to generate (default: 1e5)

        Returns
        -------
        with_noise : np.ndarray
            Output distribution of possible e- counts for input planet count rate
        
        '''
        N = detected_rate * self.t # pixel signal
        input_photons = np.array([N]) # convert to array for the following

        # Add shot noise
        with_shot_noise = np.random.poisson(input_photons, int(dist_size))
        expected_counts = with_shot_noise * self.qe # [e-]

        # Add dark current
        base_dark_current = self.dark_current * self.t + self.cic # [e-/pxl]
        expected_counts = expected_counts + base_dark_current 

        # Add gain for emccd
        k = expected_counts
        zero_mask = k == 0 # temporarily remove 0 to avoid didvide by zero errors
        k[zero_mask] = 1
        theta = self.gain - 1 + 1/k # gamma distribution scale parameter
        counts_add_gain = np.random.gamma(k, theta) + k - 1 # gamma distribution for gain
        counts_add_gain[zero_mask] = 0 # set values back

        # read noise
        read_noise_dist = np.random.normal(0, self.read_noise, int(dist_size))
        with_noise = counts_add_gain + read_noise_dist

        # if > fwc, set to fwc
        with_noise[with_noise > self.fwc] = self.fwc

        with_noise /= self.conversion_gain # [ADU]

        return with_noise

def get_planet_count_rate(Planet, Star, Detector, xs, ys, zs):
    '''
    Function to calculate planetary phase angle given x,y,z coordinates on-sky. 
    Calculates the planet-star flux ratio and converts planet flux density
    to planet photon count rate on the detector.

    Parameters
    ----------
    Planet : 
        Object with planet parameters (Ag, R_p)
    Star : 
        Object with stellar parameters (R_star, d_system)
    Detector : detector.Detector
        Detector object with all parameters defined.
    xs : numpy.ndarray
        x-values for planet location from deconfuser.sample_planets.
            Example: array([x_planet_1, x_planet_2, ..., x_planet_N])
    ys : numpy.ndarray
        y-values for planet location from deconfuser.sample_planets.
            Example: array([y_planet_1, y_planet_2, ..., y_planet_N])
    zs : numpy.ndarray
        z-values for planet location from deconfuser.sample_planets.
            Example: array([z_planet_1, z_planet_2, ..., z_planet_N])

    Returns
    -------
    phases : list of floats
        Phase angles at each planet location.
    phase_function : list of floats
        Value of phase function at each planet location.
    fpfs : list of floats
        Value of planet-star flux ratio at each planet location.
    planet_counts : list of floats
        Value of planet count rate at each planet location.

    '''
    # Set empty lists for appending later
    phases, phase_function, fpfs, separation, Fp, planet_counts = [], [], [], [], [], []
    
    # --------- Set constants ---------
    observer_distance_AU = Star.d_system.to(u.AU)  # units: AU
    d_system = Star.d_system.to(u.m)               # distance to system in meters

    # --------- Convert coordinates to orbital separation from star (star @ origin (0,0,0)) ---------
    for i in range(0,len(xs[0])):
        x_planet = xs[0][i] * u.AU                 # all coordinates from deconfuser are in units of AU
        y_planet = ys[0][i] * u.AU
        z_planet = zs[0][i] * u.AU
    
        separation.append(np.sqrt(x_planet**2 + y_planet**2 + z_planet**2)) # planet separation from star
    # print('separation: ', separation)
    # --------- Calculate star values ---------
    B_lambda_star = Star.blackbody_spec(wavelength=Detector.wavelength) 
    F_star = Star.stellar_flux(B_lambda_star=B_lambda_star)             # stellar flux density

    # --------- Calculate phase angle, lambert phase function, flux ratio, planet count rate ---------
    for detection in range(0, len(xs[0])): # For each planet detection
        # Orbital phase angle
        planet_vector = (-xs[0][detection], -ys[0][detection], -zs[0][detection]) # planet vector = (0-x_planet, 0-y_planet, 0-z_planet)
        observer_vector = (0 - xs[0][detection], 0 - ys[0][detection], -observer_distance_AU.value - zs[0][detection])  # observer location = (0,0,-observer_distance) [AU], observer vector = (0 - x_planet, 0 - y_planet, -observer_distance - z_planet)
        planet_mag = np.linalg.norm(planet_vector) # get magnitude of vector
        obs_mag = np.linalg.norm(observer_vector)

        # --------- Phase angle ---------
        phase_angle = np.arccos(np.dot(planet_vector, observer_vector) / (planet_mag * obs_mag)) # calculate phase angle
        if xs[0][detection] < 0:
            phase_angle = phase_angle - 2*phase_angle # convert to negative angle for plotting whole orbit
        phases.append(np.degrees(phase_angle))

        # --------- Lambert phase function ---------
        lambert_phase = (np.sin(np.absolute(phase_angle)) + \
                            (np.pi - np.absolute(phase_angle)) * np.cos(np.absolute(phase_angle))) / np.pi
        phase_function.append(lambert_phase)
        
        # --------- Planet flux density ---------
        F_planet = np.pi * Planet.Ag * lambert_phase * B_lambda_star * (Star.R_star / (separation[detection].to(u.m)))**2 * (Planet.R_p / d_system)**2 
        Fp.append(F_planet.value)

         # --------- Flux ratio ---------
        flux_ratio = Planet.Ag * ((Planet.R_p / (separation[detection].to(u.m)))**2) * lambert_phase
        fpfs.append(flux_ratio.value)
    
        # --------- Convert to planet counts ---------
        # c_p = np.pi * Detector.qe * Detector.f_pa * Detector.throughput * (Detector.wavelength / (const.h * const.c)) * F_planet * Detector.bandwidth * (Detector.D / 2)**2
        # print('c_p before unit conversion: ', c_p)
        # Convert F_planet to units of J / s / m^3 first
        if F_planet.unit != (u.J / u.s / u.m**3):
            F_planet = F_planet.to(u.J / u.s / u.m**2 / u.m)
        c_p = np.pi * Detector.qe * Detector.f_pa * Detector.throughput * (Detector.wavelength / (const.h * const.c)) * F_planet * Detector.bandwidth * (Detector.D / 2)**2
        planet_counts.append(c_p.value)

    return phases, phase_function, fpfs, planet_counts 

def get_count_rate_from_spectrum(Planet, Star, Detector, xs, ys, zs, filter_name, 
                                 use_lambert_phase=False, spectrum_dir="spectra/",
                                 get_new_albedo_spec=False):
    '''
    Function to calculate planetary phase angle and planet counts given x,y,z coordinates on-sky. 
    Calculates the planet-star flux ratio and converts planet flux density
    to planet photon count rate on the detector given the filter of observation.

    Parameters
    ----------
    Planet : 
        Object with planet parameters (Ag, R_p)
    Star : 
        Object with stellar parameters (R_star, d_system)
    Detector : detector.Detector
        Detector object with all parameters defined.
    xs : numpy.ndarray
        x-values for planet location from deconfuser.sample_planets.
            Example: array([x_planet_1, x_planet_2, ..., x_planet_N])
    ys : numpy.ndarray
        y-values for planet location from deconfuser.sample_planets.
            Example: array([y_planet_1, y_planet_2, ..., y_planet_N])
    zs : numpy.ndarray
        z-values for planet location from deconfuser.sample_planets.
            Example: array([z_planet_1, z_planet_2, ..., z_planet_N])
    filter_name : string
        Name of filter being used for observation
    use_lambert_phase : boolean
        Boolean indicating whether or not to apply lambert phase function to planet flux
        If True, applies Lambertian phase function in flux calculation.
        If False, assumes you are reading in albedo spectrum as function of phase angle.
    spectrum_dir : string
        Location of spectra -- should contain directories labeled as:
        "rocky", "ice_giant", "gas_giant"
    get_new_albedo_spec : boolean
        Whether or not to get new albedo spectra for the planets. 
        Spectra will already be saved if the planets have been run through one filter.
        Saving additional spectra will continue to append to the list. 
        Only set to True if running through first filter, False otherwise

    Returns
    -------
    phases : list of floats
        Phase angles at each planet location.
    phase_function : list of floats
        Value of phase function at each planet location.
    fpfs : list of floats
        Value of planet-star flux ratio at each planet location.
    planet_counts : list of floats
        Value of planet count rate at each planet location.

    '''
    # Set empty lists for appending later
    separation, phases, phase_function, Fp, Fp_filts, fpfs, Fp_band_all, planet_counts = [], [], [], [], [], [], [], []
    
    # --------- Set constants ---------
    observer_distance_AU = Star.d_system.to(u.AU)  # units: AU
    d_system = Star.d_system.to(u.m)               # distance to system in meters

    # # --------- calculate stellar contribution -------
    # try:
    #     Star.blackbody_spec(Planet.wavelength)         # B_star
    # except:
    #     print("AttributeError: 'Planet' object has no attribute 'wavelength' for Star.blackbody_spec. Using Detector.wavelength")
    #     print('Detector.wavelength: ', Detector.wavelength)
    #     Star.blackbody_spec(wavelength=Detector.wavelength)

    # --------- Convert coordinates to orbital separation from star (star @ origin (0,0,0)) ---------
    for i in range(0,len(xs[0])):
        x_planet = xs[0][i] * u.AU                 # all coordinates from deconfuser are in units of AU
        y_planet = ys[0][i] * u.AU
        z_planet = zs[0][i] * u.AU
    
        separation.append(np.sqrt(x_planet**2 + y_planet**2 + z_planet**2)) # planet separation from star

    # --------- Calculate phase angle, lambert phase function, flux ratio, planet count rate ---------
    for detection in range(0, len(xs[0])): # For each planet detection
        # Orbital phase angle
        planet_vector = (-xs[0][detection], -ys[0][detection], -zs[0][detection]) # planet vector = (0-x_planet, 0-y_planet, 0-z_planet)
        observer_vector = (0 - xs[0][detection], 0 - ys[0][detection], -observer_distance_AU.value - zs[0][detection])  # observer location = (0,0,-observer_distance) [AU], observer vector = (0 - x_planet, 0 - y_planet, -observer_distance - z_planet)
        planet_mag = np.linalg.norm(planet_vector) # get magnitude of vector
        obs_mag = np.linalg.norm(observer_vector)

        # --------- Phase angle ---------
        phase_angle = np.arccos(np.dot(planet_vector, observer_vector) / (planet_mag * obs_mag)) # calculate phase angle
        if xs[0][detection] < 0:
            phase_angle = phase_angle - 2*phase_angle # convert to negative angle for plotting whole orbit
        phases.append(np.degrees(phase_angle))
        if get_new_albedo_spec: # only save phases in first round of observations
            Planet.phase_angles.append(np.degrees(phase_angle)) # also save phases of each detection to Planet object for retrieving spectra

        # --------- Lambert phase function ---------
        lambert_phase = (np.sin(np.absolute(phase_angle)) + \
                            (np.pi - np.absolute(phase_angle)) * np.cos(np.absolute(phase_angle))) / np.pi
        phase_function.append(lambert_phase)

    # ------ Read in albedo spectra for detections/phases
    # Only need to get albedo spectra on first filter pass 
    if get_new_albedo_spec:
        get_alb_spectra(Planet, spectrum_dir=spectrum_dir)

    # ---- Calculate stellar contribution -----
    try: 
        Star.blackbody_spec(Planet.wavelength[0])
    except:
        print("AttributeError: 'Planet' object has no attribute 'wavelength' for Star.blackbody_spec. Using Detector.wavelength scalar")
        Star.blackbody_spec(wavelength=Detector.wavelength)

    for detection in range(0, len(xs[0])):
        # --------- Planet flux density & flux ratio ---------
        if use_lambert_phase:
            F_planet = np.pi * Planet.Ag * phase_function[detection] * Star.B_star * (Star.R_star / (separation[detection].to(u.m)))**2 * (Planet.R_p / d_system)**2 
            flux_ratio = Planet.Ag * ((Planet.R_p / (separation[detection].to(u.m)))**2) * lambert_phase
            Planet.phase_method = "Lambertian"
        else: # assumed Ag is an albedo spectrum that already is a function of phase angle
            F_planet = np.pi * Planet.Ag[detection] * Star.B_star * (Star.R_star / (separation[detection].to(u.m)))**2 * (Planet.R_p / d_system)**2 
            flux_ratio = Planet.Ag[detection] * ((Planet.R_p / (separation[detection].to(u.m)))**2) * lambert_phase
            Planet.phase_method = "Albedo spectrum"

        Fp.append(F_planet.value)
        fpfs.append(flux_ratio.value)

        # ----- Get filter info ---------
        if filter_name != None:
            filter_data, filter_lambda_min, filter_lambda_max = Detector.get_filter_info(filter_name)
            filt_transmission, filt_lambda = filter_data['filter_transmission'], filter_data['filter_lambda']

            # ----- Get planet flux in only filter region of interest ---------
            # Planet wavelength must be in units of um
            Fp_filt, planet_lambda_filt = spectrum_in_filter(Planet.wavelength[detection].value, F_planet, filter_lambda_min, filter_lambda_max)
            Fp_filts.append(Fp_filt)

            # Regrid filter points to planet wavelength grid
            filt_transmission = np.interp(planet_lambda_filt, filt_lambda, filt_transmission)

            # ----- Calculate band-averaged flux -----
            Fp_band = np.trapz(Fp_filt * filt_transmission, planet_lambda_filt) / np.trapz(filt_transmission, planet_lambda_filt) # units: W / m^2 / um
            Fp_band_all.append(Fp_band)
        else:
            print("No filter specified, using single-band calculation.")
            Fp_band = F_planet
        # --------- Convert to planet counts ---------
        # First convert Fp_band to J / s / m^3 for c_p
        if Fp_band.unit != (u.J / u.s / u.m**3):
            Fp_band = Fp_band.to(u.J / u.s / u.m**2 / u.m)

        # All wavelengths here need to be in meters; convert to planet counts
        c_p = np.pi * Detector.qe * Detector.f_pa * Detector.throughput * (Detector.wavelength / (const.h * const.c)) * Fp_band * Detector.bandwidth * (Detector.D / 2)**2
        planet_counts.append(c_p.value) # units: [1 / s]

        # Save info to planet object
        if filter_name != None:
            # if get_new_albedo_spec:
            #     planet_flux_dict = {'filter': filter_name, 'Fp_specs_in_filter': Fp_filts, 'Fp_band_avgs': Fp_band_all}
            if not any(d["filter"] == filter_name for d in Planet.band_fluxes): 
                planet_flux_dict = {'filter': filter_name, 'Fp_specs_in_filter': Fp_filts, 'Fp_band_avgs': Fp_band_all, 'planet_wavelength_in_filter': planet_lambda_filt}
                Planet.band_fluxes.append(planet_flux_dict) # save to planet object

    return phases, phase_function, fpfs, Fp_band_all, planet_counts

def separate_xyzs(xyzs_array):
    '''
    Function to separate x, y, and z coordinates of planet detections
    
    Parameters
    ----------
    xyzs_array : list of lists 
        List of x-, y-, z-coordinate groupings
        
    Returns 
    -------
    xs, ys, zs : np.arrays of x, y, and z coordinate values
    
    '''
    xs = np.array([[xyzs_array[i][0] for i in range(0,len(xyzs_array))]])
    ys = np.array([[xyzs_array[i][1] for i in range(0,len(xyzs_array))]])
    zs = np.array([[xyzs_array[i][2] for i in range(0,len(xyzs_array))]])
    
    return xs, ys, zs

def calc_SNR(C_p, C_b):
    '''
    Calculate SNR given planet counts per integration time and 
    background counts per integration time

    Parameters
    ----------
    C_p : numpy.float64
        Planet counts * integration time
    C_b : numpy.float64
        Background counts * integration time

    Returns
    -------
    SNR
        signal-to-noise ratio for a given detection
    '''

    # Compute total noise and SNR
    C_noise = np.sqrt( C_p + C_b )       # counts due to noise
    SNR = C_p / C_noise                  # planet counts / noise

    return SNR

def sigma_photo(FWHM, SNR, SNR_low_lim=2, sigma_lim=0.03):
    '''
    Positional uncertainty due to detected signal.
    Set to maximum uncertainty constant if SNR is below some lower limit. 

    Parameters
    ----------
    FWHM : float
        FWHM of system of interest in units of arcseconds
    SNR : float
        signal-to-noise ratio of detection
    SNR_low_lim : float
        Lower limit on SNR for setting limit on uncertainty.
        Default = 2
    sigma_lim : float
        Upper limit on uncertainty in units of arcseconds.

    Returns
    -------
    np.ndarray
        Astrometric uncertainty with same units as FWHM
    '''
    sigma_all = []
    for arr in SNR: # for each epoch, get array of detections
        sigma_subarr = []
        for snr in arr: # for each detection
            if snr >= SNR_low_lim:
                sigma = FWHM / snr
            else:
                sigma = sigma_lim
            sigma_subarr.append(sigma)
        sigma_all.append(sigma_subarr)
    sigma_all = np.array(sigma_all)

    return sigma_all

def astro_photo_uncertainty(SNRs, detector, star, SNR_low_lim, sigma_lim):
    '''
    Estimate the astrometric uncertainty due to the photometry
    and add the position error to the observation coordinates.

    Parameters
    ----------
    SNRs : list of np.arrays
        SNRs of detections
    detector : photometry.Detector
        detector object which contains FWHM and stability constant
    star : photometry.Star
        star object which contains distance to system
    SNR_low_lim : float
        Lower limit on SNR for setting uncertainty. 
        Keyword for sigma_photo()
    sigma_lim : float
        Upper limit on uncertainty in units of AU.
        Corresponds to minimum SNR value. If SNR below SNR_low_lim, 
        uncertainty set to sigma_lim.

    Returns
    -------
    np.ndarray
        Astrometric uncertainty in units of AU
    '''
    sigma_as = sigma_photo(detector.FWHM, SNRs, SNR_low_lim, sigma_lim)
    sigma_AU = arcsec_to_AU(sigma_as, star.d_system)

    return sigma_AU

def arcsec_to_AU(angular_sep_arcsec, dist_pc):
    '''
    Convert separation in arcsec to AU

    Parameters
    ----------
    angular_sep_arcsec : np.ndarray
        angular separation in arcseconds
    dist_pc : float
        observed system distance [parsecs]

    Returns
    -------
    np.ndarray
        angular separation in AU

    '''
    separation_AU = angular_sep_arcsec * dist_pc.value

    return separation_AU

def get_detections_counts(n_planets, n_detections, xyzs, Planet, Star, System, Detector): # TODO: rename function to something like "simulate_noisy_detection" ?
    '''
    Generates noisy planet detections.
    Accepts detection coordinates, calculates phase/brightness, adds detector noise.

    Parameters
    ----------
    n_planets : int
        Number of planets in the system.
    n_detections : int
        Number of detections of the system.
    xyzs : numpy.ndarray
        Array of X, Y, Z coordinates for each detection. Format: 
            [[X1_1, Y1_1, Z1_1], [X2_1, Y2_1, Z2_1], ..., [XN_M, YN_M, ZN_M]], 
            where N is the number/time of detection and M is the number of 
            planet in the system.
    Planet : photometry.Planet
        Planet object containg information about planet (Ag, R_p)
    Star : photometry.Star
        Star object containing information about host star (R_star, distance)
    Detector : photometry.Detector
        Detector object containing detecting instrument parameters.

    Returns
    -------
    noisy_counts_sys : list
        "Simulated detections". Planet detections with detector noise added [e-].
    photon_rates_sys : list
        Calculated photon rates per planet detection [photons/s].
    SNR_sys : list of floats
        SNR values for each detectoin
    phases_sys : list of floats
        Orbital phase corresponding to each detection
    '''
    
    noisy_counts_sys = []
    photon_rates_sys = []
    SNR_sys = []
    phases_sys = []
    
    for planet in range(n_planets):
        # --------- Handle detection coordinates ----------
        xyzs_planet = xyzs[planet]
        xs, ys, zs = separate_xyzs(xyzs_planet) 
        
        # ----------- Calculate phase and intensity information -----------
        phases, phase_func, fpfs, photon_counts = get_planet_count_rate(Planet, Star, Detector, xs=xs, 
                                                                       ys=ys, zs=zs)
        # ----------- append detections' photon rates to one list ---------
        photon_rates_sys.append(photon_counts)
        phases_sys.append(phases)

        # ----------- Calculate noisy detections ---------
        noisy_counts, SNRs = [], []
        # Calculate counts due to noise sources 
        bkgd_count, C_zod_exozod_lk = Detector.add_noise(System.n_zodi + System.n_exozodi + System.n_leakage) # Count rate due to zodiacal light, exozodiacal contribution, and leakage (4 + 2+ 20) (Robinson+2016)
        C_dc = Detector.dark_current * Detector.t                                # Counts due to dark current
        C_b = C_zod_exozod_lk + Detector.read_noise**2 + C_dc # background counts

        for count in photon_counts:
            noisy_count, C_p = Detector.add_noise(count) # noisy count per detection, planet count rate * integration time
            noisy_counts.append(noisy_count)
            # calculate SNR of detection
            SNR = calc_SNR(C_p, C_b)
            SNRs.append(SNR)

        noisy_counts = np.reshape(np.asarray(noisy_counts), (1,n_detections))
        noisy_counts_sys.append(noisy_counts[0]) 
        SNR = np.reshape(np.asarray(SNRs), (1,n_detections))
        SNR_sys.append(SNR[0])
        
    return noisy_counts_sys, photon_rates_sys, SNR_sys, phases_sys

def get_detections_counts_color(n_detections, xyzs, Star, System, Detector, 
                                filter_name=None, spectrum_dir="spectra/",
                                first_filter=False): 
    '''
    Generates noisy planet detections.
    Accepts detection coordinates, calculates phase/brightness, adds detector noise.

    Parameters
    ----------
    n_detections : int
        Number of detections of the system.
    xyzs : numpy.ndarray
        Array of X, Y, Z coordinates for each detection. Format: 
            [[X1_1, Y1_1, Z1_1], [X2_1, Y2_1, Z2_1], ..., [XN_M, YN_M, ZN_M]], 
            where N is the number/time of detection and M is the number of 
            planet in the system.
    Star : photometry.Star
        Star object containing information about host star (R_star, distance)
    System : photometry.System
        System object containing information about system and planets
    Detector : photometry.Detector
        Detector object containing detecting instrument parameters.
    filter_name : string
        Name of filter for observation
    spectrum_dir : string
        Path to spectra -- should contain "rocky", "ice_giant", and "gas_giant" directories
    first_filter : boolean
        Wheter or not this is the first filter for the observation set.
        True if yes; False otherwise.
        This controls when albedo spectra are pulled for the planets (see get_count_rate_from_spectrum)

    Returns
    -------
    noisy_counts_sys : list
        "Simulated detections". Planet detections with detector noise added [e-].
    photon_rates_sys : list
        Calculated photon rates per planet detection [photons/s].

    '''
    
    noisy_counts_sys, photon_rates_sys, SNR_sys, phases_sys = [], [], [], []

    for planet in range(len(System.planets)):
        # --------- Handle detection coordinates ----------
        xyzs_planet = xyzs[planet]
        xs, ys, zs = separate_xyzs(xyzs_planet) 
        
        # ----------- Calculate phase and intensity information -----------
        phases, phase_func, fpfs, Fp_band_all, photon_counts = get_count_rate_from_spectrum(System.planets[planet], 
                                                                                            Star, Detector, xs=xs, 
                                                                                            ys=ys, zs=zs, 
                                                                                            filter_name=filter_name,
                                                                                            use_lambert_phase=False,
                                                                                            spectrum_dir=spectrum_dir,
                                                                                            get_new_albedo_spec=first_filter)
        print('photon_counts: ', photon_counts) # TODO: remove
        print('fpfs: ', fpfs)

        # ----------- append detections' photon rates to one list ---------
        photon_rates_sys.append(photon_counts)
        phases_sys.append(phases)
        
        # ----------- Calculate noisy detections ---------
        noisy_counts, SNRs = [], []
        # Calculate counts due to noise sources 
        bkgd_count, C_zod_exozod_lk = Detector.add_noise(System.n_zodi + System.n_exozodi + System.n_leakage) # Count rate due to zodiacal light, exozodiacal contribution, and leakage (4 + 2+ 20) (Robinson+2016)
        C_dc = Detector.dark_current * Detector.t             # Counts due to dark current
        C_b = C_zod_exozod_lk + Detector.read_noise**2 + C_dc 

        for count in photon_counts:
            noisy_count, C_p = Detector.add_noise(count) # noisy count per detection, planet count rate * integration time
            noisy_counts.append(noisy_count)
            # calculate SNR of detection
            SNR = calc_SNR(C_p, C_b)
            SNRs.append(SNR)

        noisy_counts = np.reshape(np.asarray(noisy_counts), (1,n_detections))
        noisy_counts_sys.append(noisy_counts[0]) 
        SNR = np.reshape(np.asarray(SNRs), (1,n_detections))
        SNR_sys.append(SNR[0])
        
    return noisy_counts_sys, photon_rates_sys, SNR_sys, phases_sys
# Run deconfuser with phase information for specific systems
# S. Hasler
import numpy as np
import os
import sys
import pandas as pd
import csv
from datetime import datetime 
import matplotlib.pyplot as plt # TODO: remove
from matplotlib.patches import Circle # TODO: remove

import deconfuser.sample_planets as sample_planets
import deconfuser.orbit_fitting as orbit_fitting
import deconfuser.orbit_grouping as orbit_grouping
import deconfuser.partition_ranking as partition_ranking
import photometry.photometry as phot
import photometry.likelihood as L
import photometry.ranking as ranking

# ------------------------------------------------------------
# Specify parameters
mu = 4*np.pi**2 #Sun's gravitational parameter in AU^3/year^2
n_planets = 3
n_epochs = 3
cadence = 0.5
verbose = True
sigma_photo = True
plot = False
tolerances = [0.20]
n_systems = 11

# Deconfuser parameters
max_a = 6.0 # max a [AU]
min_a = 0.5 # AU
max_e = 0.1 # eccentricity max
sep_a = 0.3
min_i = 0.0 # rad
max_i = 1.5707963267948966 # rad
spread_i_O = 0.0 # spread of inclination and LAN in radians

ranking_path = "/Users/shasler/Code/deconfuser/output_files/ranking_files/"
# signal-dependent uncertianty parameters
SNR_lower_lim = 2.0
# TODO: Change sigma_lim depending on planet type/distance
sigma_lim = 0.015 # max uncertainty in astrometry due to SNR

start = datetime.now()
now = start.strftime("%Y-%m-%d_%H%M%S") # for text file

# File with systems to run
inc_group = "lowi" # options: lowi, medi, highi
systems_orb_params_file = f"/Users/shasler/Documents/Projects/Deconfusion/publication/ten_systems/orbparams_10confused_systems_{inc_group}.txt" # change inc_group to change file of interest
if inc_group == "lowi":
    skip_footer = 0
else:
    skip_footer = 1 # skip last line of file for highi and medi groups as there is a note there

# Output file path
path = "/Users/shasler/Code/deconfuser/output_files/"
output_file = f"{inc_group}_wPhotoErr_output_tol{tolerances[0]}_{now}.txt"
run_log =  f"run_log_{inc_group}_wPhotoErr_10systems_tol{tolerances[0]}_{now}.log"

f = open(path + output_file, "a")
logfile = open(path +run_log, "a") 
sys.stdout = logfile # redirect output to log file
sys.stderr = logfile # redirect error output to log file also
# ------------------------------------------------------------
# Read in systems
df_system_params = pd.read_csv(systems_orb_params_file, header=0, delimiter=', ', skipfooter=skip_footer)

# Write headers to text file
writer = csv.writer(f)
headers = ['system', 'n_planets', 'planet', 'n_orbit_options', 'a', 'e', 'i', 'o', 'O', 'M0', 'ts', 'xyzs', 'correct_partition', 'top_partitions',\
           'partition', 'group', 'rms_fit_err', 'L_group_options', 'L_partition_options', 'L_detections', 'noisy_detections', 'detection_photon_rates'] 

run_parameters = f'Run parameters: {n_systems} systems, {n_planets} planets, \
    {n_epochs} epochs, {cadence} cadence (yr), {tolerances} tolerances'
writer.writerow([run_parameters]) # save run parameters in file
writer.writerow(headers) # add headers to file

# Set up planet, star, and detector parameters for photometry
system = phot.System(n_exozodi=4/3600, n_leakage=20/3600, n_zodi=2/3600) # background count contributions in units of s^-1
star = phot.Star(T=5778, R_star=695700e3, d_system=10, mu=mu) # system distance in parsecs -- values for the Sun
planet = phot.Planet(R_p=6.371e6, Ag=0.3)                        # Rp = 6.371e6 km, Ag=0.3 -- values for Earth
# detector = phot.Detector(qe=0.837, cic=0.016, dark_current=1.3e-4, read_noise=120, gain=1000, 
#                     fwc=80000, conversion_gain=1.0, t=2e6, D=2.36, throughput=0.38, f_pa=0.039,
#                     wavelength=573.8e-9, bandwidth=56.5e-9) 
detector = phot.Detector(qe=0.9, cic=0.016, dark_current=5e-4, read_noise=120, gain=1000, 
                    fwc=10000000, conversion_gain=1.0, t=20*3600, D=6, throughput=0.05, f_pa=0.87,
                    wavelength=550e-9, bandwidth=50e-9) 

#observation epochs (years)
ts = cadence*np.arange(n_epochs)

#the correct partition of detection by planets
correct_partition = [tuple(range(i*len(ts),(i+1)*len(ts))) for i in range(n_planets)]
print(f'correct_partition: {correct_partition}') 

#to speed up computation, begin with coarsest tolerance and progress to finest:
#1. full orbit grouping will be performed with the coarsest tolerance (i.e., recursively consider all groupings of observation)
#2. only "full" groups that fit observation within a coarser tolerance will be fitted with a finer tolerance
#Note: "missed" detections are not simulataed here so confusion will only "arise" with full groups (n_epochs observations per planet)
tolerances = sorted(tolerances, reverse=True)
tol = tolerances[0] # TODO: ADJUST THIS SECTION + ORBIT_FITTERS LATER TO HANDLE MULTIPLE TOLERANCES

orbit_grouper = orbit_grouping.OrbitGrouper(mu, ts, min_a-tolerances[0], max_a+tolerances[0], max_e, tolerances[0], lazy_init=False)
orbit_fitters = [orbit_fitting.OrbitFitter(mu, ts, min_a-tol, max_a+tol, max_e, tol) for tol in tolerances[1:]]
orbit_fitter = orbit_fitting.OrbitFitter(mu, ts, min_a-tol, max_a+tol, max_e, tol) # TODO: remove later -- SH added

for _ in range(n_systems):
    if plot:
        fig, ax = plt.subplots() # TODO: remove
    #%% -------------------- Generate simulated systems --------------------
    print(f'\nSystem #{_+1} \n----------') # outputs which number system for readability

    # Get orbit parameters
    system_data = df_system_params[df_system_params['system'] == _ + 1] # system numbers are 1-indexed
    a_vals = np.array(system_data['a'].values)
    e_vals = np.array(system_data['e'].values)
    i_vals = np.array(system_data['i'].values)
    o_vals = np.array(system_data['o'].values)
    O_vals = np.array(system_data['O'].values)
    M0_vals = np.array(system_data['M0'].values)

    #get coordinates of planets when observed
    xs,ys,zs = sample_planets.get_observations(a_vals, e_vals, i_vals, o_vals, O_vals, M0_vals, ts, mu) 
    # TODO: remove ------
    t_range = 200
    ts_more = 0.05*np.arange(t_range)
    xs_more, ys_more, zs_more = sample_planets.get_observations(a_vals, e_vals, i_vals, o_vals, O_vals, M0_vals, ts_more, mu)
    obs_more = np.stack([xs_more,ys_more,zs_more], axis=2).reshape((-1,3))
    # ------ ^^^^ ------
    observations = np.stack([xs,ys,zs], axis=2).reshape((-1,3))

    #add radially bounded astrometry error
    if sigma_photo:
        noise_r = 0
        noise_a = 0
    else:
        noise_r = tolerances[-1]*np.random.random(len(observations)) # returns array of len(obs) * final tolerance value 
        noise_a = 2*np.pi*np.random.random(len(observations)) # radial error?  
    observations[:,0] += noise_r*np.cos(noise_a) # x-direction error  
    observations[:,1] += noise_r*np.sin(noise_a) # y-direction error 
        # observations format: array([[group1_x, group1_y, group1_z], [group2_x, ..., ...], [groupN_x, groupN_y, groupN_z]])
        # observations are the x,y,z coordinates for each of the orbit groupings, which potential orbital parameters are drawn from
    if plot:
        ax.scatter(observations[:,0][:3], observations[:,1][:3], marker='o', s=50, color='k', label='original position') # TODO: remove
        ax.scatter(observations[:,0][3:6], observations[:,1][3:6], marker='s', s=50, color='k') # TODO: remove
        ax.scatter(observations[:,0][6:], observations[:,1][6:], marker='^', s=50, color='k') # TODO: remove
        ax.plot(obs_more[:,0][:t_range], obs_more[:,1][:t_range], color='k', alpha=0.7)
        ax.plot(obs_more[:,0][t_range:t_range*2], obs_more[:,1][t_range:t_range*2], color='k', alpha=0.7)
        ax.plot(obs_more[:,0][t_range*2:], obs_more[:,1][t_range*2:], color='k', alpha=0.7)
    xs_original = observations[:,0].flatten()
    ys_original = observations[:,1].flatten()

    # first adjust coordinates for use in get_detections_counts function
    all_coords = []
    for ip in range(n_planets):
        all_coords.append(list(map(list, observations[ip*len(ts):(ip+1)*len(ts)])))

    all_coords = np.asarray(all_coords)
    # get noisy and not noisy photometric detections for simulated system -- phase information buried in this function
    noisy_detections, detections_photon_rates, SNRs, phases = phot.get_detections_counts(n_planets, n_epochs, xyzs=all_coords, 
                                                                               Planet=planet, Star=star, System=system, Detector=detector)
    print('noisy_detections: ', noisy_detections)
    print('SNRs: ', SNRs)
    print('phases: ', phases)

    if sigma_photo: # if true, add astrometric uncertainty to observations due to simulated photometry
        # Calculate error in x,y directions due to planet signal
        sigma_AU = phot.astro_photo_uncertainty(SNRs, detector, star, SNR_low_lim=SNR_lower_lim, sigma_lim=sigma_lim) # TODO: remove hard-coded values
        # Add uncertainty to coordinates as gaussian with standard deviation of sigma
        observations[:,0] = np.random.normal(observations[:,0], sigma_AU.flatten())
        observations[:,1] = np.random.normal(observations[:,1], sigma_AU.flatten())

        # TODO: remove plotting
        if plot:
            ax.scatter(observations[:,0][:3], observations[:,1][:3], marker='o', s=50, edgecolor='r', facecolor='r', 
                    linewidth=1, alpha=0.5, label='with SNR-dependent err', zorder=3)
            ax.scatter(observations[:,0][3:6], observations[:,1][3:6], marker='s', s=50, edgecolor='r', facecolor='r', 
                    linewidth=1,  alpha=0.5, zorder=3)
            ax.scatter(observations[:,0][6:], observations[:,1][6:], marker='^', s=50, edgecolor='r', facecolor='r', 
                    linewidth=1,  alpha=0.5, zorder=3) 

            for xi, yi, zi in zip(xs_original, ys_original, sigma_AU.flatten()):
                circle = Circle((xi, yi), zi*2, edgecolor='red', facecolor='none', lw=2, linestyle='dashed', alpha=0.9)
                # ^ zi = 1-sigma uncertainty radius size
                ax.add_patch(circle)

    if plot:
        ax.scatter(0, 0, marker='*', color='gold', s=100)
        ax.set_xlabel('x (AU)')
        ax.set_ylabel('y (AU)')
        ax.set_aspect('equal')
        plt.legend()
        plt.title(f'System {_+1}: Detections with vs. without\njoint astro/photo error')
        plt.show()


    if verbose:
        print("\nts =", list(ts)) # observation epochs
        for ip in range(n_planets): # for every planet
            print("\nplanet ", ip+1, ": ")
            print("a,e,i,o,O,M0 = ", (a_vals[ip],e_vals[ip],i_vals[ip],o_vals[ip],O_vals[ip],M0_vals[ip])) # true orbital parameters for each planet
            print("xyzs =", list(map(list, observations[ip*len(ts):(ip+1)*len(ts)]))) # true coordinates of detection for each planet
            print("photon_rates = ", detections_photon_rates[ip]) # calculated photon rates for each planet detection (format: [detection1, detection2, ..., detectionN])
            print("noisy_detections = ", noisy_detections[ip]) # noisy planet detections (format: [array([planet1_detection1, ..., planet1_detectionN])], ..., array([planetM_detection1, ..., planetM_detectionN])])

    # output simulated planet info to text file
    for ip in range(n_planets):
        planet_params = [_+1, n_planets, ip+1, np.NaN, a_vals[ip], e_vals[ip], i_vals[ip], o_vals[ip], O_vals[ip], M0_vals[ip], ts, list(map(list, \
                        observations[ip*len(ts):(ip+1)*len(ts)])), correct_partition, None, None, None, None, None, None, None, noisy_detections[ip], \
                        detections_photon_rates[ip]] # system #, # planets simulated, planet #, a, e, i, o, O, M0, confused?, ts, xyzs, correct_partition, top_partitions, group, 'L_detections', 'L_group_options', 'L_partition_options', 'noisy_detections', 'detection_photon_rates' 
        writer.writerow(planet_params)

    # All detections times for all observations
    all_ts = np.tile(ts, n_planets)

#%% -------------------- Do the orbit fitting --------------------
    #get all possible (full or partial) groupings of detection by orbits that fit them with the coarsest tolerance
    groupings = orbit_grouper.group_orbits(observations, all_ts)

    #select only groupings that include all epochs (these will be most highly ranked, so no need to check the rest)
    groupings = [g for g in groupings if len(g) == n_epochs] # lists found groupings of detections
    
    #check for spurious orbits and repeat for finer tolerances
    for j in range(len(tolerances)):
        found_correct = sum(cg in groupings for cg in correct_partition)

        print(f'found_correct: {found_correct}') # TODO: remove -- added for testing

        print('-------------------------------------------------------------')
        print("Tolerance %f: found %d correct and %d spurious orbits out of %d"%(tolerances[j], found_correct, len(groupings) - found_correct, n_planets))
        if verbose:
            print("Tolerance %f:"%(tolerances[j]), groupings)

        #find all partitions of observations to exactly n_planets groups
        #note that since all partial grouping were filtered out, all partitions will have exactly n_planets groups
        try:
            top_partitions = list(partition_ranking.get_ranked_partitions(groupings))
        except:
            if len(groupings) == 0:
                top_partitions = []
                print("No groups which contain all epochs of observation.")

        if found_correct < n_planets:
            for ip in range(n_planets):
                if not correct_partition[ip] in groupings:
                    print("Failed to fit a correct orbit for planet %d!"%(ip))
        elif len(top_partitions) == 1:
            print("Tolerance %f: no confusion"%(tolerances[j]))
        else:
            assert(len(top_partitions) > 1)
            L_system_options = [] # for system likelihoods
            # Get orbital parameters of spurious orbits
            for partition in top_partitions: # add to print orbit parameters
                print('partition: ', partition)
                L_partition_options = [] # pre-allocate and clear group options list
                l = 0 # for getting correct noisy counts per planet orbit option
                for group in partition: # which data points lie on the orbit
                    p=0 # TODO: remove -- SH added for testing

                    print('\ngroup: ', group)
                    L_group_options = [] # pre-allocate and clear orbit options list
                    group_orbit_parameters = []
                    errs = []
                    k = 0 # keep track of how many orbit options per group

                    # Print groups + orbital parameters
                    for err, parameters in orbit_fitter.fit(observations[group]): 
                        p=1 # TODO: remove -- SH added for testing
                        print('\nParameters: ', parameters)
                        print('err: ', err) 

                        if parameters[2] == np.nan:
                            print('Error! inclination is NaN!')
                        else:
                            # Phase information section
                            # Phase info buried in likelihood function -- add as a return parameter in likelihood.py if you want to back it out
                            # Calculate likelihood of orbit option
                            L_orbit, L_detections = L.get_L_orbit(n_detections=n_epochs,
                                                                  a=parameters[0], e=parameters[1],
                                                                  i=parameters[2],
                                                                  o=parameters[3],
                                                                  O=parameters[4],
                                                                  M0=parameters[5],
                                                                  ts=ts,
                                                                  noisy_counts=noisy_detections[l],
                                                                  Star=star, Planet=planet, 
                                                                  Detector=detector
                                                                  )
                            print(f'L_orbit: {L_orbit}') # Likelihood of entire orbit (alldetections) 
                            L_group_options.append(L_orbit) # save L of each orbit option per group
                            print(f'L_group_options: {L_group_options}') # Likelihood of each orbit option in a group 
                            group_orbit_parameters.append(parameters)
                            errs.append(err) 
                            k += 1 # track number of orbit options

                    L_partition_options.append(L_group_options) # Likelihood of all orbit options in a partition -- will be empty if i = nan

                    # Write confused orbit options to output file
                    a_s = [orbit[0] for orbit in group_orbit_parameters] # separate orbital parameters for all orbit options in a group
                    e_s = [orbit[1] for orbit in group_orbit_parameters]
                    i_s = [orbit[2] for orbit in group_orbit_parameters]
                    o_s = [orbit[3] for orbit in group_orbit_parameters]
                    O_s = [orbit[4] for orbit in group_orbit_parameters]
                    M0_s = [orbit[5] for orbit in group_orbit_parameters]
                    option_parameters = [_+1, n_planets, l+1, k, a_s, e_s, i_s, o_s, O_s, \
                                         M0_s, ts, None, correct_partition, \
                                         top_partitions, partition, group, errs, \
                                         L_group_options, L_partition_options, \
                                         L_detections, None, None] # parameters for writing to text file
                    
                    writer.writerow(option_parameters)
                    l += 1 # advance to next detected planet in the system for comparison
                    
                    if p==0: # TODO: remove -- SH added for testing
                        print('FLAG -- P=0, NO PARAMETERS TO OUTPUT')
                    
            # ------------------------------------------------------------------
            print("Tolerance %f: found %d spurious \"good\" partitions of detections by planets (confusion)"%(tolerances[j], len(top_partitions) - 1))
            if verbose:
                print("Tolerance %f:"%(tolerances[j]), top_partitions)

        # Move to a finer tolerance
        #move to a finer tolerance
        if j < len(tolerances) - 1:
            #only keep groupings that cna be fitted with an orbit with the finer tolerance
            groupings = [g for g in groupings if any(err < tolerances[j+1] for err in orbit_fitters[j].fit(observations[list(g)], only_error=True))]

# Re-rank systems with photometry
try: # create ranking files directory if it doesn't exist
    os.makedirs("output_files/ranking_files", exist_ok=True) 
except OSError as error: 
    print("ranking_files directory cannot be created.")

# Create photometry ranking object -- houses file dataframe
try:
    confused_systems = ranking.PhotometryRanking(filepath=f"output_files/{output_file}", n_planets=n_planets)
    print('\nPerforming photometry ranking...')
    df_confused = confused_systems.get_top_group_options()  # iterate over options with multiple groups
    df_ranked = confused_systems.top_ranked_partition()     # Get top ranked partition in each system
    df_recombined = confused_systems.combine_and_cleanup(save_file=True, save_path=ranking_path + f"systems_ranked_{now}.txt")  # Combine original and ranked dataframes
    # If you want to calculate percent difference between simulated and fit orbits:
    df_final = confused_systems.orbit_percent_diff()      
    df_final_wperc = confused_systems.final_recombined(save_file=True, save_path=ranking_path + f"systems_ranked_wPercDiff_{now}.txt")    
    print('\nPhotometry ranking complete.')
except:
    print("No ranking file") # TODO: update to handle more from ^


# record run time
end  = datetime.now()
runtime = end - start
runtime_string = [f"Run time: {runtime} s"]
writer.writerow(runtime_string) # write run time to end of file

f.close() # close text file
logfile.close() # close log file
sys.stdout = sys.__stdout__ # reset standard output to terminal
sys.stderr = sys.__stderr__ # reset error output to terminal
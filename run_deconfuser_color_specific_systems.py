# Run deconfuser with phase/color information for specific systems
# S. Hasler
import numpy as np
import os
import sys
import pandas as pd
import csv
from datetime import datetime 
import matplotlib.pyplot as plt # TODO: remove
from matplotlib.patches import Circle # TODO: 
import astropy.units as u

import deconfuser.sample_planets as sample_planets
import deconfuser.orbit_fitting as orbit_fitting
import deconfuser.orbit_grouping as orbit_grouping
import deconfuser.partition_ranking as partition_ranking
import photometry.photometry as phot
import photometry.likelihood as L
import photometry.ranking as ranking
import photometry.utils as utils

# ------------------------------------------------------------
# Specify parameters
mu = 4*np.pi**2 #Sun's gravitational parameter in AU^3/year^2
n_planets = 3
n_epochs = 3
cadence = 0.5
verbose = True
sigma_photo = False
plot = False
tolerances = [0.05]
n_systems = 11
spectrum_dir = "/Users/shasler/Code/deconfuser/photometry/spectra/"

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
# These files are the same as the first deconfusion + photometry paper
inc_group = "highi" # options: lowi, medi, highi # TODO: change this ! 
systems_orb_params_file = f"/Users/shasler/Documents/Projects/Deconfusion/publication/ten_systems/orbparams_10confused_systems_{inc_group}.txt" # change inc_group to change file of interest
# systems_orb_params_file = f"/Users/shasler/Documents/Projects/deconfusion_color/specific_systems/confused_system.txt" # change inc_group to change file of interest
if inc_group == "lowi":
    skip_footer = 0
else:
    skip_footer = 1 # skip last line of file for highi and medi groups as there is a note there

# ------------ Set up star and detector parameters for photometry ------------
# Set up base system parameters
star = phot.Star(T=5800, R_star=695700e3, d_system=10, mu=mu) # system distance in parsecs -- values for the Sun
detector = phot.Detector(qe=0.9, cic=0.016, dark_current=1.39e-5, read_noise=120, gain=1000, # qe=0.837, dark_current=1.3e-4, read_noise=120
                   fwc=80000, conversion_gain=1.0, t=3600, D=2.36, throughput=0.38, f_pa=0.039, # throughput=0.38, f_pa=0.039,
                   wavelength=575e-9, bandwidth=8e-9)

# Set filter files
# ------------ Filter information ------------
n_filters = 3 # number of filters
path = "/Users/shasler/Documents/Projects/deconfusion_color/passbands/bessell_passbands/"
Bfilter_file = path + "Generic_Bessell.B.dat"
Rfilter_file = path + "Generic_Bessell.R.dat"
Ifilter_file = path + "Generic_Bessell.I.dat"
filter_names = ["B", "R", "I"]
# Add filter info to detector object
detector.add_filter_info(Bfilter_file, "B", lambda_units=u.Angstrom) 
detector.add_filter_info(Rfilter_file, "R", lambda_units=u.Angstrom) 
detector.add_filter_info(Ifilter_file, "I", lambda_units=u.Angstrom) 

# -------- Create text file for each filter and log file for all output --------
writers, files = {}, {}
path = "/Users/shasler/Code/deconfuser/output_files/"
output_file = f"test_deconfuser_output_{now}_{n_systems}systems_{inc_group}"

# Headers and run parameters to write to output file  
headers = ['system', 'n_planets', 'planet', 'planet_type', 'n_orbit_options', 'a', 'e', 'i', \
        'o', 'O', 'M0', 'ts', 'xyzs', 'correct_partition', 'top_partitions', \
        'partition', 'group', 'L_group_options', 'L_partition_options', \
        'L_detections', 'noisy_detections', 'detection_photon_rates', 'detection_SNRs']  
run_parameters = f"Run parameters: {n_systems} systems, {n_planets} planets, {n_epochs} epochs, {cadence} cadence (yr), {min_a} min_a (AU), {max_a} max_a (AU), {sep_a} sep_a (AU), \
                    {min_i} min_i (rad), {max_i} max_i (rad), {max_e} max_e, {tolerances} tolerances, {detector.qe} qe, {detector.cic} cic, {detector.dark_current} dark_current, \
                            {detector.read_noise} read noise, {detector.gain} gain, {detector.t} time (s), {detector.D} Diam. [m], {detector.throughput} throughput, {detector.f_pa} f_pa"

for filter_name in filter_names:
    file = f"{output_file}_{filter_name}.txt"
    try:
        f = open(path + f"{file}", "a")
        logfile = open(path + f"run_log_{now}_{n_systems}systems_{inc_group}.log", "a") 
        sys.stdout = logfile
        sys.stderr = logfile
    except FileNotFoundError:
        print('output_files directory not found. Creating directory.')
        os.mkdir('../output_files')
        f = open(path + f"{file}", "a")
        logfile = open(path + f"run_log_{now}.log", "a")
        sys.stdout = logfile
        sys.stderr = logfile

    # Write headers to text file
    writer = csv.writer(f)
    writers[filter_name] = writer # save to dict for later use
    files[filter_name] = f        # ^
    writer.writerow([run_parameters]) # save run parameters in file
    writer.writerow(headers)          # add headers to file

# Read in systems
df_system_params = pd.read_csv(systems_orb_params_file, header=0, delimiter=', ', skipfooter=skip_footer)

# ------------ Simulate planets and observations ------------ 
# Observation epochs (years) 
ts = cadence*np.arange(n_epochs)

# The correct partition of detection by planets
correct_partition = [tuple(range(i*len(ts),(i+1)*len(ts))) for i in range(n_planets)]

# To speed up computation, begin with coarsest tolerance and progress to finest:
# 1. full orbit grouping will be performed with the coarsest tolerance (i.e., recursively consider all groupings of observation)
# 2. only "full" groups that fit observation within a coarser tolerance will be fitted with a finer tolerance
# Note: "missed" detections are not simulataed here so confusion will only "arise" with full groups (n_epochs observations per planet)
tolerances = sorted(tolerances, reverse=True)
tol = tolerances[0] 

orbit_grouper = orbit_grouping.OrbitGrouper(mu, ts, min_a-tolerances[0], max_a+tolerances[0], max_e, tolerances[0], lazy_init=False)
orbit_fitters = [orbit_fitting.OrbitFitter(mu, ts, min_a-tol, max_a+tol, max_e, tol) for tol in tolerances[1:]]
orbit_fitter = orbit_fitting.OrbitFitter(mu, ts, min_a-tol, max_a+tol, max_e, tol) 

all_phases, all_SNRs, all_sigma = [], [], [] # TODO: remove
for _ in range(n_systems):
    # -------------------- Generate simulated systems --------------------
    print(f'\nSystem #{_} \n----------') 
    system = phot.System(name=str(_)) # initialize system object

    # Choose random orbit parameters for each planet
    # Get orbital parameters for each planet
    system_data = df_system_params[df_system_params['system'] == _ + 1] # system numbers are 1-indexed
    a_vals = np.array(system_data['a'].values)
    e_vals = np.array(system_data['e'].values)
    i_vals = np.array(system_data['i'].values)
    o_vals = np.array(system_data['o'].values)
    O_vals = np.array(system_data['O'].values)
    M0_vals = np.array(system_data['M0'].values)

    # Set up planets
    for n in range(n_planets):
        if n == 0:
            planet_types = ['gas_giant']
        if n == 1:
            planet_types = ['ice_giant']
        if n == 2:
            planet_types = ['rocky']
            # planet_types = ['gas_giant']
        # if n == 3:
        #     planet_types = ['rocky']
        # else:
        #     planet_types = ['rocky', 'ice_giant', 'gas_giant']

        planet = phot.Planet() # initialize planet object
        planet.add_orb_params(a_vals[n], e_vals[n], i_vals[n], o_vals[n], O_vals[n], M0_vals[n]) # add orb parameters to planet
        planet.random_planet_params(planet_types=planet_types, assign_type_by_sep=False) # assign random planet type, radius, Ag (scalar)
        system.add_planet(planet)     # add planet to system object

    # Get coordinates of planets when observed
    xs,ys,zs = sample_planets.get_observations(a_vals, e_vals, i_vals, o_vals, O_vals, M0_vals, ts, mu) 
    # TODO: remove - this is for plotting purposes ------
    t_range = 100
    ts_more = 0.05*np.arange(t_range)
    xs_more, ys_more, zs_more = sample_planets.get_observations(a_vals, e_vals, i_vals, o_vals, O_vals, M0_vals, ts_more, mu)
    obs_more = np.stack([xs_more,ys_more,zs_more], axis=2).reshape((-1,3))
    # ------ ^^^^ ------
    observations = np.stack([xs,ys,zs], axis=2).reshape((-1,3))

    # Add radially bounded astrometry error to simulated detections
    # if photometry-dependent error not included
    if sigma_photo:
        noise_r = 0
        noise_a = 0
    else:
        noise_r = tolerances[-1]*np.random.random(len(observations)) 
        noise_a = 2*np.pi*np.random.random(len(observations))
   
    observations[:,0] += noise_r*np.cos(noise_a) # x-direction error 
    observations[:,1] += noise_r*np.sin(noise_a) # y-direction error 

    # ax.scatter(observations[:,0][:3], observations[:,1][:3], marker='o', s=50, color='k', label='original position') # TODO: remove
    # ax.scatter(observations[:,0][3:6], observations[:,1][3:6], marker='s', s=50, color='k') # TODO: remove
    # ax.scatter(observations[:,0][6:], observations[:,1][6:], marker='^', s=50, color='k') # TODO: remove
    # ax.plot(obs_more[:,0][:t_range], obs_more[:,1][:t_range], color='k', alpha=0.7)
    # ax.plot(obs_more[:,0][t_range:t_range*2], obs_more[:,1][t_range:t_range*2], color='k', alpha=0.7)
    # ax.plot(obs_more[:,0][t_range*2:], obs_more[:,1][t_range*2:], color='k', alpha=0.7)
    # xs_original = observations[:,0].flatten()
    # print('xs_original: ', xs_original)
    # ys_original = observations[:,1].flatten()

    # Calculate photometry of simulated system (these are your "observations")
    all_coords = [] 
    for ip in range(n_planets):
        all_coords.append(list(map(list, observations[ip*len(ts):(ip+1)*len(ts)])))
    all_coords = np.asarray(all_coords)

    # ------- Get filter info for first set of observations -------
    kk = 0 
    for filter_name in filter_names:
        if kk == 0: # Check whether or not this is the first filter for observation
            first_filter = True # Controls whether or not to pull new albedo spectra for the planets
        else:
            first_filter = False # False if you're on a second round of observations, otherwise you will have too many/incorrect spectra in the planet objects
            
        filter_data, lambda_min, lambda_max = detector.get_filter_info(filter_name) # assigns filter info to detector parameters (wavelength, bandwidth)
        print(f"\n\nFilter: {filter_name} [{lambda_min} - {lambda_max} um]")

        # ---- get noisy and not noisy photometric detections for simulated system ----
        noisy_detections, detections_photon_rates, SNRs, phases = phot.get_detections_counts_color(n_epochs, xyzs=all_coords, 
                                                                                                Star=star, System=system, 
                                                                                                Detector=detector, 
                                                                                                filter_name=filter_name,
                                                                                                spectrum_dir=spectrum_dir,
                                                                                                first_filter=first_filter)
        print('noisy_detections: ', noisy_detections)
        print("detections_photon_rates: ", detections_photon_rates)
        print('SNRs: ', SNRs)
        print('phases: ', phases)

        if sigma_photo: # if true, add astrometric uncertainty to observations due to simulated photometry
            # Calculate error in x,y directions due to planet signal
            sigma_AU = phot.astro_photo_uncertainty(SNRs, detector, star, SNR_low_lim=2, sigma_lim=0.015) # TODO: remove hard-coded values
            print('sigma_AU.flatten(): ', sigma_AU.flatten())
            # Add uncertainty to coordinates as gaussian with standard deviation of sigma
            observations_werr = observations
            observations_werr[:,0] = np.random.normal(observations[:,0], sigma_AU.flatten())
            observations_werr[:,1] = np.random.normal(observations[:,1], sigma_AU.flatten())
            if kk == 0: # only add astro uncertainty to all points on the first time through the loop 
                observations = observations_werr
        kk += 1
        #     # TODO: remove plotting
        #     ax.scatter(observations[:,0][:3], observations[:,1][:3], marker='o', s=50, edgecolor='r', facecolor='r', 
        #                linewidth=1, alpha=0.5, label='with photo/astro err', zorder=3)
        #     ax.scatter(observations[:,0][3:6], observations[:,1][3:6], marker='s', s=50, edgecolor='r', facecolor='r', 
        #                linewidth=1,  alpha=0.5, zorder=3)
        #     ax.scatter(observations[:,0][6:], observations[:,1][6:], marker='^', s=50, edgecolor='r', facecolor='r', 
        #                linewidth=1,  alpha=0.5, zorder=3) 

        #     for xi, yi, zi in zip(xs_original, ys_original, sigma_AU.flatten()):
        #         circle = Circle((xi, yi), zi*2, edgecolor='red', facecolor='none', lw=2, linestyle='dashed', alpha=0.9)
        #         # zi*2 = 2-sigma uncertainty radius size
        #         ax.add_patch(circle)

        # ax.scatter(0, 0, marker='*', color='gold', s=100)
        # ax.set_xlabel('x (AU)')
        # ax.set_ylabel('y (AU)')
        # ax.set_aspect('equal')
        # plt.legend()
        # plt.title('Detections with vs. without\njoint astro/photo error')
        # plt.show()

        if verbose:
            print("\nts =", list(ts)) 
            for ip in range(n_planets): # for every planet
                print("\nplanet ", ip+1, f"({system.planets[ip].type}): ")
                print("a,e,i,o,O,M0 = ", (a_vals[ip],e_vals[ip],i_vals[ip],o_vals[ip],O_vals[ip],M0_vals[ip])) # true orbital parameters for each planet
                print("xyzs =", list(map(list, observations[ip*len(ts):(ip+1)*len(ts)]))) # true coordinates of detection for each planet
                print("photon_rates = ", detections_photon_rates[ip]) # calculated photon rates for each planet detection (format: [detection1, detection2, ..., detectionN])
                print("noisy_detections = ", noisy_detections[ip]) # noisy planet detections (format: [array([planet1_detection1, ..., planet1_detectionN])], ..., array([planetM_detection1, ..., planetM_detectionN])])
                print("SNRs = ", SNRs[ip]) # SNR for each planet detection (format: [SNR1, SNR2, ..., SNRN])
        
               
        # output simulated planet info to text file
        for ip in range(n_planets):
            planet_params = [_, n_planets, ip+1, system.planets[ip].type, np.NaN, a_vals[ip], e_vals[ip], i_vals[ip], o_vals[ip], O_vals[ip], M0_vals[ip], ts, list(map(list, \
                            observations[ip*len(ts):(ip+1)*len(ts)])), correct_partition, None, None, None, None, None, noisy_detections[ip], \
                            detections_photon_rates[ip], SNRs[ip]] # system #, # planets simulated, planet #, a, e, i, o, O, M0, confused?, ts, xyzs, correct_partition, top_partitions, group, 'L_detections', 'L_group_options', 'L_partition_options', 'noisy_detections', 'detection_photon_rates' 
            writers[filter_name].writerow(planet_params)

        # all detection times for all obesrvations
        all_ts = np.tile(ts, n_planets)

        # -------------------- Do the orbit fitting --------------------
        # get all possible (full or partial) groupings of detection by orbits that fit them with the coarsest tolerance
        groupings = orbit_grouper.group_orbits(observations, all_ts)
        print('groupings: ', groupings)

        # select only groupings that include all epochs (these will be most highly ranked, so no need to check the rest)
        groupings = [g for g in groupings if len(g) == n_epochs] 
        
        # Check for spurious orbits and repeat for finer tolerances
        for j in range(len(tolerances)):
            found_correct = sum(cg in groupings for cg in correct_partition)

            print('-------------------------------------------------------------')
            print("Tolerance %f: found %d correct and %d spurious orbits out of %d"%(tolerances[j], found_correct, len(groupings) - found_correct, n_planets))
            if verbose:
                print("Tolerance %f:"%(tolerances[j]), groupings)

            # Find all partitions of observations to exactly n_planets groups
            # Note that since all partial grouping were filtered out, all partitions will have exactly n_planets groups
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
                for partition in top_partitions: 
                    print('partition: ', partition)
                    L_partition_options = [] # group options list
                    ii = 0                    # for getting correct noisy counts per planet orbit option
                    for group in partition:  # which data points lie on the orbit
                        print('\ngroup: ', group)
                        L_group_options = [] # orbit options list
                        group_orbit_parameters = []
                        k = 0                # keep track of how many orbit options per group

                        # Get groups + orbital parameters to rank with photometry
                        for err, parameters in orbit_fitter.fit(observations[group]): 
                            print('\nParameters: ', parameters)
                            print('err: ', err) # Added to return fit errors

                            # Phase information section
                            # Phase info is buried in likelihood function -- add as a return parameter in likelihood.py if you want to back it out
                            # Calculate likelihood of orbit option
                            L_orbit, L_detections = L.get_L_orbit(n_detections=n_epochs,
                                                                a=parameters[0], e=parameters[1],
                                                                i=parameters[2],
                                                                o=parameters[3],
                                                                O=parameters[4],
                                                                M0=parameters[5],
                                                                ts=ts,
                                                                noisy_counts=noisy_detections[ii],
                                                                Star=star, Planet=system.planets[ii], Detector=detector, 
                                                                use_color_mode=True, filter_name=filter_name,
                                                                spectrum_dir=spectrum_dir)
                            
                            print(f'L_orbit: {L_orbit}')    # Likelihood of entire orbit (alldetections) 
                            L_group_options.append(L_orbit) # save L of each orbit option per group
                            print(f'L_group_options: {L_group_options}') # Likelihood of each orbit option in a group 
                            group_orbit_parameters.append(parameters)
                            k += 1 # track number of orbit options

                        L_partition_options.append(L_group_options) # Likelihood of all orbit options in a partition -- will be empty if i = nan

                        # Write confused orbit options to output file
                        a_s = [orbit[0] for orbit in group_orbit_parameters] # separate orbital parameters for all orbit options in a group
                        e_s = [orbit[1] for orbit in group_orbit_parameters]
                        i_s = [orbit[2] for orbit in group_orbit_parameters]
                        o_s = [orbit[3] for orbit in group_orbit_parameters]
                        O_s = [orbit[4] for orbit in group_orbit_parameters]
                        M0_s = [orbit[5] for orbit in group_orbit_parameters]
                        option_parameters = [_, n_planets, ii+1, system.planets[ii].type, k, a_s, e_s, i_s, o_s, O_s, M0_s, ts, None, correct_partition, top_partitions, partition, \
                                            group, L_group_options, L_partition_options, L_detections, None, None] # parameters for writing to text file
                        
                        writers[filter_name].writerow(option_parameters)
                        ii += 1 # advance to next detected planet in the system for comparison

                # ------------------------------------------------------------------
                print("Tolerance %f: found %d spurious \"good\" paritions of detections by planets (confusion)"%(tolerances[j], len(top_partitions) - 1))
                if verbose:
                    print("Tolerance %f:"%(tolerances[j]), top_partitions)

            # # move to a finer tolerance
            # if j < len(tolerances) - 1:
            #     #only keep groupings that cna be fitted with an orbit with the finer tolerance
            #     groupings = [g for g in groupings if any(err < tolerances[j+1] for err in orbit_fitters[j].fit(observations[list(g)], only_error=True))]

        # TODO: remove -----
        all_phases.append(phases)
        all_SNRs.append(SNRs)
        if sigma_photo:
            all_sigma.append(sigma_AU)
        # ^ --------------

# Re-rank systems with photometry in each filter
try: # create ranking files directory if it doesn't exist
    os.makedirs("output_files/ranking_files", exist_ok=True) 
except OSError as error: 
    print("ranking_files directory cannot be created.")

# For some reason this occasionally does not work and you have to run the ranking functions in a separate file
# Check if output files exist before ranking
exists, files = utils.filter_files_exist(path="output_files/", base_output_filename=f"{output_file}_", filter_names=filter_names)
if exists:
    filter_dfs = []
    for filter_name in filter_names:
        output_filename = f"{output_file}_{filter_name}.txt"
        print('output_filename with path: ', f"output_files/{output_filename}")
        confused_systems = ranking.PhotometryRanking(filepath=f"output_files/{output_filename}", n_planets=n_planets)
        df_confused = confused_systems.get_top_group_options()  # iterate over options with multiple groups
        df_ranked = confused_systems.top_ranked_partition()     # Get top ranked partition in each system
        df_recombined = confused_systems.combine_and_cleanup(save_file=False)  # Combine original and ranked dataframes
        # If you want to calculate percent difference between simulated and fit orbits:
        df_final = confused_systems.orbit_percent_diff()
        df_final_recombined = confused_systems.final_recombined()
        ntopa_best, ntope_best, ntopi_best = ranking.check_ranking_by_planet(df_final_recombined, n_planets)
        filter_dfs.append(df_final_recombined) # append each filter df to a list
    
    df_merged = ranking.combine_filter_dfs(filter_dfs[0], filter_dfs[1], filter_dfs[2], filter_names[0],
                                            filter_names[1], filter_names[2])
    
    ranking.save_df(df_merged, ranking_path + f"systems_ranked_mergedFilters_{now}.txt")
    print('\nPhotometry ranking complete.')
else:
    print(f"Output files {files} do not exist.")

end  = datetime.now()
runtime = end - start
runtime_string = [f"Run time: {runtime} s"]
writer.writerow(runtime_string) # write run time to end of file

f.close()       
logfile.close() 
sys.stdout = sys.__stdout__ # reset standard output to terminal
sys.stderr = sys.__stderr__ # reset error output to terminal
print(f'Output written to: output_files/')
print([f for f in files])
print(" & " + ranking_path)
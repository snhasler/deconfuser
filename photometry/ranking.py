'''
@author: S. N. Hasler

Functions to read text files output by test_deconfuser_wPhotometry.py, pull information into dataframe, and return new 
dataframe with partitions re-ranked and highest ranked partitions marked as True.

How to use:
    1. Read text file into dataframe with pd.read_csv()
        df = pd.read_csv(file_path)
    2. df_confused_only = get_top_group_options()
        Return only highest likelihood groups for groups with multiple orbit options and remove
        simulated planets from dataframe.
    3. Get number of planets simulated from first row of dataframe (the same for all columns when using test_deconfuser_addphase.py)
        n_planets = df['n_planets'][0]
    3. df_top_ranked = top_ranked_partition(df_confused_only, n_planets)
        Gets top ranked partition for all systems and returns dataframe with assigned ranking ids.
    4. df_recombined = combine_and_cleanup(df, df_top_ranked)
        Combines original dataframe and dataframe with ranked partitions. Removes unnecessary/duplicate columns.
'''
import pandas as pd
import numpy as np
from tqdm import tqdm 

def percent_difference(val1, val2):
    '''
    Returns the percent difference of two values.
    '''
    return (np.abs(val1 -  val2) / ((val1 +  val2) / 2) ) * 100

class Planet:
    '''
    Planet class to store orbital parameters
    '''
    def __init__(self, number, a, e, i, omega, Omega, M0):
        self.number = number
        self.a = a
        self.e = e
        self.i = i
        self.o = omega
        self.O = Omega
        self.M0 = M0

    def __str__(self):
        return f"Planet #{self.number}: a = {self.a} au, e = {self.e}, i = {self.i} rad, o = {self.omega}, O = {self.Omega}, M0 = {self.M0}"

class PlanetarySystem:
    '''
    System class to store n_planets and system information
    '''
    def __init__(self, number):
        self.number = number
        self.planets = []
    
    def add_planet(self, planet):
        if isinstance(planet, Planet):
            self.planets.append(planet)
        else:
            print("Invalid planet object. Please provide a valid Planet object.")
    
    def __str__(self):
        system_info = f"System #{self.number}\n"
        planets_info = f"\n".join([str(planet) for planet in self.planets])
        return system_info + planets_info

class PhotometryRanking:
    """
    A class for updating orbit rankings using photometric information
    """
    def __init__(self, filepath, n_planets):
        '''
        Constructor

        Parameters
        ----------
        filepath : string
            Full path to output text file from running simulation with 
            the deconfuser. 
        n_planets : int
            Number of planets simulated per system in MC deconfuser run.
        
        '''
        self.filepath = filepath
        self.n_planets = n_planets

        # Read file into dataframe
        self.df = pd.read_csv(filepath, skiprows=1, skipfooter=1, engine='python')
    
    def get_top_group_options(self):
        '''
        Iterates over dataframe of systems output by the deconfuser with 
        photometry likelihood values to sort groups with multiple orbit options by 
        orbits with the highest likelihoods. 
        Creates new columns to save orbital parameters and likelihoods for
        the highest likelihood options in each group. 
        Returns new dataframe with new columns.

        '''
        df = self.df
        print(f"Sorting group options for each system...")
        for index, row in tqdm(df.iterrows()):
            # if there is more than one orbit option in a group, 
            # need to pick the highest likelihood option for partition
            if row['n_orbit_options'] > 1:
                L_group_options = eval(row['L_group_options'])   # convert L_group_options to list
                n_orbit_options = int(row['n_orbit_options'])    # get number of orbit options in the group

                # separate group_options into tuples with (likelihood, index)
                indices = list(range(n_orbit_options))                     # get indices for tuple
                group_options = list(zip(L_group_options, indices))        # pair group options + indices 
                sorted_group_options = sorted(group_options, reverse=True) # sort group_options by highest likelihood
                top_group = sorted_group_options[0]                        # keep only highest likelihood option
                highest_L_index = top_group[1]

                # save highest likelihood option
                df.loc[index, 'L_orbit'] = top_group[0]                    # add new column for highest likelihood orbit -- keep highest likelihood
                df.loc[index, 'id_top_orbit_in_group'] = highest_L_index   # save index for highest L orbit in group
                df.loc[index, 'top_a'] = eval(df.loc[index, 'a'])[highest_L_index]  # copy parameters for highest L option to new top_'parameter' column
                df.loc[index, 'top_e'] = eval(df.loc[index, 'e'])[highest_L_index]
                df.loc[index, 'top_i'] = eval(df.loc[index, 'i'])[highest_L_index]
                df.loc[index, 'top_o'] = eval(df.loc[index, 'o'])[highest_L_index]
                df.loc[index, 'top_O'] = eval(df.loc[index, 'O'])[highest_L_index]
                df.loc[index, 'top_M0'] = eval(df.loc[index, 'M0'])[highest_L_index]

            elif row['n_orbit_options'] == 1: # for groups with only one orbit option
                df.loc[index, 'L_orbit'] = eval(df.loc[index, 'L_group_options'])[0]
                df.loc[index, 'id_top_orbit_in_group'] = 0
                df.loc[index, 'top_a'] = eval(df.loc[index, 'a'])[0]  # copy a to top_a for groups with only one orbit option
                df.loc[index, 'top_e'] = eval(df.loc[index, 'e'])[0]
                df.loc[index, 'top_i'] = eval(df.loc[index, 'i'])[0]
                df.loc[index, 'top_o'] = eval(df.loc[index, 'o'])[0]
                df.loc[index, 'top_O'] = eval(df.loc[index, 'O'])[0]
                df.loc[index, 'top_M0'] = eval(df.loc[index, 'M0'])[0]
            
        # remove columns that we don't need anymore (original orbital parameter columns)
        df = df.drop(columns=['ts', 'xyzs', 'correct_partition', 'noisy_detections', \
                            'detection_photon_rates', 'L_detections', 'L_partition_options', 'L_group_options'])
        # finally, remove rows for simulated planets and only keep orbit options
        df = df[~df['n_orbit_options'].isnull()]
        
        self.df_confused_options = df
    
        return self.df_confused_options
    

    def top_ranked_partition(self):
        '''
        Function to assign the top ranked partition for each system. 
        Uses the dataframe of only confused system options returned
        by the get_top_group_options function.
        
        '''
        system_numbers = list(set([num for num in self.df_confused_options['system']]))
        partition_list = []

        print("Calculating likelihood of each partition...")
        for index, row in tqdm(self.df_confused_options.iterrows()):    
            if row['planet'] == self.n_planets:                                   
                partition_list.append(row['L_orbit'])  # save L_orbit for group
                L_partition = np.prod(partition_list)
                self.df_confused_options.loc[index, 'L_partition_list'] = str(partition_list) # append partition list to end of df
                self.df_confused_options.loc[index, 'L_partition'] = str(L_partition)
                partition_list = []                     # reset partition_list
            else:
                partition_list.append(row['L_orbit'])  

        print("Sorting partitions by likelihoods...")
        for num in tqdm(system_numbers): # for each system
            system_df = self.df_confused_options[self.df_confused_options['system'] == num]  
            partition_options = np.unique([partition for partition in system_df['partition']]) # get possible partitions for system
            L_partitions = [float(row['L_partition']) for index, row in system_df[~system_df['L_partition'].isnull()].iterrows()] # get L_partition for each partition
            ids = list(range(len(L_partitions)))
            
            # Put likelihood and partitions into tuple
            possible_partitions = list(zip(L_partitions, partition_options))     # create tuples of (partition likelihood, partition)       
            sorted_partitions = sorted(possible_partitions, reverse=True)        # sort tuples from highest -> lowest likelihood
            numbered_sorted_partitions = list(zip(sorted_partitions, ids))       # add ranking ids to sorted partitions
            highest_L_partition = sorted_partitions[0][1]                        # save partition of highest likelihood

            rankings = []
            for partition in numbered_sorted_partitions:
                part_tuple = (partition[0][1], partition[1])                     # create new tuple to keep rankings with partitions
                rankings.append(part_tuple)
            
            for partition in rankings: # add ranking value to df
                for index, row in self.df_confused_options.iterrows():
                    if row['partition'] == partition[0] and row['system'] == num:
                        self.df_confused_options.loc[index, 'ranking'] = partition[1] # add ranking number to df

                        # Mark highest likelihood partition in dataframe as True, the rest as False
                        if self.df_confused_options.loc[index, 'partition'] == highest_L_partition:
                            self.df_confused_options.loc[index, 'top_ranked_partition'] = True
                        else:
                            self.df_confused_options.loc[index, 'top_ranked_partition'] = False

        return self.df_confused_options # All confused options now ranked
    
    def combine_and_cleanup(self, save_file=False, save_path=None):
        '''
        Combine original dataframe and dataframe with ranked partitions.
        Also removes duplicate/unecessary columns.
        Saves file if save_file is True

        Parameters
        ----------
        save_file : bool, optional
            Whether or not you want to save the dataframe to a text file, by default False
        save_path : _type_, optional
            Path to save text file to. Recommend to save where original text files are located., by default None
        
        '''
        df_recombined = self.df.join(self.df_confused_options, how='outer', lsuffix='_original')
        df_recombined = df_recombined.drop(columns=['system', 'n_planets', 'planet', 'n_orbit_options', 'top_partitions_original', \
                           'partition_original', 'group_original', 'L_partition_options'])
    
        if save_file == True:
            df_recombined.to_csv(save_path, header=True, index=None)

        self.df_recombined = df_recombined

        return df_recombined
    
    def orbit_percent_diff(self):
        '''
        Calculate the percent difference between the simulated orbital
        parameters and the confused partitions.

        Parameters
        ----------
        confused_system_numbers : numpy.ndarray 
            Array of integers corresponding to the confused system numbers.
        '''
        confused_system_numbers = self.df['system'].unique()

        # Separate out simulated systems
        df_confused_sim = self.df_recombined[(self.df_recombined['system_original'].isin(confused_system_numbers)) & (self.df_recombined['n_orbit_options_original'].isna())]
        # Separate out the confused partitions for each system
        df_confused_partitions = self.df_recombined[(self.df_recombined['system_original'].isin(confused_system_numbers)) & (~self.df_recombined['n_orbit_options_original'].isna())]

        # For each confused system, get the number of confused partitions
        for system_num in confused_system_numbers:
            simulated_system = PlanetarySystem(system_num) # create system object

            df_simulated = df_confused_sim[df_confused_sim['system_original'] == system_num]
            df_confused = df_confused_partitions[df_confused_partitions['system_original'] == system_num]

            # For the simulated system, grab the orbital parameters
            for index, row in df_simulated.iterrows():
                simulated_planet = Planet(row['planet_original'], row['a_original'], row['e_original'], row['i_original'], 
                                    row['o_original'], row['O_original'], row['M0_original'])
                simulated_system.add_planet(simulated_planet) # add planets to system

            # Check to see how many confused partitions there are
            num_partitions = len(df_confused) / self.n_planets

            # For each confused option, compare the orbital parameters to the planet with the
            # corresponding number in the simulated system
            for index, row in df_confused.iterrows():
                planet_number = row['planet_original']

                # Compare each parameter
                sim_a = float(simulated_system.planets[planet_number-1].a)
                sim_e = float(simulated_system.planets[planet_number-1].e)
                sim_i = float(simulated_system.planets[planet_number-1].i)
                sim_o = float(simulated_system.planets[planet_number-1].o)
                sim_O = float(simulated_system.planets[planet_number-1].O)
                sim_M0 = float(simulated_system.planets[planet_number-1].M0)
                        
                pdiff_a = percent_difference(sim_a, float(row['top_a']))
                pdiff_e = percent_difference(sim_e, float(row['top_e']))
                pdiff_i = percent_difference(sim_i, float(row['top_i']))
                pdiff_o = percent_difference(sim_o, float(row['top_o']))
                pdiff_O = percent_difference(sim_O, float(row['top_O']))
                pdiff_M0 = percent_difference(sim_M0, float(row['top_M0']))

                # Add percent difference to columns
                df_confused_partitions.loc[index, 'a_%diff'] = pdiff_a
                df_confused_partitions.loc[index, 'e_%diff'] = pdiff_e
                df_confused_partitions.loc[index, 'i_%diff'] = pdiff_i
                df_confused_partitions.loc[index, 'o_%diff'] = pdiff_o
                df_confused_partitions.loc[index, 'O_%diff'] = pdiff_O
                df_confused_partitions.loc[index, 'M0_%diff'] = pdiff_M0

        self.df_confused_final = df_confused_partitions

        return df_confused_partitions
    
    def final_recombined(self, save_file=False, save_path=None):
        '''
        Final recombination after percent differences have been calculated

        Parameters
        ----------
        save_file : bool, optional
            Flag to save the file or not., by default False
        save_path : _type_, optional
            Path to save file to. Must be defined if save_file = True;
            By default None
        '''
        colnames = [col for col in self.df_confused_final.columns]
        cols_to_remove = colnames[:len(colnames)-6] # duplicate cols to remove
        
        df_final = self.df_recombined.join(self.df_confused_final, how='outer', lsuffix='1')
        df_final = df_final.drop(columns=cols_to_remove)

        if save_file == True:
            df_final.to_csv(save_path, header=True, index=None)

        self.df_final = df_final

        return df_final
    
def check_ranking_by_planet(df_confused_final, n_planets):
    '''
    Cross-check percent difference in a, e, i against the systems that were
    top ranked to answer: How many planets with top-ranked orbits were
    also the planets with the lowest percent difference out of the partitions?

    Also updates columns in df_confused_final

    Parameters
    ----------
    df_confused_final : pandas.DataFrame
        Final dataframe of confused options after ranking
    n_planets : int, optional
        Number of planets simulated per system

    Returns
    -------
    n_top_a_is_best, n_top_e_is_best, n_top_i_is_best : ints
        The number of systems for which the top-ranked system also had the lowest
        percent difference compared to the simulated system for semimajor axis, 
        eccentricity, and inclination

    '''
    systems = df_confused_final['system_original'].unique()
    top_ranked_a_is_best, top_ranked_e_is_best, top_ranked_i_is_best = [], [], []
    df_confused_final.loc[:,'best_a'] = False # initialize new column to track which orbits fit the semimajor axis best out of the bunch
    df_confused_final.loc[:,'best_e'] = False # initialize new column to track which orbits fit the semimajor axis best out of the bunch
    df_confused_final.loc[:,'best_i'] = False # initialize new column to track which orbits fit the semimajor axis best out of the bunch

    for system in systems:
        df_system = df_confused_final[df_confused_final['system_original'] == system] # get df of just single system
        for planet in range(n_planets):
            planet += 1
            df_single_planet = df_system[df_system['planet_original'] == planet] # get df of just single planet in the system

            # Get the minimum value for a, e, i_%diff cols
            min_a = df_single_planet['a_%diff'].min() # Gets min value
            min_e = df_single_planet['e_%diff'].min()
            min_i = df_single_planet['i_%diff'].min()
            # Get indices of rows with the min values
            min_a_ids = df_single_planet.index[df_single_planet['a_%diff'] == min_a].to_list()
            min_e_ids = df_single_planet.index[df_single_planet['e_%diff'] == min_e].to_list()
            min_i_ids = df_single_planet.index[df_single_planet['i_%diff'] == min_i].to_list()
            
            # Track index of which orbit is a better fit in semimajor axis for each planet
            df_confused_final.loc[min_a_ids, 'best_a'] = True
            df_confused_final.loc[min_e_ids, 'best_e'] = True
            df_confused_final.loc[min_i_ids, 'best_i'] = True 

            # check if the min percent difference is also the top ranked system
            does_topranked_have_mina = df_single_planet.loc[df_single_planet['a_%diff'] == min_a, 'top_ranked_partition'].eq(True).any()
            does_topranked_have_mine = df_single_planet.loc[df_single_planet['e_%diff'] == min_e, 'top_ranked_partition'].eq(True).any()
            does_topranked_have_mini = df_single_planet.loc[df_single_planet['i_%diff'] == min_i, 'top_ranked_partition'].eq(True).any()

            # Keep track of the top ranked which are the best option and the total number (total should be 30)
            if does_topranked_have_mina:
                top_ranked_a_is_best.append(does_topranked_have_mina)
            if does_topranked_have_mine:
                top_ranked_e_is_best.append(does_topranked_have_mine)
            if does_topranked_have_mini:
                top_ranked_i_is_best.append(does_topranked_have_mini)
    
        # C# Count the number of "True" values for every n_planets rows
        best_a_counts = df_confused_final['best_a'].groupby(df_confused_final.index // n_planets).sum() # This sums up the True counts in the "best_a" col for every 3 rows
        best_e_counts = df_confused_final['best_e'].groupby(df_confused_final.index // n_planets).sum()
        best_i_counts = df_confused_final['best_i'].groupby(df_confused_final.index // n_planets).sum()

        df_confused_final['best_a_count'] = df_confused_final.index.map(lambda i: best_a_counts[i // n_planets]) 
        df_confused_final['best_e_count'] = df_confused_final.index.map(lambda i: best_e_counts[i // n_planets]) 
        df_confused_final['best_i_count'] = df_confused_final.index.map(lambda i: best_i_counts[i // n_planets]) 

    n_top_a_is_best = len(top_ranked_a_is_best)
    n_top_e_is_best = len(top_ranked_e_is_best)
    n_top_i_is_best = len(top_ranked_i_is_best)

    return n_top_a_is_best, n_top_e_is_best, n_top_i_is_best

def print_percent_best(ntop_param_best, param_str, total):
    '''
    Printing for results of check_ranking_by_planet()

    Parameters
    ----------
    ntop_param_best : int
        Number of top parameters that are the "best" ranked per system 
        Output of check_ranking_by_planet()
    param_str : str
        Parameter that you're checking, either 'a', 'e', or 'i'
    total : int
        Total number of systems 
    '''
    print(f'Percent of planets for which top ranked {param_str} is best: ', ntop_param_best / total)

def combine_filter_dfs(df1, df2, df3, filtname1, filtname2, filtname3):
    """
    Combine three dataframes after they have been passed through check_ranking_by_planet().
    Creates new merged dataframe with each filter column labeled separately (changes
    column headers). 
    Allows for viewing likelihood due to each filter at once

    Parameters
    ----------
    df1 : pandas.DataFrame
        Dataframe from first filter
    df2 : pandas.DataFrame
        Dataframe from second filter
    df3 : pandas.DataFrame
        Dataframe from third filter
    filtname1 : str
        Name of first filter
    filtname2 : str
        Name of second filter
    filtname3 : str
        Name of third filter

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame containing all information, with each
        filter labeled separately
    """

    # Rename columns in each filter df
    df_filt1 = df1.rename(columns={"L_detections": f"L_detections_{filtname1}",
                                   "L_group_options": f"L_group_options_{filtname1}",
                                   'L_orbit': f"L_orbit_{filtname1}", 
                                   "L_partition": f"L_partition_{filtname1}",
                                   "L_partition_list": f"L_partition_list_{filtname1}",
                                   "ranking": f"ranking_{filtname1}",
                                   "top_ranked_partition": f"top_ranked_partition_{filtname1}",
                                   "best_a": f"best_a_{filtname1}",
                                   "best_e": f"best_e_{filtname1}",
                                   "best_i": f"best_i_{filtname1}",
                                   "best_a_count": f"best_a_count_{filtname1}",
                                   "best_e_count": f"best_e_count_{filtname1}",
                                   "best_i_count": f"best_i_count{filtname1}"})
    
    df_filt2 = df2.rename(columns={"L_detections": f"L_detections_{filtname2}",
                                   "L_group_options": f"L_group_options_{filtname2}",
                                   'L_orbit': f"L_orbit_{filtname2}", 
                                   "L_partition": f"L_partition_{filtname2}",
                                   "L_partition_list": f"L_partition_list_{filtname2}",
                                   "ranking": f"ranking_{filtname2}",
                                   "top_ranked_partition": f"top_ranked_partition_{filtname2}",
                                   "best_a": f"best_a_{filtname2}",
                                   "best_e": f"best_e_{filtname2}",
                                   "best_i": f"best_i_{filtname2}",
                                   "best_a_count": f"best_a_count_{filtname2}",
                                   "best_e_count": f"best_e_count_{filtname2}",
                                   "best_i_count": f"best_i_count{filtname2}"})
    
    df_filt3 = df3.rename(columns={"L_detections": f"L_detections_{filtname3}",
                                   "L_group_options": f"L_group_options_{filtname3}",
                                   'L_orbit': f"L_orbit_{filtname3}", 
                                   "L_partition": f"L_partition_{filtname3}",
                                   "L_partition_list": f"L_partition_list_{filtname3}",
                                   "ranking": f"ranking_{filtname3}",
                                   "top_ranked_partition": f"top_ranked_partition_{filtname3}",
                                   "best_a": f"best_a_{filtname3}",
                                   "best_e": f"best_e_{filtname3}",
                                   "best_i": f"best_i_{filtname3}",
                                   "best_a_count": f"best_a_count_{filtname3}",
                                   "best_e_count": f"best_e_count_{filtname3}",
                                   "best_i_count": f"best_i_count{filtname3}"})

    # Drop duplicate columns
    cols2drop = ['system_original', 'n_planets_original', 'planet_original', \
                 'planet_type_original', 'n_orbit_options_original', 'a_original', \
                 'e_original', 'i_original', 'o_original', 'O_original', 'M0_original', \
                 'ts', 'xyzs', 'correct_partition', 'noisy_detections', \
                 'detection_photon_rates', 'detection_SNRs_original', 'id_top_orbit_in_group_original', \
                 'top_a_original', 'top_e_original', 'top_i_original', 'top_o_original', \
                 'top_O_original', 'top_M0_original', 'planet_type', 'a', 'e', 'i', 'o', \
                 'O', 'M0', 'top_partitions', 'partition', 'group', 'detection_SNRs', \
                 'id_top_orbit_in_group', 'top_a', 'top_e', 'top_i', 'top_o', 'top_O', \
                 'top_M0', 'a_%diff', 'e_%diff', 'i_%diff', 'o_%diff', 'O_%diff', 'M0_%diff']
    df_filt1_reduced = df_filt1.drop(columns=cols2drop)
    df_filt2_reduced = df_filt2[["system_original", "planet_original", "a_original", \
                                  f"L_detections_{filtname2}", f"L_group_options_{filtname2}", \
                                 f"L_orbit_{filtname2}", f"L_partition_{filtname2}", \
                                    f"L_partition_list_{filtname2}", f"ranking_{filtname2}", \
                                        f"top_ranked_partition_{filtname2}", f"best_a_{filtname2}", \
                                            f"best_e_{filtname2}", f"best_i_{filtname2}", \
                                                f"best_a_count_{filtname2}", \
                                                    f"best_e_count_{filtname2}", \
                                                        f"best_i_count{filtname2}"]]
    
    df_filt3_reduced = df_filt3[["system_original", "planet_original", "a_original", \
                                 f"L_detections_{filtname3}", f"L_group_options_{filtname3}", \
                                 f"L_orbit_{filtname3}", f"L_partition_{filtname3}", \
                                    f"L_partition_list_{filtname3}", f"ranking_{filtname3}", \
                                        f"top_ranked_partition_{filtname3}", f"best_a_{filtname3}", \
                                            f"best_e_{filtname3}", f"best_i_{filtname3}", \
                                                f"best_a_count_{filtname3}", \
                                                    f"best_e_count_{filtname3}", \
                                                        f"best_i_count{filtname3}"]]
    
    # Concatenate dfs
    df_merged = df_filt1.merge(df_filt2_reduced, on=['system_original', 'planet_original', 'a_original'], suffixes=('', '_dup'),
                               how='inner').merge(df_filt3_reduced, on=['system_original', 'planet_original', 'a_original'],
                                                  suffixes=('', '_dup2'), how='inner')
    # Drop duplicate cols
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    return df_merged

def save_df(df, filename, sep=','):
    '''
    Save dataframe to filename (full path)
    '''
    df.to_csv(filename, sep=sep, index=None, header=True)
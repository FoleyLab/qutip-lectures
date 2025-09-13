import warnings
from scipy.signal import find_peaks
import numpy as np



def declutter_E_array(E_array, dipoles,  r_data, discontinuity_threshold_std = 1, energy_diff_threshold_std = 2,  num_to_declutter = 2, num_dipole_iterations = 50, num_energy_iterations = 0):


    #dipole array orderd as [numroots][numroots][dipole vector][bondlength]
    E_array = np.copy(E_array)
    new_E_array = np.zeros_like(E_array)


    d_reshaped= np.zeros_like(E_array)

    def vector_magnitude(vector):
        return np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)


    for q in range(dipoles.shape[3]):
        for i in range(dipoles.shape[0]):
            for j in range(dipoles.shape[1]):
                if i == j:
                    d_reshaped[q][i] = vector_magnitude(dipoles[i,i,:,q])
  
    dipoles = np.copy(d_reshaped)
    new_dipoles = np.zeros_like(dipoles)

    # plt.plot(dipoles)
    # plt.show()



    for q in range(0, 5):
        #dipole discontinuities
        for i in range(0,num_to_declutter):
            previous_intersection = 0
            for z in range(0,num_dipole_iterations):
                for j in range(i+1, E_array.shape[1]):
                    array1 = E_array[:, i]
                    array2 = E_array[:, j]

                    dipole_array1 = dipoles[:, i] [previous_intersection:]
                    dipole_array2 = dipoles[:, j] [previous_intersection:]


                    #Only want array from previous intersection so it doesnt get recrossed
                    array1_from_previous_intersection = array1[previous_intersection:]
                    array2_from_previous_intersection = array2[previous_intersection:]
                    #trying to determine how close two energy surfaces get, if they get very close this some evidence that they crossover
                    diff_array1 = np.diff(array1_from_previous_intersection)
                    diff_array2 = np.diff(array2_from_previous_intersection)
                    std1 = np.std(np.abs(diff_array1))
                    mean1 = np.mean(np.abs(diff_array1))
                    std2 = np.std(np.abs(diff_array2))
                    mean2 = np.mean(np.abs(diff_array2))
                    energy_diff_threshold = ((mean1 + mean2)/2) + (((std1+std2)/2) * energy_diff_threshold_std)
                    #find closest points
                    #print(np.abs(np.abs(array1[previous_intersection:]) - np.abs(array2[previous_intersection:])))
                    closest_indices =np.where(np.abs(np.abs(array1[previous_intersection:]) - np.abs(array2[previous_intersection:])) < energy_diff_threshold)
                    try:

                        #if energies are close and there are two matcjhing discontinuities in dipole arrary 
                        dipoles_diff_1 = np.diff(dipole_array1)
                        dipoles_diff_2 = np.diff(dipole_array2)

                        mean_1 = np.mean(dipoles_diff_1)
                        mean_2 = np.mean(dipoles_diff_2)

                        std_1 = np.std(dipoles_diff_1)
                        std_2 = np.std(dipoles_diff_2)
                        discontinuity_threshold_pos =  mean_1 + (std_1*discontinuity_threshold_std)
                        discontinuity_threshold_neg =  mean_1 - (std_1*discontinuity_threshold_std)
                        idx_1 = np.sort(np.concatenate([ np.where( dipoles_diff_1 >  discontinuity_threshold_pos )[0]+1 ,  np.where( dipoles_diff_1 <  discontinuity_threshold_neg )[0]+1 ]))

                        discontinuity_threshold_pos =  mean_2 + (std_2*discontinuity_threshold_std)
                        discontinuity_threshold_neg =  mean_2 - (std_2*discontinuity_threshold_std)
                        idx_2 = np.sort(np.concatenate([ np.where( dipoles_diff_2 >  discontinuity_threshold_pos )[0]+1 ,  np.where( dipoles_diff_2 <  discontinuity_threshold_neg )[0]+1 ]))



                        peaks_1 = find_peaks( np.abs(np.diff(dipoles_diff_1)) , prominence= np.mean(np.abs(np.diff(dipoles_diff_1))) )
                        peaks_2 = find_peaks( np.abs(np.diff(dipoles_diff_2)) , prominence= np.mean(np.abs(np.diff(dipoles_diff_1))) )


                        
                        peaks_1 = find_peaks( np.abs(dipoles_diff_1) , prominence= np.mean(np.abs(dipoles_diff_1)) )
                        peaks_2 = find_peaks( np.abs(dipoles_diff_2) , prominence= np.mean(np.abs(dipoles_diff_1)) )

                        # plt.plot(peaks_1)
                        # plt.plot(peaks_2)
                        # plt.show()



                        idx_1 = peaks_1[0]+1
                        idx_2 = peaks_2[0]+1

                        # print(idx_1)
                        # print(idx_2)


                        if (len(idx_1)!= 0 and len(idx_2) != 0 ):
                            mask_idx1_idx2 = np.isin(idx_1, idx_2)
                            indices_idx1_in_idx2 = np.where(mask_idx1_idx2)[0]
                            indices_idx1_in_idx2 = idx_1[indices_idx1_in_idx2]

                            
                            if(len(indices_idx1_in_idx2) != 0 ):
                                mask_discontinuties_energydiff = np.isin(indices_idx1_in_idx2, closest_indices)
                                indices_discontinuties_in_energydiff = np.where(mask_discontinuties_energydiff)[0]

                                if len(indices_discontinuties_in_energydiff) != 0 :

                                    idx = indices_idx1_in_idx2[indices_discontinuties_in_energydiff[0]]+ previous_intersection
                                    #print(idx)
                                    array1_copy = np.array(array1, copy=True)
                                    array1 = np.concatenate([array1[:idx],  array2[idx:]])
                                    array2 =np.concatenate([array2[:idx] , array1_copy[idx:]])


                                    dipole_array1 = dipoles[:, i] 
                                    dipole_array2 = dipoles[:, j]


                                    dipole_array1_copy = np.array(dipole_array1, copy=True)
                                    dipole_array1 = np.concatenate([dipole_array1[:idx],  dipole_array2[idx:]])
                                    dipole_array2 =np.concatenate([dipole_array2[:idx] , dipole_array1_copy[idx:]])

                                    dipoles[:,i] = dipole_array1
                                    dipoles[:,j] = dipole_array2
                                    E_array[:,i] = array1
                                    E_array[:,j] = array2
                                    previous_intersection = idx+1
                    except():
                        print("uh oh")
            new_E_array[:,i ] = E_array[:,i]
            new_dipoles[:,i ] = dipoles[:, i]


            # plt.plot(new_dipoles)
            # plt.show()

        E_array = np.copy(new_E_array)
        new_E_array = np.zeros_like(E_array)


    for i in range(0,num_to_declutter):
        previous_intersection = 0

        for z in range(0,num_energy_iterations):
            for j in range(i+1, E_array.shape[1]):
                array1 = E_array[:, i]
                array2 = E_array[:, j]
                #Only want array from previous intersection so it doesnt get recrossed
                array1_from_previous_intersection = array1[previous_intersection:]
                array2_from_previous_intersection = array2[previous_intersection:]
                #trying to determine how close two energy surfaces get, if they get very close this some evidence that they crossover
                diff_array1 = np.diff(array1_from_previous_intersection)
                diff_array2 = np.diff(array2_from_previous_intersection)
                std1 = np.std(np.abs(diff_array1))
                mean1 = np.mean(np.abs(diff_array1))
                std2 = np.std(np.abs(diff_array2))
                mean2 = np.mean(np.abs(diff_array2))


                energy_diff_threshold = ((mean1 + mean2)/2) + (((std1+std2)/2) * energy_diff_threshold_std)
                #find closest points
                #print(np.abs(np.abs(array1[previous_intersection:]) - np.abs(array2[previous_intersection:])))

                # plt.plot(array1[previous_intersection:]- array2[previous_intersection:])
                # plt.show()
                print(energy_diff_threshold)

                closest_indices = np.where(np.abs(array1[previous_intersection:]- array2[previous_intersection:]) < energy_diff_threshold)


                #closest_indices =np.where(np.abs(np.abs(array1[previous_intersection:]) - np.abs(array2[previous_intersection:])) < energy_diff_threshold)
                print("closes indices: " , closest_indices)
                
                try:
                    #print(array1_from_previous_intersection)
                    #use discontinuties in second derivative, discontinutities defined using standard deviation
                    dy_1 = np.abs(np.gradient(np.gradient(array1_from_previous_intersection, r_data[previous_intersection:], edge_order = 1), r_data[previous_intersection:], edge_order = 1))
                    std = np.std(abs(dy_1))
                    mean = np.mean(abs(dy_1))
                    discontinuity_threshold =  mean + (std*discontinuity_threshold_std)
                    idx_1 = np.where(abs(dy_1) >  discontinuity_threshold)[0]+2
                    dy_2= np.abs(np.gradient(np.gradient(array2_from_previous_intersection, r_data[previous_intersection:], edge_order=1), r_data[previous_intersection:], edge_order=1))
                    std = np.std(abs(dy_2))
                    mean = np.mean(abs(dy_2))
                    discontinuity_threshold =  mean + (std*discontinuity_threshold_std)
                    idx_2 = np.where(abs(dy_2) > discontinuity_threshold)[0]+2

                    # plt.plot(dy_1)
                    # plt.plot(dy_2)
                    # plt.show()

                    print(idx_1)
                    print(idx_2)
                    print("ayo")



                    #if energies are close and there are two matcjhing discontinuities in dipole arrary 
                    # diff_1 = np.diff(array1_from_previous_intersection)
                    # diff_2 = np.diff(array2_from_previous_intersection)

                    # mean_1 = np.mean(diff_1)
                    # mean_2 = np.mean(diff_2)

                    # std_1 = np.std(diff_1)
                    # std_2 = np.std(diff_2)
                    # discontinuity_threshold_pos =  mean_1 + (std_1*discontinuity_threshold_std)
                    # discontinuity_threshold_neg =  mean_1 - (std_1*discontinuity_threshold_std)
                    # idx_1 = np.sort(np.concatenate([ np.where( diff_1 >  discontinuity_threshold_pos )[0]+1 ,  np.where( diff_1 <  discontinuity_threshold_neg )[0]+1 ]))

                    # discontinuity_threshold_pos =  mean_2 + (std_2*discontinuity_threshold_std)
                    # discontinuity_threshold_neg =  mean_2 - (std_2*discontinuity_threshold_std)
                    # idx_2 = np.sort(np.concatenate([ np.where( diff_2 >  discontinuity_threshold_pos )[0]+1 ,  np.where( diff_2 <  discontinuity_threshold_neg )[0]+1 ]))



                    # peaks_1 = find_peaks( np.abs(np.diff(diff_1)) , prominence= np.mean(np.abs(np.diff(diff_1))) )
                    # peaks_2 = find_peaks( np.abs(np.diff(diff_2)) , prominence= np.mean(np.abs(np.diff(diff_2))) )
                    # idx_1 = peaks_1[0]+1
                    # idx_2 = peaks_2[0]+1



                    if (len(idx_1)!= 0 and len(idx_2) != 0 ):
                        mask_idx1_idx2 = np.isin(idx_1, idx_2)
                        indices_idx1_in_idx2 = np.where(mask_idx1_idx2)[0]
                        indices_idx1_in_idx2 = idx_1[indices_idx1_in_idx2]
                        # indices_idx1_in_idx2  = indices_idx1_in_idx2[ending_index:]
                        # starting_index=ending_index
                        # ending_index = starting_index
                        starting_index = 0
                        ending_index = 0
                        for elem_index in range(len(indices_idx1_in_idx2)-1):
                            #print("ayo: ", abs((indices_idx1_in_idx2[elem_index]) - (indices_idx1_in_idx2[elem_index+1])))
                            if abs((indices_idx1_in_idx2[elem_index]) - (indices_idx1_in_idx2[elem_index+1])) < 25 :
                                ending_index = ending_index+1
                            else:
                                break
                        indices_idx1_in_idx2 = indices_idx1_in_idx2[starting_index:ending_index]
                        print(indices_idx1_in_idx2)
                        if(len(indices_idx1_in_idx2) != 0 ):
                            mask_discontinuties_energydiff = np.isin(indices_idx1_in_idx2, closest_indices)
                            indices_discontinuties_in_energydiff = np.where(mask_discontinuties_energydiff)[0]
                            #print(indices_discontinuties_in_energydiff)
                            #print(indices_discontinuties_in_energydiff)
                            if len(indices_discontinuties_in_energydiff) != 0 :
                                for k in range(len(indices_discontinuties_in_energydiff) - 1):
                                    idx = indices_idx1_in_idx2[indices_discontinuties_in_energydiff[k]]+ previous_intersection
                                    #print(idx)
                                    array1_copy = np.array(array1, copy=True)
                                    array1 = np.concatenate([array1[:idx],  array2[idx:idx+1], array1[idx+1:]])
                                    array2 = np.concatenate([array2[:idx] , array1_copy[idx:idx+1], array2[idx+1:]])
                                    E_array[:,i] = array1
                                    E_array[:,j] = array2
                                idx = indices_idx1_in_idx2[indices_discontinuties_in_energydiff[-1]]+ previous_intersection
                                #print(idx)
                                array1_copy = np.array(array1, copy=True)
                                array1 = np.concatenate([array1[:idx],  array2[idx:]])
                                array2 =np.concatenate([array2[:idx] , array1_copy[idx:]])
                                #print(indices_idx1_in_idx2)
                                fitting_distance=10
                                if abs(indices_idx1_in_idx2[-1] - indices_idx1_in_idx2[0]) < fitting_distance:
                                    array1 = array1.tolist()
                                    array2 = array2.tolist()
                                    r_data_list = r_data.tolist()
                                    #fitting region
                                    end_discontinuity = indices_idx1_in_idx2[indices_discontinuties_in_energydiff[-1]]+ previous_intersection
                                    start_discontinuity = indices_idx1_in_idx2[indices_discontinuties_in_energydiff[0]]+ previous_intersection
                                    fit_E_data_end = min(end_discontinuity+fitting_distance, len(array1))
                                    fit_E_data_start= max(start_discontinuity-fitting_distance, 0)
                                    # print(fit_E_data_start)
                                    # print(fit_E_data_end)
                                    fitting_E_data = array1[fit_E_data_start: start_discontinuity] + array1[end_discontinuity: fit_E_data_end]
                                    fitting_r_data = r_data_list[fit_E_data_start: start_discontinuity] + r_data_list[end_discontinuity:fit_E_data_end]
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        poly = np.poly1d(np.polyfit(fitting_r_data, fitting_E_data, 12))
                                    r_data_fitting_list =  r_data_list[fit_E_data_start:fit_E_data_end]
                                    polyvals = np.polyval(np.asarray(poly),r_data_fitting_list )
                                    array1 = array1[:fit_E_data_start] + polyvals.tolist() + array1[fit_E_data_end:]
                                    fitting_E_data = array2[fit_E_data_start: start_discontinuity] + array2[end_discontinuity: fit_E_data_end]
                                    fitting_r_data = r_data_list[fit_E_data_start: start_discontinuity] + r_data_list[end_discontinuity:fit_E_data_end]
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        poly = np.poly1d(np.polyfit(fitting_r_data, fitting_E_data, 12))
                                    r_data_fitting_list =  r_data_list[fit_E_data_start:fit_E_data_end]
                                    polyvals = np.polyval(np.asarray(poly),r_data_fitting_list )
                                    array2 = array2[:fit_E_data_start] + polyvals.tolist() + array2[fit_E_data_end:]
                                E_array[:,i] = array1
                                E_array[:,j] = array2
                                previous_intersection = idx
                except():
                    print("uh oh")
        new_E_array[:,i ] = E_array[:,i]
    



    return new_E_array




import warnings
from scipy.signal import find_peaks
import random


def declutter_E_array_TDM(E_array, dipoles,  r_data, discontinuity_threshold_std = 2, energy_diff_threshold_std = 2,  num_to_declutter = 2, upper_triangular_dipoles = True, plot_d_matrices = False):

    print("hi")
    if upper_triangular_dipoles:
        for r_ in range(0, dipoles.shape[3]):

            for i in range(0, dipoles.shape[0]):
                for j in range(i, dipoles.shape[0]):

                    dipoles[j,i, :, r_] =  dipoles[i,j, :, r_]


    
    E_array_copy = np.array(E_array, copy = True)


    for crossover_nums in range(0, 250):

        d_matrices = np.zeros((num_to_declutter , num_to_declutter, r_data.shape[0]) )
        diff_d_matrices =  np.zeros((num_to_declutter , num_to_declutter, r_data.shape[0]-1))

        def build_dipole_mag_matrix(dipoles, n_elec, r_):

            
            d_matrix = np.zeros((n_elec,n_elec))

            def vector_magnitude(vector):
                return np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)


            for i in range(n_elec):
                for j in range(n_elec):
                    d_matrix[i][j] = vector_magnitude(dipoles[i,j,:,r_])
                    #d_matrix[i][j] = dipoles[i,j,2,r_]


            #d_matrix = d_matrix + d_matrix.T - np.diag(np.diag(d_matrix))

            return d_matrix
        
        
        
        for i in range(0, r_data.shape[0] ):
            d_matrices[:, : , i] = build_dipole_mag_matrix(dipoles, num_to_declutter, i)

        d_matrices[d_matrices < 10**-6] = 0



        #plotting d matrices
        if plot_d_matrices and crossover_nums == 0 :
            for i in range(0, dipoles.shape[0]):
                for j in range(i, dipoles.shape[0]):
                    plt.plot(d_matrices[i,j,:])

            plt.show()


        diff_d_matrices = np.diff(d_matrices)
        diff_d_matrices[diff_d_matrices < 10**-11] = 0



        #store discontinyutity locs as a list of lists like  
        # [   [loc, surface_1]  ,     [loc, surface_1] , [loc, surface_1] ]
        #find all discontinutiniutiy locations based on continutity of transition dipoles and dipoles
        discontinuity_locs = []


        previous_intersection = 0 
        for i in range(0,num_to_declutter): 

            for j in range(i, num_to_declutter):
                dipoles_diff = diff_d_matrices[i,j, previous_intersection:]

                peaks_1 = find_peaks( np.abs(np.diff(dipoles_diff)) , prominence= np.mean(np.abs(np.diff(dipoles_diff))) * discontinuity_threshold_std)
                idx_1= peaks_1[0]+2

                peaks_1 = find_peaks( np.abs(dipoles_diff) , height= np.mean(np.abs(dipoles_diff)) * discontinuity_threshold_std)
                idx_1= peaks_1[0]+1
                

                #plt.plot(diff_d_matrices[i, j, previous_intersection:])

                if len(idx_1) != 0:
                    #getting first point of crossover
                    if len(idx_1) > 1:
                        idx_1_copy = idx_1.copy()
                        while(len(idx_1_copy) > 0 ):
                            #print(idx_1_copy)
                            crossover_loc = None
                            for q in range(len(idx_1_copy) - 1):
                                #print(idx_1_copy)
                                if idx_1_copy[q] +1 == idx_1_copy[q+1]:
                                    idx_1_copy = idx_1_copy[q+1:]
                                    pass
                                else:
                                    crossover_loc = idx_1_copy[q]

                                    discontinuity_locs.append([crossover_loc, j])
                                    try:
                                        idx_1_copy = idx_1_copy[q+1:]
                                    except():
                                        idx_1_copy = []
                                        break

                                    break

                                if len(idx_1_copy == 1):
                                    break

                            if len(idx_1_copy) == 1:    
                                idx_1_copy = []
                                discontinuity_locs.append([idx_1[-1], j])
                                break                
                        
                    elif len(idx_1) == 1:
                        discontinuity_locs.append([idx_1[0], j])


        #sort and remove repaets
        discontinuity_locs  = [list(x) for x in set(tuple(x) for x in discontinuity_locs )]
        discontinuity_locs = sorted(discontinuity_locs, key=lambda x: x[0])
        #print(discontinuity_locs)

        #store discontinity locations like this
        # [   [loc, surface_1, surface_2]  ,     [loc, surface_1, surface_2] , [loc, surface_1, surface_2] ]
        #find all discontinutiniutiy locations based on continutity of transition dipoles and dipoles

        new_discontinuity_locs = []
        for q in range(len(discontinuity_locs)):
            loc1 = discontinuity_locs[q]
            for w in range(len(discontinuity_locs)):
                if w != q:
                    loc2 = discontinuity_locs[w]

                    if loc1[1] != loc2[1]:

                        if loc1[0] == loc2[0]:
                            new_discontinuity_locs.append([loc1[0] ,discontinuity_locs[q][1], discontinuity_locs[w][1]])

                        if loc1[0] == loc2[0] -1:
                            new_discontinuity_locs.append([loc2[0] ,discontinuity_locs[q][1], discontinuity_locs[w][1]])

                        if loc1[0] -1 == loc2[0]:
                            new_discontinuity_locs.append([loc1[0] ,discontinuity_locs[q][1], discontinuity_locs[w][1]])

        #sort and remove repeats again
        discontinuity_locs = new_discontinuity_locs
        # we need to remove one of these [loc, 1,2] and [loc, 2,1]
        for i in range(len(discontinuity_locs)):
            if discontinuity_locs[i][1] > discontinuity_locs[i][2]:

                copy = discontinuity_locs[i][2] 
                discontinuity_locs[i][2] = discontinuity_locs[i][1] 
                discontinuity_locs[i][1] = copy

        discontinuity_locs  = [list(x) for x in set(tuple(x) for x in discontinuity_locs )]
        discontinuity_locs = sorted(discontinuity_locs, key=lambda x: x[0])


        #for every discontinuity loc check and see if energies are very close using standard deviations of differences
        #crossover energy arrays

        crossover_points = []
        for i in range(len(discontinuity_locs)):
            loc = discontinuity_locs[i]

            array1 = E_array[:, loc[1]]
            array2 = E_array[:, loc[2]]


            #trying to determine how close two energy surfaces get, if they get very close this some evidence that they crossover
            diff_array1 = np.diff(array1 )
            diff_array2 = np.diff(array2 )
            std1 = np.std(np.abs(diff_array1))
            mean1 = np.mean(np.abs(diff_array1))
            std2 = np.std(np.abs(diff_array2))
            mean2 = np.mean(np.abs(diff_array2))
            energy_diff_threshold = ((mean1 + mean2)/2) + (((std1+std2)/2) * energy_diff_threshold_std)
            #find closest points
            #print(np.abs(np.abs(array1[previous_intersection:]) - np.abs(array2[previous_intersection:])))
            closest_indices =np.where(np.abs(np.abs(array1[previous_intersection:]) - np.abs(array2[previous_intersection:])) < energy_diff_threshold)[0]
            closest_indices = closest_indices +1 
            if loc[0] in closest_indices:

                crossover_points.append(loc)


        # a few points are like this :[1084, 3, 4] ,[1085, 3, 4], choose the larger one
        indices_to_pop = []
        for i in range(len(crossover_points)-1):
            if crossover_points[i][1] == crossover_points[i+1][1] and crossover_points[i][2] == crossover_points[i+1][2]:
                if crossover_points[i][0] +1 == crossover_points[i+1][0]:
                    indices_to_pop.append(i)
        indices_to_pop.sort()
        indices_to_pop.reverse()
        for i in indices_to_pop:
            crossover_points.pop(i)

        #only get first crossover in list
        #get a random one from crossover list and put it in
        print(crossover_points)
        if len(crossover_points) >0 :
            random_int = random.randint(0, len(crossover_points)-1)
            crossover_points = [crossover_points[random_int]]
        else:
            print("jhiii")
            #plotting d matrices
            if plot_d_matrices :
                print(dipoles.shape)
                print(d_matrices.shape)
                
                for i in range(0, dipoles.shape[0]):
                    for j in range(i, dipoles.shape[0]):
                        plt.plot(d_matrices[i,j,:])

                plt.show()


            print("done")
            break


        #perform crossover:
        for i in range(len(crossover_points)):

            #if using first derivative
            idx = crossover_points[i][0]

            #if using second derivatve
            # idx = crossover_points[i][0] - 1

            array1 = E_array[:, crossover_points[i][1]]
            array2 = E_array[:, crossover_points[i][2]]

            array1_copy = np.array(array1, copy=True)
            array1 = np.concatenate([array1[:idx],  array2[idx:]])
            array2 =np.concatenate([array2[:idx] , array1_copy[idx:]])

            E_array[:,crossover_points[i][1]] = array1
            E_array[:,crossover_points[i][2]] = array2


            #go in and change crossover points to reflect changes in E_array
            surface_1 = crossover_points[i][1]
            surface_2 = crossover_points[i][2] 


            for p in range(len(crossover_points)):

                if crossover_points[p][1] == surface_1:
                    crossover_points[p][1] = surface_2

                elif crossover_points[p][1] == surface_2:
                    crossover_points[p][1] = surface_1

                if crossover_points[p][2] == surface_1:
                    crossover_points[p][2] = surface_2

                elif crossover_points[p][2] == surface_2:
                    crossover_points[p][2] = surface_1


        #reorder dipole array based off that crossover to repeat
        crossover_point = crossover_points[0][0] 
        surface_1 = crossover_points[0][1]  
        surface_2 = crossover_points[0][2]      



        for r_ in range(crossover_point, dipoles.shape[3]):
            for switch_num in range(dipoles.shape[0]):

                holder = np.copy(dipoles[switch_num, surface_1, :, r_])
                dipoles[switch_num, surface_1, :, r_] = dipoles[switch_num, surface_2, :, r_]
                dipoles[switch_num, surface_2, :, r_] = holder


                holder = np.copy(dipoles[surface_1, switch_num, :, r_])
                dipoles[surface_1, switch_num, :, r_] = dipoles[surface_2, switch_num, :, r_]
                dipoles[surface_2, switch_num, :, r_] = holder

    if plot_d_matrices :
                print(dipoles.shape)
                print(d_matrices.shape)
                
                for i in range(0, dipoles.shape[0]):
                    for j in range(i, dipoles.shape[0]):
                        plt.plot(d_matrices[i,j,:])

                plt.show()


    return E_array, dipoles



hbar = 1

# number of grid points 
#N = 3001


def get_fd_wfn(x, V_y: np.array, use_5_point_stencil = False, N = 100, mu_au = 1):

    hbar = 1


    # define grid spacing h
    h = x[1]-x[0]

    # create arrays for T, V, and H - we truncate the smallest and largest grid points where 
    # the centered finite difference derivatives cannot be defined
    

    # this uses the 3 point stencil; we can adapt to use a 5 point and it might improve accuracy
    ### JJF Comment - initializing all arrays in conditionals since they have different sizes 
    ### JJF Comment - depending on the stencil used 
    if not use_5_point_stencil:

        T = np.zeros((N-2, N-2)) 
        V = np.zeros((N-2, N-2))
        H = np.zeros((N-2, N-2))
        for i in range(N-2):
            for j in range(N-2):
                if i==j:
                    T[i,j]= -2
                    # potential is offset by 1 for 3 point stencil 
                    V[i,j]= V_y[i+1]

                elif np.abs(i-j)==1:
                    T[i,j]=1
                    V[i,j]=0
                else:
                    T[i,j]=0
                    V[i,j]=0

        # finish building H 
        H = -T *( hbar ** 2 / (2 * mu_au* h**2)) + V
        


    elif use_5_point_stencil:
        T = np.zeros((N-4, N-4))
        V = np.zeros((N-4, N-4))
        H = np.zeros((N-4, N-4))
        for i in range(N-4):
            for j in range(N-4):
                if i==j:
                    T[i,j]= -30
                    # potential is offset by 2 for 5 point stencil
                    V[i,j]= V_y[i+2]
                elif np.abs(i-j)==1:
                    T[i,j]=16
                    V[i,j]=0
                elif np.abs(i-j)==2:
                    T[i,j]=-1
                    V[i,j]=0

        # finish building H
        H = -T *  ((hbar ** 2) / (2* mu_au))*  (1 / ( 12 * h**2)) + V


    vals, vecs = np.linalg.eigh(H)

    if np.average(vecs[:, 0]) < 0:
        vecs = vecs * -1

    return vals, vecs




import matplotlib.pyplot as plt



def get_transition_frequencies(potential_1, r_data , N = 100, plot = False):
    bohr2angstroms = 0.529177249
    print("rdata shape , ", r_data.shape)
    r_data_au = r_data / bohr2angstroms

    print(r_data_au.shape)

    min_potential_1_loc = np.argmin(potential_1[:])
    r_eq_au =r_data_au[potential_1.argmin()]

    print("r_eq_au : " , r_eq_au)


    # Fitting S0 PES to a quintic polynomial

    poly = np.poly1d(np.polyfit(r_data_au, potential_1, 19))

    poly_array = np.asarray(poly)


    #Taking first and second derivative of S0 PES and evaluating at r_eq
    first_derivative = poly.deriv()
    second_derivative = first_derivative.deriv()
    k_au = second_derivative(r_eq_au)
    print("k_au: ", k_au)


    angstrom_to_bohr = 1.88973
    x_min = r_data_au[0]
    x_max = r_data_au[-1]

    hbar = 1

    # define grid
    x = np.linspace(x_min, x_max, N)

    V_y = np.polyval(np.asarray(poly), (x))


    # # number of grid points 
    # N = r_data.shape[0]
    # # define grid
    # x = np.linspace(x_min, x_max, N)

    # V_y = np.polyval(np.asarray(poly), (x))


    vals1, vecs1 = get_fd_wfn(r_data_au, potential_1, use_5_point_stencil=True, N = N)
    #vals1, vecs1 = get_fd_wfn(x, V_y)


    if plot:
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('bond length (m)')
        ax1.set_ylabel('wfn', color=color)
        ax1.plot(x[1:N-1], vecs1[:,0], 'r', label = "$\psi_0$")
        ax1.plot(x[1:N-1], vecs1[:,1], 'b',label = "$\psi_1$" )
        ax1.plot(x[1:N-1], vecs1[:,2], 'g',label = "$\psi_2$")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('energy (hartree)', color=color)  # we already handled the x-label with ax1
        ax2.plot(r_data_au, potential_1, 'bo', label='PES_1')
        ax2.plot(r_data_au, poly(r_data_au), 'm-', label='fit')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.legend()
        plt.show()




    # fig, ax1 = plt.subplots()

    # color = 'tab:red'
    # ax1.set_xlabel('bond length (m)')
    # ax1.set_ylabel('wfn', color=color)
    # ax1.plot(x[1:N-1], vecs1[:,0], 'r', label = "$\psi_0$")
    # ax1.plot(x[1:N-1], vecs1[:,0]**2, 'r', label = "$\|psi_0|**2$")

    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('energy (hartree)', color=color)  # we already handled the x-label with ax1
    # ax2.plot(r_data_au, potential_1, 'bo', label='PES_1')
    # ax2.plot(r_data_au, poly(r_data_au), 'm-', label='fit')
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # fig.legend()
    # plt.show()

    # fig, ax1 = plt.subplots()

    # color = 'tab:red'
    # ax1.set_xlabel('bond length (m)')
    # ax1.set_ylabel('wfn', color=color)
    # ax1.plot(x[1:N-1], vecs1[:,4], 'r', label = "$\psi_1$")
    # ax1.plot(x[1:N-1], vecs1[:,4]**2, 'r', label = "$\|psi_1|**2$")

    # ax1.tick_params(axis='y', labelcolor=color)

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    # ax2.set_ylabel('energy (hartree)', color=color)  # we already handled the x-label with ax1
    # ax2.plot(r_data_au, potential_1, 'bo', label='PES_1')
    # ax2.plot(r_data_au, poly(r_data_au), 'm-', label='fit')
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # fig.legend()
    # plt.show()


    return vals1, vecs1



def get_fcf_matrix(potential_1, potential_2,  r_data , N = 100, num_matrix_elements = 6, mu_au = 1,  plot = False,):
    bohr2angstroms = 0.529177249

    #print("rdata shape , ", r_data.shape)
    r_data_au = r_data / bohr2angstroms

    #print(r_data_au.shape)

    min_potential_1_loc = np.argmin(potential_1[:])
    r_eq_au =r_data_au[potential_1.argmin()]

    #print("r_eq_au : " , r_eq_au)


    # Fitting S0 PES to a quintic polynomial

    poly = np.poly1d(np.polyfit(r_data_au, potential_1, 12))

    poly_array = np.asarray(poly)


    #Taking first and second derivative of S0 PES and evaluating at r_eq
    first_derivative = poly.deriv()
    second_derivative = first_derivative.deriv()
    k_au = second_derivative(r_eq_au)
    #print("k_au: ", k_au)


    angstrom_to_bohr = 1.88973
    x_min = r_data_au[0]
    x_max = r_data_au[-1]

    hbar = 1

    # define grid
    x = np.linspace(x_min, x_max, N)

    V_y = np.polyval(np.asarray(poly), (x))


    # # number of grid points 
    # N = r_data.shape[0]
    # # define grid
    # x = np.linspace(x_min, x_max, N)

    # V_y = np.polyval(np.asarray(poly), (x))


    vals1, vecs1 = get_fd_wfn(r_data_au, potential_1, use_5_point_stencil=True, N = N,mu_au=mu_au)
    print("IN GET_FCF_MATRIX")
    print("Getting fundamental Frequency")
    print(F"Fundamental frequency: {vals1[1] - vals1[0]}")
    #vals1, vecs1 = get_fd_wfn(x, V_y)


    if plot:
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('bond length (m)')
        ax1.set_ylabel('wfn', color=color)
        ax1.plot(x[1:N-1], vecs1[:,0], 'r', label = "$\psi_0$")
        ax1.plot(x[1:N-1], vecs1[:,1], 'b',label = "$\psi_1$" )
        ax1.plot(x[1:N-1], vecs1[:,2], 'g',label = "$\psi_2$")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('energy (hartree)', color=color)  # we already handled the x-label with ax1
        ax2.plot(r_data_au, potential_1, 'bo', label='PES_1')
        ax2.plot(r_data_au, poly(r_data_au), 'm-', label='fit')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.legend()
        plt.show()


    #print("rdata shape , ", r_data.shape)
    r_data_au = r_data / bohr2angstroms

    #print(r_data_au.shape)

    min_potential_2_loc = np.argmin(potential_2[:])
    r_eq_au =r_data_au[potential_1.argmin()]

    #print("r_eq_au : " , r_eq_au)


    # Fitting S0 PES to a quintic polynomial

    poly = np.poly1d(np.polyfit(r_data_au, potential_2, 12))

    poly_array = np.asarray(poly)


    #Taking first and second derivative of S0 PES and evaluating at r_eq
    first_derivative = poly.deriv()
    second_derivative = first_derivative.deriv()
    k_au = second_derivative(r_eq_au)
    #print("k_au: ", k_au)


    angstrom_to_bohr = 1.88973
    x_min = r_data_au[0]
    x_max = r_data_au[-1]

    hbar = 1

    # define grid
    x = np.linspace(x_min, x_max, N)

    V_y = np.polyval(np.asarray(poly), (x))


    # # number of grid points 
    # N = r_data.shape[0]
    # # define grid
    # x = np.linspace(x_min, x_max, N)

    # V_y = np.polyval(np.asarray(poly), (x))


    vals2, vecs2 = get_fd_wfn(r_data_au, potential_2, use_5_point_stencil=True, N = N,mu_au=mu_au)
    #vals1, vecs1 = get_fd_wfn(x, V_y)


    if plot:
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('bond length (m)')
        ax1.set_ylabel('wfn', color=color)
        ax1.plot(x[1:N-1], vecs2[:,0], 'r', label = "$\psi_0$")
        ax1.plot(x[1:N-1], vecs2[:,1], 'b',label = "$\psi_1$" )
        ax1.plot(x[1:N-1], vecs2[:,2], 'g',label = "$\psi_2$")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('energy (hartree)', color=color)  # we already handled the x-label with ax1
        ax2.plot(r_data_au, potential_2, 'bo', label='PES_1')
        ax2.plot(r_data_au, poly(r_data_au), 'm-', label='fit')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.legend()
        plt.show()

    FCF_matrix = np.zeros((num_matrix_elements,num_matrix_elements))

    for i in range(FCF_matrix.shape[0]):
        for j in range(FCF_matrix.shape[0]):

            FCF_matrix[i][j] = np.trapz(vecs1[:,i] * vecs2[:,j]) 
            FCF = np.absolute(FCF_matrix) ** 2 

    return FCF


from scipy.interpolate import make_interp_spline


def calculate_HR(potential_1, potential_2, r_data_ang, mu_au ):
    bohr2angstroms = 0.529177249
    #takes r_data in angstroms
    r_data = r_data_ang/bohr2angstroms


    potential_1_spline = make_interp_spline(r_data, potential_1, k=7)


    potential_2_spline = make_interp_spline(r_data, potential_2, k= 7)


    vals1, vecs1 =  get_fd_wfn(r_data, potential_1, use_5_point_stencil=True, N = r_data.shape[0],mu_au=mu_au)


    r_data_extended = np.linspace(   r_data[0],r_data[-1], 8000 )
    plot_spline_1 = potential_1_spline(r_data_extended)
    plot_spline_2 = potential_2_spline(r_data_extended)


    r_eq_1 =r_data_extended[plot_spline_1.argmin()]
    r_eq_2 =r_data_extended[plot_spline_2.argmin()]

    # print(r_eq_1)
    # print(r_eq_2)

    delta_x = np.abs(r_eq_1 - r_eq_2)

    classical_turning_point_energy = vals1[0]

    
    # plt.axhline(classical_turning_point_energy)
    # plt.plot(r_data_extended, plot_spline_1)
    # plt.plot(r_data_extended, plot_spline_2)
    # plt.show()

    min_index_spline1 = plot_spline_1.argmin()
    sliced_at_min_spline_1 = plot_spline_1[: min_index_spline1]
    # plt.axhline(classical_turning_point_energy)
    # plt.plot(r_data_extended[: min_index_spline1], sliced_at_min_spline_1)
    # plt.plot(r_data_extended, plot_spline_2)
    # plt.show()

    # Compute pairwise differences and their absolute values
    distances = np.abs(sliced_at_min_spline_1 [:, None] - np.array([classical_turning_point_energy]))

    # Find the indices of the minimum distance
    min_index = np.unravel_index(np.argmin(distances), distances.shape)

    # Closest points
    # closest_point_array1 = sliced_at_min_spline_1 [min_index[0]]
    # closest_point_array2 = np.array([classical_turning_point_energy])[min_index[1]]
    # min_distance = distances[min_index]
    #print(min_index[0])
    #print(plot_spline_1[min_index[0]])

    x0 = r_data_extended[min_index[0]]

    x0 = np.abs(r_eq_1 - x0)

    #calculates S using harmonic approximation slightly sifferently
    S = ((delta_x) / (x0))**2 / 2
    #print("S first way " , S)

    #print("omega fd  " ,vals1[1] - vals1[0])

    ##calculates S using harmonic approximation also
    spline_second_derivative = potential_1_spline.derivative(2)
    # Evaluate the second derivative at a specific point
    second_derivative_min = spline_second_derivative(r_eq_1)
    omega = np.sqrt(second_derivative_min / mu_au)
    #print("omega harmonic  " , omega)
    S= (mu_au * (delta_x)**2 * (omega))    /  ((  2))
    #print("S third way " , S)


    #calcules S not using harmonic approximation  ## mu * deltax**2 * omega_vib  / 2
    S= (mu_au * (delta_x)**2 * (vals1[1] - vals1[0]) )    /  ((  2))
    print("S second way " , S)
    return S



import numpy as np
import matplotlib.pyplot as plt

class PQED:
    def __init__(self, energies, dipoles):
        self.E_array = energies
        self.dipoles = dipoles
    def create_annihilation_operator(self,N):
        """
        Creates the matrix representation of the annihilation operator (b) for a harmonic oscillator
        in a Hilbert space with N levels.
        Parameters:
        N (int): Number of levels.
        Returns:
        np.ndarray: The matrix representation of the annihilation operator.
        """
        b = np.zeros((N, N))
        for j in range(1, N):
            b[j-1, j] = np.sqrt(j)
        return b
    def create_creation_operator(self,N):
        """
        Creates the matrix representation of the creation operator (b†) for a harmonic oscillator
        in a Hilbert space with N levels.
        Parameters:
        N (int): Number of levels.
        Returns:
        np.ndarray: The matrix representation of the creation operator.
        """
        b_dagger = np.zeros((N, N))
        for j in range(1, N):
            b_dagger[j, j-1] = np.sqrt(j)
        return b_dagger
    def create_number_operator(self,N):
        """
        Creates the matrix representation of the number operator (n) = b† * b.
        Parameters:
        N (int): Number of levels.
        Returns:
        np.ndarray: The matrix representation of the number operator.
        """
        b = self.create_annihilation_operator(N)
        b_dagger = self.create_creation_operator(N)
        # The number operator is n = b† * b
        n = np.dot(b_dagger, b)
        return n
    def build_d_array(
        self,
        n_el,
        lambda_vector,
        mu_array,
        coherent_state=False,
        coherent_state_pos = None,
        coherent_state_val = None
        ):
        """
        method to compute the array d = \lambda \cdot \mu if coherent_state==False
        or d = \lambda \cdot (\mu - <\mu>) if coherent_state == True
        and store to attribute self.d_array
        """
        if coherent_state == False:
            d_array = np.einsum(
                "k,ijk->ij", lambda_vector, mu_array[:n_el, :n_el, :]
            )
        else:
            _I = np.eye(n_el)
            d_array = np.einsum(
                "k,ijk->ij", lambda_vector, mu_array[:n_el, :n_el, :]
            )
            if coherent_state_val == None and coherent_state_pos == None:
                _d_exp = d_array[0, 0]
            elif coherent_state_val == None and coherent_state_pos != None:
                _d_exp = d_array[coherent_state_pos, coherent_state_pos]
            else:
                _d_exp = coherent_state_val
            d_array = d_array - _I * _d_exp
        return d_array
    def compute_energy_corrections(self, order, n_el, n_ph, omega, lambda_vector,  coherent_state = False, coherent_state_val= None, coherent_state_pos = None, ):
        n_max = n_el * n_ph
        n = 0
        # Identity matrices for each subsystem
        I_matter = np.eye(n_el)
        I_photon = np.eye(n_ph)
        # Build dipole moment matrix _d
        d_array = self.build_d_array(
                n_el, lambda_vector, self.dipoles, coherent_state, coherent_state_pos, coherent_state_val
            )
        _d = np.copy(d_array)
        # Create bosonic subspace operators
        b = self.create_annihilation_operator(n_ph)
        b_dagger = self.create_creation_operator(n_ph)
        N = self.create_number_operator(n_ph)
        # Electronic energy contribution (diagonal in matter space, identity in photon space)
        E_matter = np.diag(self.E_array[:n_el])
        E = np.kron(I_photon, E_matter)
        # Photon energy contribution (diagonal in photon space, identity in matter space)
        Cav_photon = omega * N
        Cav = np.kron( Cav_photon, I_matter)
        #Bilinear light-matter coupling term
        BLC_matter = _d
        BLC_photon = (b_dagger + b)
        BLC = - np.sqrt(omega / 2)  *  np.kron( BLC_photon, I_matter) @ np.kron(I_photon, BLC_matter)
        # Dipole -energy term
        DSE_matter = 1 / 2 * _d @ _d
        DSE = np.kron( I_photon, DSE_matter)
        # Total Hamiltonian
        H0 = E + Cav
        V= BLC +DSE
        # Diagonalize the unperturbed Hamiltonian
        E0, psi0 = np.linalg.eigh(H0)
        #psi0[:, 0] is firt wavefunction
        #print(psi0.shape)
        # Initialize corrections to energy and wavefunction
        energy_corrections = np.zeros(( order+1))
        wavefunction_corrections = np.zeros((n_max, order+1))
        # Set unperturbed energies and wavefunctions
        energy_corrections[ 0] = E0[n]
        wavefunction_corrections[ :, 0] = psi0[:, n]
        n_0 = 0
        for k in range(1, order + 1):
            if k == 1:
                #first order energy correction
                energy_corrections[1] = np.dot(psi0[:, n].T, np.dot(V, psi0[:,n]))
                #print(energy_corrections[1])
                coeff_m = 0
                for m in range(0, n_max):
                    coeff_m = 0
                    if m!= n:
                        coeff_m -=   np.dot(psi0[:, m].T, np.dot(V, psi0[:,n]))
                        for l in range(1, k+1):
                            #goes from l to k-1
                            coeff_m += energy_corrections[l] * np.dot(psi0[:, m].T, wavefunction_corrections[:, k-l])
                        coeff_m = coeff_m/np.abs(E0[m] - E0[n])
                        wavefunction_corrections[:, k ] += (coeff_m * psi0[:,m])
            if k!=1:
                for m in range(0, n_max):
                    coeff_m = 0
                    if m!= n:
                        coeff_m -=  np.dot(psi0[:,m].T, np.dot(V, wavefunction_corrections[:, k-1]))
                        for l in range(1, k+1):
                            #goes from l to k-1
                            coeff_m += energy_corrections[l] * np.dot(psi0[:, m].T, wavefunction_corrections[:, k-l])
                        coeff_m = coeff_m/np.abs(E0[m] - E0[n])
                        # print("coeff_ m for state ", m , ": ", coeff_m)
                        wavefunction_corrections[:, k ] += (coeff_m * psi0[:,m])
            if k!= 1:
                energy_correction =  np.dot(psi0[:,n].T, np.dot(V, wavefunction_corrections[:, k-1]))
                for j in range(0, k):
                    if j != k:
                        #print("energy correction for ", j ," : " ,energy_correction)
                        energy_correction -=  energy_corrections[j]  * np.dot(psi0[:,n].T,  wavefunction_corrections[:, k-j])
                energy_corrections[ k] = energy_correction
       #print("energy_corrections: " , energy_corrections)
        return energy_corrections


    def get_dipole_array_in_polariton_basis(self,):


        I_matter = np.eye(self.n_el)
        I_photon = np.eye(self.n_ph)

        dipole_transformed_size = np.kron(I_photon, I_matter).shape[0]
        
        self.dipoles_transformed = np.zeros((dipole_transformed_size, dipole_transformed_size, 3))

        dipoles_truncated = self.dipoles[:self.n_el, :self.n_el, :]


        for i in range(0,3):
            current_dipole = dipoles_truncated[:,:,  i]
            current_dipole = np.kron(I_photon, current_dipole)
            transformed_dipole = self.eigenvectors.T @ current_dipole @ self.eigenvectors

            self.dipoles_transformed[:,:, i] = transformed_dipole





    def PQED_Hamiltonian(self, n_el, n_ph, omega, lambda_vector,  coherent_state = False, coherent_state_val= None, coherent_state_pos = None,  ):
        """
        Build the PF Hamiltonian for a system with n_el electronic states and n_ph photon states.
        """

        self.n_el = n_el
        self.n_ph = n_ph
        # Identity matrices for each subsystem
        I_matter = np.eye(n_el)
        I_photon = np.eye(n_ph)
        # Build dipole moment matrix _d
        d_array = self.build_d_array(
                n_el, lambda_vector, self.dipoles, coherent_state, coherent_state_pos, coherent_state_val
            )
        _d = np.copy(d_array)
        # Create bosonic subspace operators
        b = self.create_annihilation_operator(n_ph)
        b_dagger = self.create_creation_operator(n_ph)
        N = self.create_number_operator(n_ph)
        # Electronic energy contribution (diagonal in matter space, identity in photon space)
        E_matter = np.diag(self.E_array[:n_el])
        E = np.kron(I_photon, E_matter)
        # Photon energy contribution (diagonal in photon space, identity in matter space)
        Cav_photon = omega * N
        Cav = np.kron( Cav_photon, I_matter)
        #Bilinear light-matter coupling term
        BLC_matter = _d
        BLC_photon = (b_dagger + b)
        BLC = - np.sqrt(omega / 2)  *  np.kron( BLC_photon, I_matter) @ np.kron(I_photon, BLC_matter)
        # Dipole -energy term
        DSE_matter = 1 / 2 * _d @ _d
        DSE = np.kron( I_photon, DSE_matter)
        # Total Hamiltonian
        self.H = E + Cav  + BLC + DSE
        self.PCQED_MU = np.kron(I_photon, _d)
        # Return eigenvalues of the Hamiltonian
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H)

        #calculate the transformed dipole operator to the polariton basis
        self.get_dipole_array_in_polariton_basis()

        return self.eigenvalues, self.eigenvectors, self.H
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:13:23 2024

@author: bmillet
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import os


# Reference the region based on the index of the 'fs_3d' field
dyes = ['dyeLL', 'dyeMS',  'dyeNP', 'dyeHS', 'dyeNA', 'dyeAA']
simus = ['tm21ah21', 'tm20ah21', 'tm21ah20', 'tm21ahgr', 'tm22ah21']; simu_labels = ['REF', 'DIALOW', 'ISOCST', 'ISO3D', 'DIAHIGH']

l_patches2 = ['LL', 'MS', 'NP', 'HS', 'NA', 'AA']
panels_letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w']
panels_letter_parenthesis = ['(' + letter + ')' for letter in panels_letter]
earth_rad = 6.371*10**6

# Function that creates a list of 45 depth levels
def create_l_depth():
    l_depth0 = []
    for i in range(6):
        l_depth0.append(i*10)
    for i in range(6):
        l_depth0.append(75+i*25)
    for i in range(3):
        l_depth0.append(250+i*50)
    for i in range(11):
        l_depth0.append(400+i*100)
    for i in range(19):
        l_depth0.append(1500+i*250)
    return (l_depth0)



def approx_depth(d, l_depth):
    ''' Return the closest value in the list to the value given and the index associated to this value. '''
    for i in range(0,len(l_depth)):
        diff = abs(d-l_depth[i])
        if i ==0:
            min = (diff,i)
        elif min[0] > diff:
            min = (diff,i)
    return((l_depth[min[1]],min[1]))

# same function as approx_depth, but allows us to search for a whole table instead of a value
def approx_depth_array(array, list):
    substract_array = np.array(list)
    array_plus1D = np.tile(array, (len(list), 1, 1)).transpose(1, 2, 0)

    # we need to account for the nanvalues, otherwise we will have an error raised
    array_diff = np.where(~np.isnan(array_plus1D), np.abs(array[..., np.newaxis] - substract_array), np.nan)

    # Ici on définit le tableau des indices de notre liste l_depth tq la différence est minimale, toujours en faisant attention aux nan values
    ind_min_array = np.where(~np.isnan(array), np.argmin(array_diff, axis = len(np.shape(array_diff)) - 1), -1)
    # We return the array containing the indices of the minimum
    return(ind_min_array)
    

    
def get_BoundNorm(vmin, vmax, cmapname = 'coolwarm', nbins = 10):
    '''Creates a specific colormap with the number of bins.'''
    cmap = plt.colormaps[cmapname]
    levels = MaxNLocator(nbins = nbins).tick_values(vmin, vmax)
    return(BoundaryNorm(levels, ncolors = cmap.N, clip=True))
    


def get_xylabels(n_lines, n_col, i):
    '''Return (xlabel, ylabel), two bools, which indiactes whether or not there should be a label for the x axis or y axis, based on the index i and the number of lines and cols in the plot.'''
    i_line, i_col = i//n_col, i%n_col
    xlabel, ylabel = False, False
    if i_line == (n_lines - 1):
        xlabel = True
    if i_col == 0:
        ylabel = True
    return(xlabel, ylabel)



def plot_details_axis(ax, pco, cb = True, cbarlabel = '', xlim = (-50, 60), ylim = (6000, 0), font = 15, cmapname = 'viridis', nbins = 10, title=''
                  , xlabels = True, ylabels = True, xticks = [-40+20*i for i in range(6)], yticks = [1000*i for i in range(7)]):
    '''Function to save some space for the figure which does a bunch of things depending on the arguments given.'''
    if cb:
        cbar = plt.colorbar(pco)#, ticks = [i*age_max/4 for i in range(6)])
        cbar.ax.tick_params(labelsize=font-2)
        cbar.set_label(cbarlabel, fontsize = font)
        
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    if xlabels:
        if xticks == [-40+20*i for i in range(6)]:
            ax.set_xlabel('Latitude (°N)', fontsize=font)
        elif xticks == [-80 + i * 30 for i in range(5)]:
            ax.set_xlabel('Latitude (°N)', fontsize=font)
    else: 
        ax.set_xticklabels([])
    if ylabels:
        if yticks == [1000*i for i in range(7)]:
            ax.set_yticklabels([1*i for i in range(7)],fontsize = font)
            ax.set_ylabel('Depth (km)', fontsize=font)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis='both', labelsize = font)
    ax.set_title(title, fontsize = font)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    

def plot_Pac_mean(ax, ds_averaged, field, l_depth, mask, nav_lat, vmin = 0, vmax = 1, cb = True, cbarlabel = '', xlim = (-50, 60), ylim = (6000, 0), font = 15, cmapname = 'viridis', nbins = 10, title=''
                  , xlabels = True, ylabels = True, xticks = [-40+20*i for i in range(6)], yticks = [1000*i for i in range(7)]):
    '''Function that plots a zonal mean of a given field in the Pacific. 
    If we want to plot the Age, we need to specify a value for vmin and vmax to get an appropriate scale.'''
    array_to_plot = np.nanmean(np.where(np.tile(mask, (len(list(l_depth)), 1, 1)), ds_averaged[field], np.nan), axis=2)
    lats_avg = get_lats_avg(nav_lat, mask)

    cmap = plt.colormaps[cmapname]
    levels = MaxNLocator(nbins = nbins).tick_values(vmin, vmax)
    norm = BoundaryNorm(levels, ncolors = cmap.N, clip=True)
    
    pco = ax.pcolormesh(lats_avg, l_depth, array_to_plot[:,:], antialiased = False, norm = norm, cmap = cmapname, linewidth = 0, rasterized = True)
    plot_details_axis(ax, pco, cb = cb, cbarlabel = cbarlabel, xlim = xlim, ylim = ylim, font = font, cmapname = cmapname, nbins = nbins, title = title, xlabels = xlabels, ylabels = ylabels, xticks = xticks, yticks = yticks)
    return (pco, lats_avg)


def add_cbar(fig, pco, x = 0.92, y = 0.1, width = 0.015, height = 0.8, fontsize = 14, label = '', ticks = []):
    cax = fig.add_axes([x, y, width, height])  # [x, y, width, height]
    cb = plt.colorbar(pco, cax=cax)
    cb.ax.tick_params(labelsize=fontsize - 1)
    cb.set_label(label, fontsize=fontsize)
    if ticks != []:
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=fontsize)
    return cb



def plot_dye_depthlvl(ax, ds_averaged, dye, nav_lat, nav_lon, idepth=0, mask = None, xlim=(73.37911, 432.62088), ylim=(-80, 90), fontsize = 15,
                      cb = True, cbarlabel = '', vmin = 0, vmax = 1, cmapname = 'viridis', nbins = 10, extend = True, title = '', xlabels = True, ylabels = True, 
                      xticks = [40*i for i in range(10)], yticks = [-40+20*i for i in range(6)]):
    '''Take a dataset, the name of a field as well as the axis on which to plot, a potential mask and makes the plot.'''
    if not (mask is None):
        array_to_plot = np.where(mask, ds_averaged[dye], np.nan)
    else:
        array_to_plot = ds_averaged[dye]
    cmap = plt.colormaps[cmapname]
    levels = MaxNLocator(nbins = nbins).tick_values(vmin, vmax)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    pco = ax.pcolormesh(nav_lon, nav_lat, array_to_plot[idepth], antialiased = False, norm = norm, cmap = cmapname)
                       
    plot_details_axis(ax, pco, cb = cb, cbarlabel = cbarlabel, xlim = xlim, ylim = ylim, font = fontsize, cmapname = cmapname, nbins = nbins, title = title, xlabels = xlabels, ylabels = ylabels, xticks = xticks, yticks = yticks)
    return(pco)



def non_monotone(l):
    '''Renvoie la première valeur pour laquelle la monotonie (ici croissante) n'est plus vrai.
    To apply with the nav_lon table.'''
    l = [x for x in l if not np.isnan(x)]
    if len(l) == 0:
        return((-1, None))
        
    aux = l[0]
    for i in range(len(l)):
        if l[i] < aux:
            return (l[i], i)
        aux = l[i]
            
    return((-1, None))



def transfo(nav_lat, nav_lon):
    '''Function that transform both nav_lat and nav_lon in arrays that are useable to plot.'''
    nav_lon = np.where(nav_lon != -1, nav_lon, np.nan)
    # nav_lon[nav_lon < 0] += 180 
    
    nav_lon2 = nav_lon.copy()
    for i in range(np.shape(nav_lat)[0]):
        value = non_monotone(nav_lon2[i])[0]
        if value != -1:
            # on ne peut pas le récup directement avec notre fonction car celui serait erroné puisque l'on enlève les nan de notre liste.
            index = list(nav_lon2[i]).index(value)
            nav_lon2[i][index:] += 360
        
    nav_lon2[np.isnan(nav_lon2)] = -1
    nav_lat[nav_lat == -1] = np.nan
    nav_lat[np.isnan(nav_lat)] = -100
    return(nav_lat, nav_lon2)


def get_lats_avg(nav_lat, mask):
    '''Function that returns the average latitude for each column of the nav_lat array based on a mask.
    Mandatory to plot zonal means, because of the uneven grid.'''
    lats_avg = np.nanmean(np.where(mask, nav_lat, np.nan), axis = 1)
    lats_avg[np.isnan(lats_avg)] = -100
    index = non_monotone(lats_avg)[1] # ie. la première fois où l'on retrouve -100 comme valeur
    lats_avg[index:] = lats_avg[index - 1]
    return(lats_avg)



def get_new_nav_lat_lon(data_path):
    new_grid = xr.open_dataset(data_path + 'basin_masks_orca1_nemo4p2.nc')
    new_nav_lat, new_nav_lon = new_grid['nav_lat'].values.copy(), new_grid['nav_lon'].values.copy()
    new_grid.close()
    return(transfo(new_nav_lat, new_nav_lon))


def filename(year, card='tm21ah21'):
    ''' The function takes 2 argument: the year of the simulation we want and which type (optional).'''
    if year <=1000:
        if year == 10:
            return(card+'_000'+str(year-9)+'0101_00'+str(year)+'1231_1Y_Age')
        elif year < 100:
            return(card+'_00'+str(year-9)+'0101_00'+str(year)+'1231_1Y_Age')
        elif year == 100:
            return(card+'_00'+str(year-9)+'0101_0'+str(year)+'1231_1Y_Age')
        elif year != 1000:
            return(card+'_0'+str(year-9)+'0101_0'+str(year)+'1231_1Y_Age')
        else:
            return(card+'_0'+str(year-9)+'0101_'+str(year)+'1231_1Y_Age')
    else:
        return(card+'_'+str(year-9)+'0101_'+str(year)+'1231_1Y_Age')
    

def get_new_nav_lat_lon(data_path):
    new_grid = xr.open_dataset(data_path + 'basin_masks_orca1_nemo4p2.nc')
    new_nav_lat, new_nav_lon = new_grid['nav_lat'].values.copy(), new_grid['nav_lon'].values.copy()
    new_grid.close()
    return(transfo(new_nav_lat, new_nav_lon))



def optimized_mask_Pac_creation(nav_lat, nav_lon, n_x, n_y, l_lat_clim, l_lon_clim, mask_clim, clim = False, east_Pac = True):
    '''Function that creates a mask for the Pacific region.'''
    nav_lat2, nav_lon2 = nav_lat.copy(), nav_lon.copy()
    if not clim:
        nav_lat2[nav_lat2==-1], nav_lon2[nav_lon2==-1] = np.nan, np.nan
        nav_lon2[nav_lon2 < 0] += 360  # Adjust negative longitudes
    mask = np.zeros((n_y, n_x), dtype=bool)  # Initialize the mask array
        
    ilat_approx = approx_depth_array(nav_lat2, l_lat_clim)
    ilon_approx = approx_depth_array(nav_lon2, l_lon_clim)

    lat_approx = np.where(~np.isnan(nav_lat2), l_lat_clim[ilat_approx], np.nan)
    lon_approx = np.where(~np.isnan(nav_lon2), l_lon_clim[ilon_approx], np.nan)

    if east_Pac:
        # mask = np.where(~np.isnan(nav_lat2), ((mask_clim[ilat_approx, ilon_approx] == 7) | (mask_clim[ilat_approx, ilon_approx] == 8)) | 
        #                 point_in_trapeze_array(lat_approx, lon_approx, lat_max, lat_min, lon_lat_min_w, lon_lat_max_w, lon_lat_min_e, lon_lat_max_e), False)
        mask = np.where(~np.isnan(nav_lat2), ((mask_clim[ilat_approx, ilon_approx] == 7) | (mask_clim[ilat_approx, ilon_approx] == 8)) | ((nav_lat2 <= -8.5) & (nav_lon2 >= 145) & (nav_lon2 <=292)), False)
    else:
        # mask = np.where(~np.isnan(nav_lat2), (mask_clim[ilat_approx, ilon_approx] == 7) | 
        #             point_in_trapeze_array(lat_approx, lon_approx, lat_max, lat_min, lon_lat_min_w, lon_lat_max_w, lon_lat_min_e, lon_lat_max_e), False)
        mask = np.where(~np.isnan(nav_lat2), (mask_clim[ilat_approx, ilon_approx] == 7) | ((nav_lat2 <= -8.5) & (nav_lon2 >= 145) & (nav_lon2 <=292)), False)
    return mask


def mask_Pac_creation_marginSeas(nav_lat, nav_lon, n_x, n_y, l_lat_clim, l_lon_clim, mask_clim, clim = False):
    nav_lat2, nav_lon2 = nav_lat.copy(), nav_lon.copy()
    if not clim:
        nav_lat2[nav_lat2==-1], nav_lon2[nav_lon2==-1] = np.nan, np.nan
        nav_lon2[nav_lon2 < 0] += 360  # Adjust negative longitudes
    mask = np.zeros((n_y, n_x), dtype=bool)  # Initialize the mask array
        
    ilat_approx = approx_depth_array(nav_lat2, l_lat_clim)
    ilon_approx = approx_depth_array(nav_lon2, l_lon_clim)

    lat_approx = np.where(~np.isnan(nav_lat2), l_lat_clim[ilat_approx], np.nan)
    lon_approx = np.where(~np.isnan(nav_lon2), l_lon_clim[ilon_approx], np.nan)

    mask = np.where(~np.isnan(nav_lat2), ((mask_clim[ilat_approx, ilon_approx] == 7) | (mask_clim[ilat_approx, ilon_approx] == 8) | 
                                          (mask_clim[ilat_approx, ilon_approx] == 13) | (mask_clim[ilat_approx, ilon_approx] == 14)) | 
                    ((nav_lat2 <= -8.5) & (nav_lon2 >= 145) & (nav_lon2 <=292)), False)

    return mask

def optimized_mask_Atl_creation(nav_lat, nav_lon, n_x, n_y, l_lat_clim, l_lon_clim, mask_clim, clim = False):
    '''Function that creates a mask for the Atlantic region.'''
    nav_lat2, nav_lon2 = nav_lat.copy(), nav_lon.copy()
    if not clim:
        nav_lat2[nav_lat2==-1], nav_lon2[nav_lon2==-1] = np.nan, np.nan
        nav_lon2[nav_lon2 < 0] += 360  # Adjust negative longitudes
        nav_lon2[nav_lon > 360] += -360
    mask = np.zeros((n_y, n_x), dtype=bool)  # Initialize the mask array
        
    ilat_approx = approx_depth_array(nav_lat2, l_lat_clim)
    ilon_approx = approx_depth_array(nav_lon2, l_lon_clim)

    lat_approx = np.where(~np.isnan(nav_lat2), l_lat_clim[ilat_approx], np.nan)
    lon_approx = np.where(~np.isnan(nav_lon2), l_lon_clim[ilon_approx], np.nan)
    
    mask = np.where(~np.isnan(nav_lat2), ((mask_clim[ilat_approx, ilon_approx] >= 9) & (mask_clim[ilat_approx, ilon_approx] <= 11)) | ((nav_lat2 <= -8.5) & ((nav_lon2 <= 25) | (nav_lon2 >= 292))), False)
    return mask

def optimized_mask_Ind_creation(nav_lat, nav_lon, n_x, n_y, l_lat_clim, l_lon_clim, mask_clim, clim = False):
    '''Function that creates a mask for the Indian region.'''
    nav_lat2, nav_lon2 = nav_lat.copy(), nav_lon.copy()
    if not clim:
        nav_lat2[nav_lat2==-1], nav_lon2[nav_lon2==-1] = np.nan, np.nan
        nav_lon2[nav_lon2 < 0] += 360  # Adjust negative longitudes
        nav_lon2[nav_lon > 360] += -360
    mask = np.zeros((n_y, n_x), dtype=bool)  # Initialize the mask array
        
    ilat_approx = approx_depth_array(nav_lat2, l_lat_clim)
    ilon_approx = approx_depth_array(nav_lon2, l_lon_clim)

    lat_approx = np.where(~np.isnan(nav_lat2), l_lat_clim[ilat_approx], np.nan)
    lon_approx = np.where(~np.isnan(nav_lon2), l_lon_clim[ilon_approx], np.nan)
    
    mask = np.where(~np.isnan(nav_lat2), ((mask_clim[ilat_approx, ilon_approx] >= 3) & (mask_clim[ilat_approx, ilon_approx] <= 6)) | ((nav_lat2 <= 0) & ((nav_lon2 >= 25) & (nav_lon2 <= 120))), False)
    return mask


def average_datasets(dir_datasets):
    '''Function that averages the datasets in the directory dir_datasets.'''
    os.makedirs(dir_datasets+'averaged', exist_ok = True)
    for file in os.listdir(dir_datasets):
        if file == 'averaged':
            continue
        print(file)
        ds = xr.open_dataset(dir_datasets+file)
        ds_averaged = ds.mean(dim='time_counter')
        ds.close()
        ds_averaged.to_netcdf(dir_datasets+'averaged/'+file.replace('.nc', '')+'_averaged.nc')
        ds_averaged.close()



        
def get_filterAA(nav_lat, nav_lon):
    return ((nav_lat <= -68.25) | ((nav_lat <= -62.5) & ((nav_lon <= 178) | (nav_lon >= 292))))

def get_filterNA(nav_lat, nav_lon, C14 = False):
    if C14:
        return((nav_lat <= 75) & (nav_lat >= 30) & ((nav_lon <= 15) | (nav_lon >= 285)) & (((nav_lon <= 360) | (nav_lat >= 48)) & ((nav_lon <= 350) | (nav_lat >= 40))))
    else:
        return(((nav_lat >= 30) & ((nav_lon <= 45) | (nav_lon >= 265))) | (nav_lat >= 65))

def get_filterHS(nav_lat, nav_lon):
    return((nav_lat <= -45) & ~get_filterAA(nav_lat, nav_lon))

def get_filterMS(nav_lat, nav_lon):
    return((nav_lat <= -30) & (nav_lat >= -45))

def get_filterLL(nav_lat, nav_lon):
    return((nav_lat >= -30) & (nav_lat <= 30))

def get_filterNP(nav_lat, nav_lon):
    return((nav_lat >= 30) & (nav_lat <= 65) & (nav_lon >= 105) & (nav_lon <= 255))


def get_seafloor_mask(ds_averaged, labelx = 'x', labely = 'y', labeldye = 'DyeLL'):
    '''Function that return the depth index for the seafloor.'''
    n_depth, n_y, n_x = np.shape(ds_averaged[labeldye])
    mask_seafloor = np.zeros((n_depth, n_y, n_x), dtype = bool)
    seafloor_depth_index = np.nansum(~np.isnan(ds_averaged[labeldye]), axis = 0)
    for x in range(n_x):
        for y in range(n_y):
            if seafloor_depth_index[y, x] != 0:
                mask_seafloor[seafloor_depth_index[y, x] - 1, y, x] = True
    return mask_seafloor


def get_extrapolated_dyes(simu, data_path = 'D:/Data/NEMO/', year1 = 2900, year2 = 3000):
    '''Function that extrapolates the dyes based on the difference between two timesteps of written output files.'''
    extrapolated_dyes = np.zeros((6, 75, 331, 360))
    aux1, aux2 = xr.open_dataset(data_path + simu + '/averaged/' + filename(year1, card = simu) + '_averaged.nc'), xr.open_dataset(r'D:/Data/NEMO/' + simu + '/averaged/' + filename(year2, card = simu) + '_averaged.nc')
    sum_dyes_grad = (aux2['DyeLL'] + aux2['DyeMS'] + aux2['DyeNP'] + aux2['DyeHS'] + aux2['DyeNA'] + aux2['DyeAA']).values - (aux1['DyeLL'] + aux1['DyeMS'] + aux1['DyeNP'] + aux1['DyeHS'] + aux1['DyeNA'] + aux1['DyeAA']).values
    sum_dyes_year2 = (aux2['DyeLL'] + aux2['DyeMS'] + aux2['DyeNP'] + aux2['DyeHS'] + aux2['DyeNA'] + aux2['DyeAA']).values
    sum_dyes_grad[sum_dyes_grad == 0] = 1 # this line is mandatory to not get inf values
    for dye in l_patches2:
        extrapolated_dyes[l_patches2.index(dye)] = aux2['Dye' + dye].values + (aux2 - aux1)['Dye' + dye].values/sum_dyes_grad * (1 - sum_dyes_year2)
    return extrapolated_dyes

def get_extrapolated_dyes2(simu, year1 = 2900, year2 = 3000):
    extrapolated_dyes = np.zeros((6, 75, 331, 360)); dyes_grad = np.zeros((6, 75, 331, 360))
    aux1, aux2 = xr.open_dataset(r'D:/Data/NEMO/' + simu + '/averaged/' + filename(year1, card = simu) + '_averaged.nc'), xr.open_dataset(r'D:/Data/NEMO/' + simu + '/averaged/' + filename(year2, card = simu) + '_averaged.nc')
    
    for dye in dyes: dyes_grad[dyes.index(dye)] = (aux2 - aux1)[dye].values; dyes_grad[dyes_grad == 0] = np.nan
    dyes_grad[dyes_grad < 0] = 0 # ignore the negative variations
    
    sum_dyes_grad = np.nansum(dyes_grad, axis = 0) 
    sum_dyes_year2 = (aux2['DyeLL'] + aux2['DyeMS'] + aux2['DyeNP'] + aux2['DyeHS'] + aux2['DyeNA'] + aux2['DyeAA']).values
    
    to_be_filled = 1 - sum_dyes_year2
    sum_dyes_grad[sum_dyes_grad == 0] = 1 # this line is mandatory to not get inf values
    for dye in dyes:
        extrapolated_dyes[dyes.index(dye)] = aux2[dye].values + dyes_grad[dyes.index(dye)]/sum_dyes_grad * (1 - sum_dyes_year2)
    return (extrapolated_dyes)

def get_extrapolated_age(simu, year1 = 2900, year2 = 3000):
    '''Function that extrapolates the dyes based on the difference between two timesteps of written output files.'''
    extrapolated_age = np.zeros((75, 331, 360))
    aux1, aux2 = xr.open_dataset(r'D:/Data/NEMO/' + simu + '/averaged/' + filename(year1, card = simu) + '_averaged.nc'), xr.open_dataset(r'D:/Data/NEMO/' + simu + '/averaged/' + filename(year2, card = simu) + '_averaged.nc')
    sum_dyes_grad = (aux2['DyeLL'] + aux2['DyeMS'] + aux2['DyeNP'] + aux2['DyeHS'] + aux2['DyeNA'] + aux2['DyeAA']).values - (aux1['DyeLL'] + aux1['DyeMS'] + aux1['DyeNP'] + aux1['DyeHS'] + aux1['DyeNA'] + aux1['DyeAA']).values
    sum_dyes_year2 = (aux2['DyeLL'] + aux2['DyeMS'] + aux2['DyeNP'] + aux2['DyeHS'] + aux2['DyeNA'] + aux2['DyeAA']).values
    sum_dyes_grad[sum_dyes_grad == 0] = 1 # this line is mandatory to not get inf values
    sum_dyes_grad[sum_dyes_grad < 0] = 1
    extrapolated_age = aux2['Age'].values + (aux2 - aux1)['Age'].values/sum_dyes_grad * (1 - sum_dyes_year2)
    return extrapolated_age


def return_year(filename):
    '''Function that takes the filename of a NEMO output simulation and that return the last year of the simulation'''
    parts = filename.split('_')
    return(int(parts[2].replace('1231', '')))


def return_simulation(filename):
    '''Function that takes the filename of a NEMO output simulation and that return the associated simulation name'''
    parts = filename.split('_')
    return(parts[0])


def vol_depth_TMI(TMI, dgrille = 2):
    '''Function that takes the TMI file and that computes the volume of each grid cell and return the associated array'''
    l_lat_TMI, l_lon_TMI = TMI['yt'].values.copy(), TMI['xt'].values.copy(); l_depth_TMI = TMI['zt'].values
    n_y = len(l_lat_TMI); n_x = len(l_lon_TMI); n_depth = len(l_depth_TMI)

    # here we compute the area of the ocean surface
    dy = earth_rad*dgrille*(np.pi/180)
    dx = earth_rad*dgrille*(np.pi/180)*abs(np.cos(l_lat_TMI*np.pi/180))
    area = np.empty((n_y, n_x))
    for ilon in range(n_x): 
        for ilat in range(n_y): area[ilat, ilon] = dy * dx[ilat]

    
    vol_depth_TMI = np.empty((n_depth, n_y, n_x))
    # a table which counts the number of non-nan values in each cell, thus giving us (in a 2nd time) the maximum depth at each grid cell
    number_levels = np.count_nonzero(~np.isnan(TMI['dyeAA']), axis = 0) - 1
    vol_depth_TMI[0, :, :] = np.where(number_levels > 0, area, 0) * (l_depth_TMI[1] - l_depth_TMI[0])/2
    
    for i in range(1, n_depth - 2):
        aux = np.where(number_levels > i, area, 0)
        vol_depth_TMI[i, :, :] = aux*((l_depth_TMI[i + 1] + l_depth_TMI[i])/2 - (l_depth_TMI[i] + l_depth_TMI[i - 1])/2)
    vol_depth_TMI[n_depth - 1, :, :] = np.where(number_levels == n_depth, area, 0) * (l_depth_TMI[-1] - l_depth_TMI[n_depth - 2])/2
    
    return(vol_depth_TMI)


def area_TMI(TMI, dgrille = 2):
    l_lat_TMI, l_lon_TMI = TMI['yt'].values.copy(), TMI['xt'].values.copy(); l_depth_TMI = TMI['zt'].values
    n_y = len(l_lat_TMI); n_x = len(l_lon_TMI); n_depth = len(l_depth_TMI)

    # here we compute the area of the ocean surface
    dy = earth_rad*dgrille*(np.pi/180)
    dx = earth_rad*dgrille*(np.pi/180)*abs(np.cos(l_lat_TMI*np.pi/180))
    area = np.empty((n_y, n_x))
    for ilon in range(n_x): 
        for ilat in range(n_y): area[ilat, ilon] = dy * dx[ilat]
    return(area)


def extract_year(TMI, year, fields = ['ideal_age']):
    aux_ds = xr.Dataset(); i_year = list(TMI['years']).index(year)

    aux_ds['xt'] = xr.DataArray(TMI['xt'], dims=('Longitude'))
    aux_ds['yt'] = xr.DataArray(TMI['yt'], dims=('Latitude'))
    aux_ds['zt'] = xr.DataArray(TMI['zt'], dims=('Depth'))

    for field in fields:
        aux_ds[field] = xr.DataArray(TMI[field][i_year].values, dims=('Depth', 'Latitude', 'Longitude'))
    return aux_ds

def extension_mask_Pac(nav_lon, nav_lat, NEMO = True):
    nav_lat2, nav_lon2 = nav_lat.copy(), nav_lon.copy()
    if NEMO:
        nav_lat2[nav_lat2==-1], nav_lon2[nav_lon2==-1] = np.nan, np.nan
        nav_lon2[nav_lon2 < 0] += 360  # Adjust negative longitudes
    return ((nav_lon2 >= 100) & (nav_lon2 <= 295) & (nav_lat2 <= -50))


def plot_climato_month(climato_winds, l_lat_cas, l_lon_cas, basin_masks):
    '''
    Function thazt takes into argument the climatology of the winds, the list of lat/lon and the basin_masks of Casimir. 
    It returns the mask of the continents, as well as the esatward/northward wind stress and the list of longitude starting from 0 to 360.
    '''
    l_lon_climato, l_lat_climato = climato_winds['longitude'].values.copy(), climato_winds['latitude'].values.copy()
    l_lon_climato = np.where(l_lon_climato < 0, 360 + l_lon_climato, l_lon_climato)
    l_ilat_approx, l_ilon_approx = approx_depth_array(l_lat_climato, l_lat_cas), approx_depth_array(l_lon_climato, l_lon_cas)
    l_ilon_approx2D, l_ilat_approx2D = np.meshgrid(l_ilon_approx, l_ilat_approx)

    mask_continents_climato = np.isnan(basin_masks[l_ilat_approx2D, l_ilon_approx2D])
    l_lon_climato2D, l_lat_climato2D = np.meshgrid(l_lon_climato, l_lat_climato)
    l_lon_climato2D = np.where(l_lon_climato2D < 0, 360 + l_lon_climato2D, l_lon_climato2D)

    l_lon_plot = np.empty(np.shape(l_lon_climato)); l_lon_plot[:720] = l_lon_climato[720:]; l_lon_plot[720:] = l_lon_climato[:720]
    e_stress = np.empty(np.shape(climato_winds['eastward_stress'].values)); e_stress[:, :, :720] = climato_winds['eastward_stress'].values[:, :, 720:]
    e_stress[:, :, 720:] = climato_winds['eastward_stress'].values[:, :, :720]
    n_stress = np.empty(np.shape(climato_winds['northward_stress'].values)); n_stress[:, :, :720] = climato_winds['northward_stress'].values[:, :, 720:]
    n_stress[:, :, 720:] = climato_winds['northward_stress'].values[:, :, :720]

    mask_continents_plot = np.empty(np.shape(mask_continents_climato)); mask_continents_plot[:, :720] = mask_continents_climato[:, 720:]; mask_continents_plot[:, 720:] = mask_continents_climato[:, :720]

    return(l_lon_plot, e_stress, n_stress, mask_continents_plot)
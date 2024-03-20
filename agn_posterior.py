import matplotlib
matplotlib.use("Agg");
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
#import healpy as hp
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u 
from scipy.stats import norm
#from joblib import Parallel, delayed
from scipy import interpolate
from scipy.integrate import simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from math import pi
import random
import emcee
from scipy.optimize import curve_fit
from multiprocessing import Pool
import corner
from multiprocessing import cpu_count
from optparse import OptionParser

import os
import sys
import time

import random

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Tableau Color Blind 10
tableau20blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
                  (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
                  (255, 188, 121), (207, 207, 207)]

# Rescale to values between 0 and 1
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)
for i in range(len(tableau20blind)):
    r, g, b = tableau20blind[i]
    tableau20blind[i] = (r / 255., g / 255., b / 255.)
    
def plt_style():
    plt.rcParams.update({
    'lines.linewidth':1.0,
    'lines.linestyle':'-',
    'lines.color':'black',
    'font.family':'serif',
    'font.weight':'normal',
    'font.size':16.0,
    'text.color':'black',
    'text.usetex':True,
    'axes.edgecolor':'black',
    'axes.linewidth':1.0,
    'axes.grid':False,
    'axes.titlesize':'large',
    'axes.labelsize':'large',
    'axes.labelweight':'normal',
    'axes.labelcolor':'black',
    'axes.formatter.limits':[-4,4],
    'xtick.major.size':7,
    'xtick.minor.size':4,
    'xtick.major.pad':8,
    'xtick.minor.pad':8,
    'xtick.labelsize':'large',
    'xtick.minor.width':1.0,
    'xtick.major.width':1.0,
    'ytick.major.size':7,
    'ytick.minor.size':4,
    'ytick.major.pad':8,
    'ytick.minor.pad':8,
    'ytick.labelsize':'large',
    'ytick.minor.width':1.0,
    'ytick.major.width':1.0,
    'legend.numpoints':1,
    'legend.fontsize':'large',
    'legend.shadow':False,
    'legend.frameon':False})
    


def lnprior(theta):

    if model=='lam':
        lam_pr = theta[0]
        if 0.2 < lam_pr < 0.8:
            return 0.0
        return -np.inf
    elif model=='lam_h0_Om0':
        lam_pr, H0_pr, omegam_pr = theta
        if 0.0 < lam_pr < 1.0 and 0.0 < H0_pr < 120.0 and  0.0 < omegam_pr < 1.0: #h0 prior was 60-80 until 06-set 
            return 0.0
        return -np.inf
    elif model=='lam_h0_Om0_w':
        if 0. < lam_pr < 1.0 and 60. < H0_pr < 80.0 and  0.0 < omegam_pr < 1.0 and -2.5 < w_pr < -0.01:
            return 0.0
        return -np.inf
    else:
        lam_pr, H0_pr = theta
        if 0.2 < lam_pr < 0.8 and 60.0 < H0_pr < 80.0:
            return 0.0
        return -np.inf

def lnlike(lam_li,s_arr,b_arr,f):
    return np.sum(np.log(lam_li*s_arr+b_arr))-f*lam_li

def lnprob(theta): #cand_hpixs_arr, cand_zs_arr, pb, distnorm,distmu,distsigma, f,N_GW_followups #model
    lp = lnprior(theta)
    if model=='lam':
        lam_po=theta
        h0_=70
        omegam_=0.3
    elif model=='lam_h0_Om0':
        lam_po, H0_po, omegam_po = theta
    elif model=='lam_h0_Om0_w':
        lam_po, H0_po, omegam_po,w_x = theta
    else:
        lam_po, H0_po = theta
        omegam_=0.3
    if not np.isfinite(lp):
        return -np.inf
    #zs_arr=np.linspace(z_min_b,z_max_b,num=1000) #changing z limits per event 20220706
    lnlike_arr= np.zeros(N_GW_followups)
    if model=='lam':
        thiscosmo = FlatLambdaCDM(H0=h0_, Om0=omegam_)
    elif model=='lam_h0_Om0':
        thiscosmo = FlatLambdaCDM(H0=H0_po, Om0=omegam_po)
    else: 
        thiscosmo = FlatLambdaCDM(H0=H0_po, Om0=omegam_)
    for i in range(N_GW_followups):
        zs_arr=np.linspace(z_min_b,z_max_b[i],num=1000) #changing z limits per event 20220706
        this_cand_hpixs=cand_hpixs[i] # _arr
        this_cand_zs=cand_zs[i] # _arr
        if len(this_cand_zs)<1: #this is the check I added.
            lnlike_arr[i]=-lam_po
            lnlikesum=np.sum(lnlike_arr,axis=0) # Is this really necessary? Just add not to change things in the code, but I guess We dont need it.
            continue
        cand_ds=thiscosmo.luminosity_distance(this_cand_zs)
        ncands=this_cand_zs.shape[0]
                 
        jacobian = thiscosmo.comoving_distance(this_cand_zs).value+(1.+this_cand_zs)/thiscosmo.H(this_cand_zs).value
        #Renormalize posterior along los to be normalized between the same z range as the background
        new_normaliz = np.ones(ncands)
        ds_from_z=thiscosmo.luminosity_distance(zs_arr).value
        jacobian_arr = thiscosmo.comoving_distance(zs_arr).value+(1.+zs_arr)/thiscosmo.H(zs_arr).value
        for l in range(ncands):
          
            this_post = gauss(distmus[i][this_cand_hpixs[l]],distsigmas[i][this_cand_hpixs[l]], ds_from_z) * ds_from_z**2 * jacobian_arr
            new_normaliz[l] = distnorms[i][this_cand_hpixs[l]]*np.trapz(this_post, zs_arr)
        
        s_arr = pbs[i][this_cand_hpixs] * distnorms[i][this_cand_hpixs] * norm(distmus[i][this_cand_hpixs], distsigmas[i][this_cand_hpixs]).pdf(cand_ds) * cand_ds**2 /pb_frac /new_normaliz * jacobian

        pb_Vunif_this=thiscosmo.comoving_distance(zs_arr).value**2/thiscosmo.H(zs_arr).value
        normaliz = np.trapz(pb_Vunif_this,zs_arr)
        b_arr = thiscosmo.comoving_distance(this_cand_zs).value**2/thiscosmo.H(this_cand_zs).value/normaliz/len_roi[i]*B_expected_n[i]#clecio fix 20220624

        #Check there are canidates at all
        if (s_arr.shape[0])>0:
            #print(b_arr[np.where(s_arr==min(s_arr))])
            #lnlike_arr.append(lnlike(s_arr,b_arr,lam_arr,f))
            lnlike_arr[i]=lnlike(lam_po,s_arr.value,b_arr,f)
        else:
            #I think we need to do something different here, maybe s_arr needs to be set to 0
            lnlike_arr[i]=-lam_po

        lnlikesum=np.sum(lnlike_arr,axis=0)
        #fixer=lnlikesum[0]
        #like=np.exp(lnlikesum-fixer)
        #normalization=np.trapz(like,lam_arr)

    return lp + lnlikesum


def lnprob_original(theta,cand_hpixs_arr, cand_zs_arr, pb, distnorm,distmu,distsigma, f,N_GW_followups):
    lp = lnprior(theta)
    lam_po, H0_po = theta
    omegam=0.3
    if not np.isfinite(lp):
        return -np.inf
    zs_arr=np.linspace(z_min_b,z_max_b,num=1000) 
    lnlike_arr= np.zeros(N_GW_followups)
    thiscosmo = FlatLambdaCDM(H0=H0_po, Om0=omegam)
    for i in range(N_GW_followups):
        this_cand_hpixs=cand_hpixs_arr[i]
        this_cand_zs=cand_zs_arr[i]
        cand_ds=thiscosmo.luminosity_distance(this_cand_zs)
        ncands=this_cand_zs.shape[0]
        #ddL/dz
        jacobian = thiscosmo.comoving_distance(this_cand_zs).value+(1.+this_cand_zs)/thiscosmo.H(this_cand_zs).value
        #Renormalize posterior along los to be normalized between the same z range as the background
        new_normaliz = np.ones(ncands)
        ds_from_z=thiscosmo.luminosity_distance(zs_arr).value
        jacobian_arr = thiscosmo.comoving_distance(zs_arr).value+(1.+zs_arr)/thiscosmo.H(zs_arr).value
        for l in range(ncands):
            #this_post = gauss(distmu[cand_hpixs[l]],distsigma[cand_hpixs[l]], ds_arr_norm) * ds_arr_norm**2
            #new_normaliz[l] = distnorm[cand_hpixs[l]]*np.trapz(this_post, ds_arr_norm)
            this_post = gauss(distmu[i][this_cand_hpixs[l]],distsigma[i][this_cand_hpixs[l]], ds_from_z) * ds_from_z**2 * jacobian_arr
            new_normaliz[l] = distnorm[i][this_cand_hpixs[l]]*np.trapz(this_post, zs_arr)
        s_arr = pb[i][this_cand_hpixs] * distnorm[i][this_cand_hpixs] * norm(distmu[i][this_cand_hpixs], distsigma[i][this_cand_hpixs]).pdf(cand_ds) * cand_ds**2 /pb_frac /new_normaliz * jacobian

        #normaliz = np.trapz(ds_arr_norm**2,ds_arr_norm)
        #b_arr = cand_ds**2/normaliz/len(idx_sort_cut)*B_expected_n

        normaliz = np.trapz(pb_Vunif,zs_arr)
        b_arr = thiscosmo.comoving_distance(this_cand_zs).value**2/thiscosmo.H(this_cand_zs).value/normaliz/len(idx_sort_cut)*B_expected_n

        #Check there are canidates at all
        if (s_arr.shape[0])>0:
            #lnlike_arr.append(lnlike(s_arr,b_arr,lam_arr,f))
            lnlike_arr[i]=lnlike(lam_po,s_arr.value,b_arr,f)

        else:
            #I think we need to do something different here, maybe s_arr needs to be set to 0
            lnlike_arr[i]=-lam_po

        lnlikesum=np.sum(lnlike_arr,axis=0)
        #fixer=lnlikesum[0]
        #like=np.exp(lnlikesum-fixer)
        #normalization=np.trapz(like,lam_arr)

    return lp + lnlikesum



def gauss(mu_gauss, std_gauss, x_value):
    return np.exp(-(x_value-mu_gauss)*(x_value-mu_gauss)/(2*std_gauss*std_gauss))/(std_gauss*(2.*pi)**0.5)

def cl_around_mode(edg,myprob):

    peak = edg[np.argmax(myprob)]
    idx_sort_up=np.argsort(myprob)[::-1]

    i=0
    bins=[]
    integr=0.
    bmax=idx_sort_up[0]
    bmin=bmax
    bmaxbound=edg.shape[0]-1
    bminbound=0

    while (integr<0.68):
        if bmax==bmaxbound:
            bmin = bmin-1
        elif bmin==bminbound:
            bmax=bmax+1
        elif (myprob[bmax+1]>myprob[bmin-1]):
            #print("Adding ",bmax_lo+1)
            bmax = bmax+1
            bmin = bmin
            #bins_now_good = np.append(bins_now_good,
            bins.append(bmax+1)
        else:
            #print("Adding ",bmin-1)
            bmin = bmin-1
            bmax = bmax
            bins.append(bmin-1)
        integr = simps(myprob[bmin:bmax],edg[bmin:bmax])
    print(integr, edg[bmin], edg[bmax])
    
    return peak, edg[bmin], edg[bmax]


def plotgwmap(bay_file,ra=None,dec=None, title='Bayestar',out_name='Bayestar.png',plot_center=[320,20]):


    fig = plt.figure(111, figsize=(15, 10)) 
    #with World2ScreenMPL(fig, 
    #    fov=FOV * u.deg,
    #    center=SkyCoord(plot_center[0], plot_center[1], unit='deg', frame='icrs'),
    #    coordsys="icrs",
    #    rotation=Angle(0, u.degree),
    #    projection="AIT") as wcs:
            
    import ligo.skymap.plot

    ax = plt.axes(projection='astro aitoff')#astro #rotate=Angle(0, u.degree) #radius='50 deg' #'astro zoom' center=SkyCoord(plot_center[0], plot_center[1], unit='deg', frame='icrs')
    ax.grid()
    ax.contour_hpx(bay_file )

    ax.set_xlabel('ra')
    #plt.xlabel('ra')
    ax.set_ylabel('dec')
    ax.set_title(title)
    ax.grid(b=True,color="black", linestyle="-")
    plt.savefig(out_name)

if __name__ == '__main__':
    parser = OptionParser(__doc__)
    parser.add_option('--model', default='h0lam', help="define the model and free parameters")
    parser.add_option('--lamb', default=0.7, help="Lambda")
    parser.add_option('--nfu', default=200, help="Number of follow-up events")
    parser.add_option('--run_name', default="Test", help="name to save outputfiles")
    parser.add_option('--background',  default=1.0, help="Number of expected backgroung events")
    parser.add_option('--fraction', default=1.0, help="f=1 in the limit where we detect all AGNs flares")
    options, args = parser.parse_args(sys.argv[1:])




#cand_hpixs_arr, cand_zs_arr, pb, distnorm,distmu,distsigma, f,N_GW_followups
global cand_hpixs
global cand_zs
global pbs
global distnorms
global distmus
global distsigmas
global f
global N_GW_followups
global model
global B_expected_n
global z_max_b
global z_min_b
global len_roi
len_roi=[]
B_expected_n=[]

z_max_b=[] #new attemp limiting the redshift

n_back=float(options.background)
model=options.model
#end_file=options.end_file#'h0lam'
#f=1 in the limit where we detect all AGNs flares
f=float(options.fraction)#1.
lamb=float(options.lamb)#0.7#0.5 #da_array=[0.2,0.9]
N_GW_followups=int(options.nfu)#200#50#280#280 #[50,10]
limit_agn_cand=2000#200#50#50#100
limit_agn_cand_low=0
randomize=True
limit_nfu=True
nfu_limit=280
flare_factor=0.0001#0.01#0.1#1.0
#lamba_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
#N_GW_followups=[500,200,100,100,100,100,100,50,50,50,50]
events_file='stats_bbh_GWTC3_pareto_285.tsv'#'pareto_id_Bs_top3percent.dat'#'pareto_id_Bs_top3percent.dat'##'pareto_id_Bs_top3percent.dat' 'stats_bbh_GWTC3_pareto_285.tsv'
event_name='O4_top3'
H0=70.
omegam=0.3
mapdir='/home/Dscmind/dadosCloud/bbh_GWTC3/'#'/home/dadosSSD/bbh_GWTC3/'#'/home/cenpes/Dscmind/dadosCloud/bbh_GWTC3/'#'/home/dadosSSD/bbh_GWTC3/'#'/home/cleciobom/bbh_o4_sims/mass_gt_50_flatten_fits/'##'/home/cleciobom/bbh_o4_sims/mass_gt_50_flatten_fits/'#'/home/dadosSSD/bbh_GWTC3/'#'/home/cleciobom/bbh_o4_sims/mass_gt_50_flatten_fits/'
steps= 25000#100000 #5000
autocorr_test=True
burnin=200
max_n = steps#100000
z_min_b = 0.01 #Minimum z for background events
#z_max_b = 2.0#1.# change zlimits 20220706
mpi=False #True # MPI paralelizes in multiples CPUS
multiprocessing=False # Multiprocessing parallelizes in a single CPU
njobs=192#64
lam_arr=np.linspace(0,1.,num=100)
#events_data=np.genfromtxt(mapdir+events_file, delimiter="\t")
sufix=options.run_name#'monitor_10ksample_h0_lam'+str(lamb)+'gwfollowup'+str(N_GW_followups)+'_new_sims_poisson_10_bkg_revPalmese_run03'
sufix_map='new_sims_0bkg'
cosmo = FlatLambdaCDM(H0=H0, Om0=omegam)
out_name=event_name

print('parameters')
print(model)
print(n_back)
print(f)
print(N_GW_followups)
print(sufix)
print(lamb)

b_agn=True # set true to define the background here in the code, false to take it from the sims
plot_gw=False

if multiprocessing==False and mpi==False:
    os.environ["OMP_NUM_THREADS"] = "60"
else:
    os.environ["OMP_NUM_THREADS"] = "1"

#Select 0 or 1 signal events

if b_agn==True:
    sim_pareto = np.genfromtxt(mapdir+events_file, dtype=type("mytxt"), comments='#', delimiter='\t', skip_header=2) #dtype=<class 'float'>, comments='#', delimiter=None, skip_header=0
    if randomize==True:
        idx_rand= np.random.permutation(len(sim_pareto))
        sim_pareto = sim_pareto[idx_rand]

    simid_f=sim_pareto[:,0]
    simid = [element.replace("coinc_event:coinc_event_id:","") for element in simid_f]
    Bn= sim_pareto[:,13].astype("float")
    vol=sim_pareto[:,13].astype("float")
    mapdir=mapdir+"fits/"
    

    
    Bn=Bn*pow(10,-4.75)
    
else:
    sim_pareto = np.genfromtxt(mapdir+events_file)
    simid = sim_pareto[:,0]
    Bn= sim_pareto[:,1]
    
    #map_file_name=mapdir+str(int(simid[i]))+'.fits.fits.gz'
    #hs=fits.open(map_file_name)[1]#.fits.
#print('AGNs number')
#mask_bn= Bn < 300
#print(Bn[mask_bn])
#print(len(Bn[mask_bn]))

randnum=np.random.uniform(0, 1,size=N_GW_followups)
S_cands=np.ones(N_GW_followups,dtype=int)
mask=(randnum>lamb)
S_cands[mask]=0


#maxdist = cosmo.luminosity_distance(z_max_b)  # change zlimits 20220706
#ds_arr_norm=np.linspace(0,maxdist.value,num=1000) #change zlimits 20220706
    
cand_hpixs=[]
cand_zs=[]
pbs,distnorms,distmus,distsigmas = [],[],[],[]
real_N_GW_followups=0
for i in range(N_GW_followups):
    if limit_nfu==True:
        if real_N_GW_followups>=nfu_limit:
            break 

    map_file_name=mapdir+str(int(simid[i]))+'.fits.gz'
    hs=fits.open(map_file_name)[1]#.fits.

    #pb,distmu,distsigma,distnorm = hp.read_map(bayestar_file,field=range(4),verbose=False)#clecio dtype='numpy.float64'
    #NSIDE=hp.npix2nside(pb.shape[0])
    B_cands= np.random.poisson(Bn[i]*flare_factor)#(n_back)
    #n_back #fix 20220630
    print("Background events: ",B_cands)
    print("Expected Background events: ",Bn[i])
    
    pb = hs.data['PROB']
    distnorm= hs.data['DISTNORM']
    distmu= hs.data['DISTMU']
    distsigma= hs.data['DISTSIGMA']
    NSIDE = hs.header['NSIDE']
    #======================= change zlimits 20220706
    pb_hr=pb[np.logical_not(np.isinf(distmu))]
                    
    #distsigma_hr=distsigma[np.logical_not(np.isinf(distmu))]
    distsigma_hr_average=hs.header['DISTSTD']#np.average(distsigma_hr,weights=pb_hr)
    #distsigma_hr_average=np.average(distsigma_hr,weights=pb_hr)
    #distmu_hr=distmu[np.logical_not(np.isinf(distmu))]
    #distmu_hr_average= np.average(distmu_hr,weights=pb_hr)
    #distmu_std=weighted_avg_and_std(distmu_hr, weights=pb_hr)
    distmu_hr_average=hs.header['DISTMEAN'] #np.average(distmu_hr,weights=pb_hr)
    z_limit_event=z_at_value(cosmo.luminosity_distance, (distmu_hr_average+3*distsigma_hr_average) * u.Mpc,zmin=0.00,zmax=5.)#2.0#
    maxdist = cosmo.luminosity_distance(z_limit_event)  #(distmu_hr_average+2*distsigma_hr_average)# change zlimits 20220706
    ds_arr_norm=np.linspace(0,maxdist.value,num=1000) #change zlimits 20220706 #np.linspace(0,maxdist,num=1000)
    #z_limit_event=z_limit_event+0.1 #adding zerr
    #==========================
    hs._close
        
    if plot_gw==True:
        plotgwmap(map_file_name,ra=None,dec=None, title='Bayestar',out_name=sufix_map+str(int(simid[i]))+'.png',plot_center=[320,20])

    pb_frac = 0.90
    idx_sort = np.argsort(pb)
    idx_sort_up = list(reversed(idx_sort))
    sum = 0.
    id = 0
    while sum<pb_frac:
        this_idx = idx_sort_up[id]
        sum = sum+pb[this_idx]
        id = id+1

    idx_sort_cut = idx_sort_up[:id]
    #print("size of  ROI ", len(idx_sort_cut))
    print("Current ID ",simid[i]) 
    print("Current Follow-up ",i) 
    #print(idx_sort_cut)
    #Now get positions and redshifts for the signal candidates

    new_distnorm=distnorm[idx_sort_cut]
    if len(new_distnorm[new_distnorm==0.0])/len(distnorm)>0:
        #np.save("distnormtest"+str(i)+".npy",distnorm)
        print("distnorm fraction ",i)  
        print(len(distnorm[distnorm==0.0])/len(distnorm))
        print(len(new_distnorm[new_distnorm==0.0])/len(distnorm))
        continue
    if (Bn[i]>limit_agn_cand) or (Bn[i]<limit_agn_cand_low):
          print("Disregarding event ",map_file_name," with ",str(Bn[i])," expected background events" )
          continue
    else:
        B_expected_n.append(Bn[i]*flare_factor)
        len_roi.append(len(idx_sort_cut)) 
        real_N_GW_followups+=1
        s_hpixs=np.random.choice(idx_sort_cut, p=pb[idx_sort_cut]/pb[idx_sort_cut].sum(),size=S_cands[i])
        s_zs=[]
        for j in range(S_cands[i]):
            #Sampling posterior
            ln_los=gauss(distmu[s_hpixs[j]],distsigma[s_hpixs[j]], ds_arr_norm)
            post_los=ln_los *ds_arr_norm**2
            #print("Candidate J ",j+1)
            #print (ds_arr_norm)
            #print(post_los)
            #print(post_los.sum())
            s_ds=np.random.choice(ds_arr_norm, p=post_los/post_los.sum())
            #print (s_ds)
            #print("end candidate J ")
            #s_ds=np.random.normal(loc=distmu[s_hpixs[j]], scale=distsigma[s_hpixs[j]])
            #while (s_ds<0):
            #    s_ds=np.random.normal(loc=distmu[s_hpixs[j]], scale=distsigma[s_hpixs[j]])
            s_zs.append(z_at_value(cosmo.luminosity_distance, s_ds * u.Mpc,zmin=0.00,zmax=5.) )

        z_max_b.append(z_limit_event) ### zlimits per event 20220706
        #Now get positions and redshifts for the background candidates   
        #Let's assume they are just uniform in comoving volume between z_min and z_max
        s_zs=np.array(s_zs)
        b_hpixs=np.random.choice(idx_sort_cut, size=B_cands)
        zs_arr=np.linspace(z_min_b,z_limit_event,num=1000) # change z limits per event 20220706 #z_limit_event
        #Define a probability that is uniform in comoving volume
        #in redshift that is D_comoving^2/H
        pb_Vunif=cosmo.comoving_distance(zs_arr).value**2/cosmo.H(zs_arr).value
        b_zs=np.random.choice(zs_arr, p=pb_Vunif/pb_Vunif.sum(), size=B_cands)#
        #b_ds=np.random.choice(ds_arr_norm, p=ds_arr_norm**2/(ds_arr_norm**2).sum(),size=B_cands[i])
        #print(s_zs,b_zs)
    
        #Positions and z for All candidates in this follow up
        cand_hpixs.append(np.concatenate((s_hpixs,b_hpixs)))
        cand_zs.append(np.concatenate((s_zs,b_zs)))
        pbs.append(pb)
        distnorms.append(distnorm)
        distmus.append(distmu)
        distsigmas.append(distsigma)
        #cand_ds=np.concatenate((s_ds,b_ds))
        #ncands=cand_hpixs.shape[0]
        #cand_ds=np.zeros(ncands)
        #for l in range(ncands):
        #    cand_ds[l]=cosmo.luminosity_distance(cand_zs[l]).value
    
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))
if model=='lam':
    pos=[lamb]  + 1e-4 * np.random.randn(32, 1)
elif model=='lam_h0_Om0':
    pos = [lamb,H0,omegam]  + 1e-4 * np.random.randn(32, 3)
else:
    pos = [lamb,H0]  + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape
print ("this is the number of walkers ",nwalkers)

start = time.time()

filename = 'out/'+sufix+'_O4.h5'#"tutorial.h5"
dados_run=[model,lamb,N_GW_followups,sufix,H0,n_back,f,omegam]
#[mmodel,lambda,followups,run_name_bkg,fraction]

np.save('out/'+sufix+'_O4_info.npy',dados_run)

backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

N_GW_followups=real_N_GW_followups
mpi=False
print('final number of events')
print(N_GW_followups)
if mpi:
    from schwimmbad import MPIPool

index = 0 
if mpi:
    with MPIPool() as pool:#processes=10
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(cand_hpixs, cand_zs, pbs, distnorms,distmus,distsigmas, f,N_GW_followups),pool=pool)
        sampler.run_mcmc(pos, steps, progress=True) #progress=True
#elif multiprocessing:
     
#    with Pool(processes=20) as pool:
#        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(cand_hpixs, cand_zs, pbs, distnorms,distmus,distsigmas, f,N_GW_followups),pool=pool)
#        sampler.run_mcmc(pos, steps, progress=True) #

else:

    mypool = Pool(njobs)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,backend=backend, pool=mypool)
    #args=(cand_hpixs, cand_zs, pbs, distnorms,distmus,distsigmas, f,N_GW_followups)

    if autocorr_test==False: 
        sampler.run_mcmc(pos, steps, progress=True)
    else:
        autocorr = np.empty(max_n)

    # This will be useful to testing convergence
        old_tau = np.inf

    # Now we'll sample for up to max_n steps
        for sample in sampler.sample(pos, iterations=max_n, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1
          
            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            else:
                print("Did not converged Keep Walking...") 
            old_tau = tau

        #sampler.run_mcmc(pos, steps, progress=True)
    
end= time.time()
serial_time = end - start
if multiprocessing:
    print ("Multi processing {0:.1f} seconds".format(serial_time)) 
else:
    print("Serial took {0:.1f} seconds".format(serial_time))

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)
if model=='lam':
    labels=[r"$\lambda$"]
    truth_vals=[lamb]
elif model=='lam_h0_Om0':
    labels=[r"$\lambda$",r"$H_0$ [km/s/Mpc]",r"$\Omega_M$"]
    truth_vals=[lamb, H0,omegam]
else:
    labels=[r"$\lambda$",r"$H_0$ [km/s/Mpc]"]
    truth_vals=[lamb, H0]

fig = corner.corner(flat_samples, labels=labels, truths=truth_vals,quantiles=[0.16, 0.5, 0.84],\
                       show_titles=True, title_kwargs={"fontsize": 12})
fig.savefig(sufix+'corner_O4_50_clecio.png',dpi=200)

np.savetxt('out/'+sufix+'O4_clecio.dat',flat_samples)

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:08:27 2019

@author: kutkin
"""

import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import subprocess
from astropy.time import Time
import h5py
import pandas as pd
from matplotlib.patches import Circle
import glob
import logging
logging.info('Starting logger for {}'.format(__name__))
logger = logging.getLogger(__name__)

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import yaml

from cluster import main as cluster

# from configparser import ConfigParser
# config = ConfigParser()
# config.read('ddcal.cfg')
# print config('dical2')


dppp_bin = '/home/soft/DP3/build/DPPP/DPPP'
wsclean_bin = '/home/soft/wsclean/wsclean/build/wsclean'
makesourcedb_bin = '/home/soft/DP3/build/makesourcedb'
bbs2model_bin = '/home/soft/modeltools/build/bbs2model'
render_bin = '/home/soft/modeltools/build/render'

# TODO
def check_binaries():
    pass

def wsclean(msin, pixelsize=3, multifreq=7, autothresh=0.3, automask=3, multiscale=False):
    """
    wsclean
    """
    wsclean_bin = subprocess.check_output('which wsclean; exit 0', shell=True).strip() or wsclean_bin
    logging.debug('wsclean binary: {}'.format(wsclean_bin))
    msbase = os.path.splitext(msin)[0]
    if multiscale:
        multiscale = '-multiscale'
    else:
        multiscale=''
    if multifreq:
        mf = '-join-channels -channels-out {} -fit-spectral-pol 1'.format(multifreq)
    else:
        mf = ''
    cmd = '{wsclean} -name {msbase} -size 3072 3072 -scale {pix}asec -niter 1000000 -auto-threshold {autothresh} \
              {mf} -auto-mask {automask} -save-source-list -mgain 0.8 -local-rms -use-wgridder {multiscale} {msin}'.\
              format(wsclean=wsclean_bin, msbase=msbase, pix=pixelsize, mf=mf, autothresh=autothresh,
                     automask=automask, multiscale=multiscale, msin=msin)
    # print cmd
    cmd = " ".join(cmd.split())
    subprocess.call(cmd, shell=True)

    if multifreq:
        for fname in glob.glob(msbase+'-*.fits'):
            newname = fname.replace('MFS-', '')
            os.rename(fname, newname)
    return 0


def makesourcedb(modelfile, out=None):
    """ Make sourcedb file from a clustered model """
    makesourcedb_bin = subprocess.check_output('which makesourcedb; exit 0', shell=True).strip() or makesourcedb_bin
    logging.debug('makesourcedb binary: {}'.format(makesourcedb_bin))
    out = out or os.path.splitext(modelfile)[0] + '.sourcedb'
    cmd = '{} in={} out={}'.format(makesourcedb_bin, modelfile, out)
    subprocess.call(cmd, shell=True)
    return out


def bbs2model(inp, out=None):
    """ Convert model file to AO format """
    bbs2model_bin = subprocess.check_output('which bbs2model; exit 0', shell=True).strip() or bbs2model_bin
    out = out or os.path.splitext(inp)[0] + '.ao'
    cmd = '{} {} {}'.format(bbs2model_bin, inp, out)
    subprocess.call(cmd, shell=True)
    return out


def render(bkgr, model, out=None):
    render_bin = subprocess.check_output('which render; exit 0', shell=True).strip() or render_bin
    out = out or os.path.split(bkgr)[0] + '/restored.fits'
    cmd = '{} -a -r -t {} -o {} {}'.format(render_bin, bkgr, out, model)
    subprocess.call(cmd, shell=True)
    return out


def dical(msin, srcdb, msout=None, h5out=None, solint=1, startchan=40, nchan=192, mode='phaseonly'):
    """ direction independent calibration with DPPP """
    h5out = h5out or os.path.split(msin)[0] + '/dical.h5'
    msbase = os.path.basename(msin).split('.')[0]
    msout = msout or '{}_{}_{}.MS'.format(msbase, mode, solint)
    cmd = '{dppp_bin} msin={msin} msout={msout} \
          msin.startchan={startchan} \
          msin.nchan={nchan} \
          msout.overwrite=true \
          cal.type=gaincal \
          cal.caltype={mode} \
          cal.sourcedb={srcdb} \
          cal.solint={solint} \
          cal.h5parm={h5out} \
          cal.applysolution=True \
          cal.nchan=31 \
          cal.uvlambdamin=500 \
          steps=[cal] \
          '.format(dppp_bin=dppp_bin, msin=msin, msout=msout,
                   startchan=startchan, nchan=nchan, mode=mode,
                   srcdb=srcdb, solint=solint, h5out=h5out)
    cmd = " ".join(cmd.split())
    subprocess.call(cmd, shell=True)
    return msout

def ddecal(msin, srcdb, msout=None, h5out=None, solint=1, nfreq=15, startchan=0, nchan=192, mode='diagonal', subtract=True):
    """ Perform direction dependent calibration with DPPP """
    h5out = h5out or os.path.split(msin)[0] + '/ddcal.h5'
    msbase = os.path.basename(msin).split('.')[0]
    msout = msout or '{}_{}_{}.MS'.format(msbase,mode, solint)
    cmd = '{dppp_bin} msin={msin} msout={msout} \
          msin.startchan={startchan} \
          msin.nchan={nchan} \
          msout.overwrite=true \
          cal.type=ddecal \
          cal.mode={mode} \
          cal.sourcedb={srcdb} \
          cal.solint={solint} \
          cal.h5parm={h5out} \
          cal.subtract={subtract} \
          cal.propagatesolutions=true \
          cal.propagateconvergedonly=true \
          cal.nchan={nfreq} \
          steps=[cal] \
          '.format(dppp_bin=dppp_bin, msin=msin, msout=msout, startchan=startchan, nchan=nchan, mode=mode,
            srcdb=srcdb, solint=solint, h5out=h5out, subtract=subtract, nfreq=nfreq)
    cmd = " ".join(cmd.split())
    subprocess.call(cmd, shell=True)
    return msout, h5out


def view_sols(h5param):
    """ read and plot the gains """
    path = os.path.split(os.path.abspath(h5param))[0]
    def plot_sols(h5param, key):
        with h5py.File(h5param, 'r') as f:
            grp = f['sol000/{}'.format(key)]
            data = grp['val'][()]
            time = grp['time'][()]
            print data.shape
            ants = ['RT2','RT3','RT4','RT5','RT6','RT7','RT8','RT9','RTA','RTB','RTC','RTD']
            fig = plt.figure(figsize=[20, 15])
            fig.suptitle('Freq. averaged {} gain solutions'.format(key.rstrip('000')))
            for i, ant in enumerate(ants):
                ax = fig.add_subplot(4, 3, i+1)
                ax.set_title(ant)
                if len(data.shape) == 5: # several directions
                    # a = ax.imshow(data[:,:,i,1,0].T, aspect='auto')
                    # plt.colorbar(a)
                    gavg = np.nanmean(data, axis=1)
                    ax.plot((time-time[0])/60.0, gavg[:, i, :, 0], alpha=0.7)
                elif len(data.shape) == 4: # a single direction
                    ax.plot((time-time[0])/60.0, data[:, 0, i, 0], alpha=0.7)

                if i == 0:
                    ax.legend(['c{}'.format(_) for _ in range(data.shape[-2])])
                if i == 10:
                    ax.set_xlabel('Time')
        return fig, ax

    try:
        fig, ax = plot_sols(h5param, 'amplitude000')
        fig.savefig(path + '/amp_sols.png')
    except:
        logger.error('No amplitude solutions found')

    try:
        fig, ax = plot_sols(h5param, 'phase000')
        fig.savefig(path + '/phase_sols.png')
    except:
        logger.error('No phase solutions found')

    # plt.show()

# TODO
def model_apply_threshold(model, threshold=0.0, out=None):
    """
    Clip the model to be above the given threshold

    Parameters
    ----------
    model : STR, model file name
        the input model file name
    threshold : FLOAT, optional
        the threshold above which the components are kept. The default is 0.0.
    out : STR, optional
        The output model filename. The default is None (the model file will be overwritten).

    Returns
    -------
    None.
    """
    out = out or model
    logging.warning('Overwriting the model')
    df = pd.read_csv(model, skipinitialspace=True)
    new = df.query('I>@threshold')
    new.to_csv(out, index=False)
    return out


def main(msin, cfgfile='ddcal.yml'):
    """
    """
    logging.info('Processing {}'.format(msin))
    logging.info('The config file: {}'.format(cfgfile))
    with open(cfgfile) as f:
        cfg = yaml.safe_load(f)


    print cfg['clean1']

    sys.exit()

    mspath = os.path.split(os.path.abspath(msin))[0]

    msbase = os.path.splitext(msin)[0]
    model1 = os.path.join(mspath, 'model1.sourcedb')
    model2 = os.path.join(mspath, 'model2.sourcedb')
    dical1 = os.path.join(mspath, 'dical1.MS')
    h5_1 = os.path.join(mspath, 'dical1.h5')
    dical2 = os.path.join(mspath, 'dical2.MS')
    h5_2 = os.path.join(mspath, 'dical2.h5')
    ddsub = os.path.join(mspath, 'ddsub.MS')
    dd_h5 = os.path.join(mspath, 'ddgains.h5')
    clustered_model = mspath + '/clustered.txt'
    final_image = mspath+'/restored.fits'
    clean_model = mspath+'/dical2-sources.txt'

    if os.path.exists(final_image):
        logging.info('The final image exists. Exiting...')
        return 0

# Clean + DIcal
    # sys.exit()
    if (not os.path.exists(dical1)) and (not os.path.exists(dical2)):
        wsclean(msin, **cfg['clean1']) # fast shallow clean
        makesourcedb(msbase + '-sources.txt', out=model1)
        dical(msin, model1, msout=dical1, h5out=h5_1, **cfg['dical1'])
        wsclean(dical1, **cfg['clean2'])
        makesourcedb(mspath+'/dical1-sources.txt', out=model2)

    if not os.path.exists(dical2):
        dical(dical1, model2, msout=dical2, h5out=h5_2, **cfg['dical2'])
        wsclean(dical2, **cfg['clean3'])

# Take only positive components from the model:
    # clean_model = model_apply_threshold(clean_model, 1e-6, out=clean_model)

    dical_image = glob.glob(mspath+'/dical2-image.fits')[0]
    dical_resid = glob.glob(mspath+'/dical2-residual.fits')[0]

# Cluster
    cluster(dical_image, dical_resid, clean_model, auto=True, nclusters=7)

# Makesourcedb
    makesourcedb(clustered_model, mspath+'/clustered.sourcedb')

# DDE calibration + peeling everything
    ddsub, h5out = ddecal(dical2, mspath+'/clustered.sourcedb', msout=ddsub, h5out=dd_h5,
                            solint=120, mode='diagonal')

# view the solutions
    view_sols(dd_h5)

    wsclean(ddsub, pix=3, multifreq=3, multiscale=True, automask=5, autothresh=0.5)

    model = bbs2model(mspath+'/dical2-sources.txt', mspath+'/model.ao')
    render(mspath+'/ddsub-image.fits', mspath+'/model.ao', out=final_image)

    if os.path.exists(final_image):
        logging.info('Clearing files')
        cmd = 'rm -r {} {}'.format(dical1, ddsub)
        subprocess.call(cmd, shell=True)

    # plt.show()
    return 0

if __name__ == "__main__":
    t0 = Time.now()
    msin = '/home/kutkin/191102001/24/M1403+5324_avg.MS'
    main(msin)
    extime = Time.now() - t0
    print "Execution time: {:.1f} min".format(extime.to("minute").value)


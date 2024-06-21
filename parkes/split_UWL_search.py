# This scripts is used for splitting Parkes/Murriyang UWL subbands data

import numpy as np
from astropy.io import fits

import argparse

import logging

import os
import glob
import subprocess
from multiprocessing import Pool

def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def split_uwl_data(
    logger, in_file, out_file, sub_start, sub_end,
):
    in_file_name = in_file.split("/")[-1]
    logger.info(f"extract subbands of UWL pulsar searching data...")
    logger.info(f"file name: {in_file_name}...")
    logger.info(f"subband Range: {sub_start} ~ {sub_end}...")

    hdulist = fits.open(in_file)
    tbdata = hdulist['SUBINT'].data
    obsbw = hdulist['PRIMARY'].header['OBSBW']
    nbits = int(hdulist['SUBINT'].header['NBITS'])
    nchan = int(hdulist['SUBINT'].header['NCHAN'])
    nsblk = int(hdulist['SUBINT'].header['NSBLK'])
    nsub =  int(hdulist['SUBINT'].header['NAXIS2'])
    npol =  int(hdulist['SUBINT'].header['NPOL'])
    nstot = int(hdulist['SUBINT'].header['NSTOT'])

    logger.info(f"data information: nchan - {nchan}, npol - {npol}, nsub - {nsub}")
    logger.info(f"nsblk - {nsblk}, nbits - {nbits}")
    
    hdu_primary_header = hdulist['PRIMARY'].header
    hdu_subint_header = hdulist['SUBINT'].header

    dat_freq = np.reshape(tbdata['DAT_FREQ'], (nsub, nchan))
    dat_wts = np.reshape(tbdata['DAT_WTS'], (nsub, nchan))
    dat_off = np.reshape(tbdata['DAT_OFFS'], (nsub, npol, nchan))
    dat_scl = np.reshape(tbdata['DAT_SCL'], (nsub, npol, nchan))
    data = np.reshape(tbdata['DATA'], (nsub*nsblk, npol, int(nchan/(8/nbits))))

    logger.info(f"data shape information: data_freq - {dat_freq.shape}, dat_wtd - {dat_wts.shape}, dat_off - {dat_off.shape}")
    logger.info(f"dat_scl - {dat_scl.shape}, data = {data.shape}")
    
    # copy columns from the first file
    indexval = tbdata['INDEXVAL']
    tsubint = tbdata['TSUBINT']
    offssub = tbdata['OFFS_SUB']
    auxdm = tbdata['AUX_DM']
    auxrm = tbdata['AUX_RM']

    ############### determine channels to include #################
    sub_nchan = int(nchan/26)
    sub_nstot = int(nstot/26)

    mask = np.zeros(nchan, dtype=bool)
    mask[sub_start*sub_nchan:(sub_end+1)*sub_nchan] = True

    nchan_new = (sub_end - sub_start + 1)*sub_nchan
    nstot_new = (sub_end - sub_start + 1)*sub_nstot

    logger.info("working out channel indices to start and stop...")

    mask_dat = np.zeros(int(nchan/(8/nbits)), dtype=bool)
    mask_dat[sub_start*int(sub_nchan/(8/nbits)):(sub_end+1)*int(sub_nchan/(8/nbits))] = True

    ############### create some new Columns #######################
    logger.info("creating new columns...")
    col_index = fits.Column(name='INDEXVAL', format='1D', array=indexval)
    col_tsubint = fits.Column(name='TSUBINT', format='1D', array=tsubint, unit='s')
    col_offssub = fits.Column(name='OFFS_SUB', format='1D', array=offssub, unit='s')
    col_auxdm = fits.Column(name='AUX_DM', format='1D', array=auxdm, unit='CM-3')
    col_auxrm = fits.Column(name='AUX_RM', format='1D', array=auxrm, unit='RAD')

    #col_freq = fits.Column(name='DAT_FREQ', format='{0}D'.format(nchan_new*npol), array=dat_freq[:,:,mask])
    #col_wts = fits.Column(name='DAT_WTS', format='{0}E'.format(nchan_new*npol), array=dat_wts[:,:,mask])
    col_freq = fits.Column(name='DAT_FREQ', format='{0}D'.format(nchan_new), array=dat_freq[:,mask])
    col_wts = fits.Column(name='DAT_WTS', format='{0}E'.format(nchan_new), array=dat_wts[:,mask])
    col_off = fits.Column(name='DAT_OFFS', format='{0}E'.format(nchan_new*npol), array=dat_off[:,:,mask])
    col_scl = fits.Column(name='DAT_SCL', format='{0}E'.format(nchan_new*npol), array=dat_scl[:,:,mask])

    #data = np.reshape(data, (nsub, nsblk/(8/nbits), npol, nchan_new)).astype(int)
    data = data[:,:,mask_dat]
    data = np.reshape(data, (nsub, int(nsblk/(8/nbits)), npol, nchan_new)).astype(int)
    logger.info(f"new data shape: {data.shape}...")

    col_data = fits.Column(name='DATA', format='{0}B'.format(nchan_new*int(nsblk/(8/nbits))*npol), dim='({0},{1},{2})'.format(nchan_new, npol, int(nsblk/(8/nbits))), array=data)

    #obsfreq_new = np.mean(dat_freq[0, mask])
    obsfreq_new = np.mean(dat_freq[0, mask])
    obsbw_new = (sub_end - sub_start + 1)*128   # MHz
    obsnchan_new = nchan_new

    # create new primary HDU and update parameters
    primary_hdu = fits.PrimaryHDU(header=hdu_primary_header)
    # update OBSFREQ, OBSBW and OBSNCHAN
    primary_hdu.header.set('OBSFREQ', obsfreq_new)
    primary_hdu.header.set('OBSBW', obsbw_new)
    primary_hdu.header.set('OBSNCHAN', obsnchan_new)

    # fix other problems
    primary_hdu.header.set('OBS_MODE', 'SEARCH')
    #primary_hdu.header.set('CHAN_DM', dm)        # required by PRESTO
    primary_hdu.header.set('TRK_MODE', 'TRACK')  # required by PRESTO

    stt_crd1 = primary_hdu.header['STT_CRD1']
    stt_crd2 = primary_hdu.header['STT_CRD2']

    # Online cycle time
    primary_hdu.header.set('NRCVR', 1)

    # Scan length
    #hdulist[0].header.set('SCANLEN', 60.)

    # Online cycle time
    #hdulist[0].header.set('TCYCLE', 10.)

    primary_hdu.header.set('STP_CRD1', stt_crd1)
    primary_hdu.header.set('STP_CRD2', stt_crd2)
    primary_hdu.header.set('CAL_MODE', 'SYNC')
    primary_hdu.header.set('CAL_FREQ', 11.123)
    primary_hdu.header.set('CAL_DCYC', 0.)
    primary_hdu.header.set('CAL_PHS', 0.25)

    # Start LST, second
    #hdulist[0].header.set('STT_LST', 10.)

    # create new subint HDU and update parameters
    cols = fits.ColDefs([col_index, col_tsubint, col_offssub, col_auxdm, col_auxrm, col_freq, col_wts, col_scl, col_off, col_data])
    hdu_subint = fits.BinTableHDU.from_columns(cols, header=hdu_subint_header)
    #print hdu_subint.columns

    # update NCHAN, NSTOT, and REFFREQ
    hdu_subint.header.set('NCHAN', obsnchan_new)
    hdu_subint.header.set('REFFREQ', obsfreq_new)
    hdu_subint.header.set('NSTOT', nstot_new)

    # fix other problems
    # Nr of bins/pulse period (for gated data)
    hdu_subint.header.set('NBIN_PRD', 1)
    hdu_subint.header.set('NCHNOFFS', 0)   # required by PRESTO

    # Phase offset of bin 0 for gated data
    hdu_subint.header.set('PHS_OFFS', 0.)
    if npol == 4:
        hdu_subint.header.set('POL_TYPE', 'AABBCRCI')

    # create new HDU list and write to a new fits
    hdul = fits.HDUList([primary_hdu, hdu_subint])
    logger.info(f"new hdulist info: {hdul.info()}")

    logger.info(f"writing hdulist to {out_file}...")
    hdul.writeto(out_file, overwrite=True, checksum=True)

    hdulist.close()


### make it for multi-processing run...
def multi_proc(logger, in_files, sub_start, sub_end, ncpu=4):
    args = []
    for in_file in in_files:
        data_folder = "/".join(in_file.split("/")[:-1])
        in_file_name = in_file.split("/")[-1]
        out_file = f"{data_folder}/{in_file_name}.sub{sub_start}_{sub_end}.fits"

        if os.path.exists(out_file):
            logger.info(f"{in_file_name} already done... aborted...")

        args.append([logger, in_file, out_file, sub_start, sub_end])

    logger.info(f"start to run splitting on {ncpu} cpus...")
    pool = Pool(ncpu)
    pool.starmap(split_uwl_data, args)

if __name__ == "__main__":
    logger = _init_logger()

    parser = argparse.ArgumentParser(description='split Parkes UWL search mode data...')
    parser.add_argument('-f',  '--data_folder',  metavar='Input folder name',  nargs='+', required=True, help='Folder that stores .sf data')
    parser.add_argument('-sub0',  '--subband_start', metavar='Starting subband', nargs='+', required=True, help='Starting subband, from 0-25')
    parser.add_argument('-sub1',  '--subband_end', metavar='Ending subband', nargs='+', required=True, help='Ending subband, from 0-25')
    parser.add_argument('-ncpu', '--ncpu', metavar='number of cpu to be used', nargs='+', default="2", help="number of cpu to be used")

    args = parser.parse_args()
    data_folder = args.data_folder[0]
    sub_start = int(args.subband_start[0])
    sub_end = int(args.subband_end[0])
    ncpu = int(args.ncpu[0])

    in_files = glob.glob(f"{data_folder}/*.sf")
    logger.info(f"{len(in_files)} Parkes search mode data files found...")
    multi_proc(logger, in_files, sub_start, sub_end)








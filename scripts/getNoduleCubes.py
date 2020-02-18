import numpy as np
import copy
from matplotlib import pyplot as plt
from utils import readMhd, readCsv, getImgWorldTransfMats, convertToImgCoord, extractCube
from readNoduleList import nodEqDiam

dispFlag = False

# Read nodules csv
csvlines = readCsv('/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/trainset_csv/trainNodules_gt.csv')
header = csvlines[0]
nodules = csvlines[1:]

lndloaded = -1
for n in nodules:
        vol = float(n[header.index('Volume')])
        if nodEqDiam(vol)>3: #only get nodule cubes for nodules>3mm
                ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
                lnd = int(n[header.index('LNDbID')])
                rads = list(map(int,list(n[header.index('RadID')].split(','))))
                radfindings = list(map(int,list(n[header.index('RadFindingID')].split(','))))
                finding = int(n[header.index('FindingID')])
                
                print(lnd,finding,rads,radfindings)
                        
                # Read scan
                if lnd!=lndloaded:
                        [scan,spacing,origin,transfmat] =  readMhd('/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/data/LNDb-{:04}.mhd'.format(lnd))                
                        transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
                        lndloaded = lnd
                
                # Convert coordinates to image
                ctr = convertToImgCoord(ctr,origin,transfmat_toimg)                
                
                for rad,radfinding in zip(rads[:1],radfindings[:1]):
                        # Read segmentation mask
                        [mask,_,_,_] =  readMhd('/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/masks/LNDb-{:04}_rad{}.mhd'.format(lnd,rad))
                        
                        # Extract cube around nodule
                        scan_cube = extractCube(scan,spacing,ctr)
                        masknod = copy.copy(mask)
                        masknod[masknod!=radfinding] = 0
                        masknod[masknod>0] = 1
                        mask_cube = extractCube(masknod,spacing,ctr)
                        
                        # Display mid slices from resampled scan/mask
                        if dispFlag:
                                fig, axs = plt.subplots(2,3)
                                axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:])
                                axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])
                                axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:])
                                axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])
                                axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)])
                                axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])    
                                plt.show()
                        
                        # Save mask cubes
                        np.save('/media/tungthanhlee/SSD/grand_challenge/dataset/LNDB_segmentation/split/scan_cubes/LNDb-{:04d}_finding{}.npy'.format(lnd,finding,rad),scan_cube)                
                
                
#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images
usage:
    calibrate.py [--out <output path>] [--square_size] [<image mask>]
default values:
    --out:    .sample/output/
    --square_size: 1.0
    <image mask> defaults to .sample/chessboard/*.jpg


Code forked from OpenCV:
https://github.com/opencv/opencv/blob/a8e2922467bb5bb8809fc8ae6b7233893c6db917/samples/python/calibrate.py
released under BSD 3 license
'''

# Python 2/3 compatibility
from __future__ import print_function

# local modules
from common import splitfn

# built-in modules
import os
import sys
from glob import glob
import numpy as np
import cv2
import logging
import argparse
import glob
import numpy as np

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate camera matrix and '
                                     'distortion parameters from chessboard '
                                     'images')
    parser.add_argument('--chessboard', help='path to images', default = os.path.join('sample', 'chessboard'))
    parser.add_argument('--input', help='path to images', default = os.path.join('sample', 'images'))
    parser.add_argument('--pattern_x', metavar='X', default=10, type=int,
                        help='pattern x')
    parser.add_argument('--pattern_y', metavar='Y', default=7, type=int,
                        help='pattern y')
    parser.add_argument('--out', help='optional path for output', default='undistorted')
    parser.add_argument('--square_size', default=1.0)

    args=parser.parse_args()

    logging.debug(args)

    # get images into a list
    extensions = ['*jpg', '*JPG', '*jpeg', '*png']

    img_to_undist = []
    # images to undistort
    if os.path.isdir(args.input):
        for ext in extensions:
            img_to_undist += glob.glob(os.path.join(args.input, ext))
            print(img_to_undist)

    # chessboard images
    img_names = []
    if os.path.isdir(args.chessboard):
        for ext in extensions:
            img_names += glob.glob(os.path.join(args.chessboard, ext))
        proj_root = args.chessboard # os.path.join(os.path.join(args.chessboard, '..'))
    else: 
        logging.error("%s is not a directory" % args.chessboard)
        exit()

    if not args.out:
        out = os.path.join(proj_root, 'out')
    else:
        out = args.out
        
    if not os.path.isdir(out):
        os.mkdir(out)

    square_size = float(args.square_size)

    pattern_size = (args.pattern_x -1, args.pattern_y - 1) # For some reason you have to subtract 1
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    print('img: ', img_names)
    for fn in img_names:
        print('processing %s... ' % fn, end='')
        #img = cv2.imread(os.path.join(proj_root, fn), 0)
        img = cv2.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            continue

        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            print("ok...")
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if out:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            outfile = os.path.join(out, name + '_chess.png')
            cv2.imwrite(outfile, vis)
            if found:
                pass
                #img_names_undistort.append(outfile)

        if not found:
            print('chessboard not found')
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        import pickle as pkl
        with open('img.pkl', 'wb') as f:
            pkl.dump(img_points, f)
        with open('obj.pkl', 'wb') as f:
            pkl.dump(obj_points, f)

        print('ok')

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    # print("matrix: \n", type(camera_matrix))
    print("distortion coefficients: ", dist_coefs.ravel())

    # write to matrix to be used as input
    with open(os.path.join(out, "matrix.txt"), "w") as matf:
        camera_matrix.reshape((3, 3))
        np.savetxt(matf, (camera_matrix[0], camera_matrix[1], camera_matrix[2]), fmt='%-12.8f')

    with open(os.path.join(out, "distortion.txt"), "w") as distf:
        np.savetxt(distf, dist_coefs.ravel(), fmt='%.12f')

    # undistort the image with the calibration
    for img_found in img_to_undist:
        img = cv2.imread(img_found)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w] # only extracts a tiny rectangle of the undistorted image
        path, name, ext = splitfn(img_found)
        outfile = os.path.join(path, name + '_undistorted.png')
        cv2.imwrite(outfile, dst)
        remove_artifacts(dst)
        print('Undistorted image written to: %s' % outfile)

    cv2.destroyAllWindows()

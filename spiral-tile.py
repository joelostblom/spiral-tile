# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:32:07 2016

@author: joel
"""

import datetime
import argparse
import logging
import shutil
import re
import os

from skimage import exposure
from itertools import cycle
import skimage.io as io
from glob import iglob
from PIL import Image # Change to pillow?
import numpy as np
# `shutil.move` handles some edgecase in `os.rename`
#http://pythoncentral.io/how-to-rename-move-a-file-in-python/

# Ideas
'''
- For comparing the same channel between wells. The best is probably a plate overview image
  where all the tiled well images are stitched together in the structure of the plate (96 only for now) bonus feature
'''
def nat_key(key):
    '''
    A key to use with the `sorted()` function to sort naturally.
    '''
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', key)]

def main():
    parser = argparse.ArgumentParser(description='A small utility for field images '
        'exported from automated microscope platforms.\nStarts in the middle and '
        'stitches fields in a spiral pattern to create the well image.\n'
        'Each well and channel need to be in a separate directory. Use `-c` to sort '
        'images into\ndirectories automatically. Make sure to specify the correct field '
        'and well string (-f, -w).\nExample usage when the images from all wells are in the '
        'same directory:\n\npython stitch_fields.py -cr -f <field_prefix> -w <well_prefix>',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('path', default='./', nargs='?',
        help='path to images  (default: current directory)')
    parser.add_argument('-i', '--input-format', default='tiff',
        help='format for images to be stitched, can also be a list of formats (default: %(default)s)')
    parser.add_argument('-o', '--output-format', default='jpeg',
        help='format for the stitched image (default: %(default)s)')
    parser.add_argument('-w', '--well-prefix', default='001_',
        help='string immediately preceding the well id in the file name (default: %(default)s)')
    parser.add_argument('-c', '--channel-prefix', default='d',
        help='string immediately preceding the channel id in the file name (default: %(default)s)')
    parser.add_argument('-f', '--field-prefix', default='f',
        help='string immediately preceding the field number in the file name (default: %(default)s)')
    parser.add_argument('--flip', default='vertical', choices=['horizontal', 'vertical', 'both', 'none'], help='How to flip the image (default: %(default)s)')
    parser.add_argument('--cutoff', default=99.9, help='Saturate intensities above this percentile. Prevents outliers from making images too dim and allows for intensity comparisons across wells. (default: %(default)s)')
    parser.add_argument('--scan-direction', default='left_down', choices=[
        'left_down', 'down_left', 'left_up', 'up_left', 'right_down',
        'down_right', 'right_up', 'up_right'], help='The directions from the 1st field to the 2nd and 3rd (default: %(default)s)\n'
        '9 8 7\n'
        '2 1 6\n'
        '3 4 5')
    # Initialize some variables
    args = parser.parse_args()
    # PIL's image function takes 'jpeg' instead of 'jpg' as an argument, But I
    # still prefer to write stitched image as `jpg`.
    if args.output_format.lower() == 'jpeg':
        output_format = 'jpg'
    else:
        output_format = args.output_format.lower()
    input_format = args.input_format.lower()
    logging.basicConfig(filename='well_stitch.log', level=logging.DEBUG, format='%(message)s')
    # Print out the runtime parameters and store them in a log file
    for key in ['path', 'input_format', 'output_format', 'well_prefix', 'channel_prefix', 'field_prefix', 'scan_direction', 'flip', 'cutoff']: #sorted(vars(args)): # Returns a dictionary instead of a Namespace object
        print('{: <20}{}'.format(key, vars(args)[key]))

    # Create a new timestamped directory.
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S')
    stitched_dir = os.path.join(args.path, 'stitched-well-images-{}'.format(timestamp))
    os.makedirs(stitched_dir)
    logging.info('Created directory ' + os.path.join(stitched_dir))
    # Loop through only the well subdirectories, the current directory does not need to be
    # included as the files will already be sorted into subdirectories
    sort_wells_and_channels(args.path, args.well_prefix, args.channel_prefix, input_format)
        
    dirs = [name for name in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, name))]
    well_dirs = [name for name in dirs if not name.startswith('stitched-wells')]

    well_dirs = os.listdir(os.path.join(args.path, 'sorted-well-images'))
    channels = os.listdir(os.path.join(args.path, 'sorted-well-images', well_dirs[0]))
    print('')
    # Compute the percentile cutoffs from all images within a channel
    cutoffs = {}
    for channel in sorted(channels):
        dye_img_names = [fname for fname in iglob('{}/**/*{}.{}'.format(
            os.path.join(args.path, 'sorted-well-images'), channel,
            input_format.upper()), recursive=True)]
        # Get image dimensions to preallocate array correctly
        img_res = io.imread(dye_img_names[0]).shape
        dye_imgs = np.ndarray((len(dye_img_names), img_res[0], img_res[1]), dtype='uint16')
        for num, img_name in enumerate(dye_img_names):
            dye_imgs[num, :, :] = io.imread(img_name)
        # Show some percentiles to give an idea of what's a suitable choice
        percentiles = [99, 99.9, 99.99, 99.999]
        dye_percentile_values = np.percentile(dye_imgs, percentiles)
#         dye_percentiles = {x:y for x, y in zip(percentiles, dye_percentile_values)}
        # Currently the same percentile cutoff is applied to all channels
        cutoffs[channel] = np.percentile(dye_imgs, args.cutoff)
        print('Percentiles channel {}:'.format(channel))
        print("\t".join([str(perc) for perc in percentiles]))
        print("\t".join([str(int(perc)) for perc in dye_percentile_values.tolist()]))
        print('Selected cutoff is {}'.format(cutoffs[channel]))
        print('---------------------')

    # Process images well by well
    print('')
    for num, dir_name in enumerate(sorted(well_dirs, key=nat_key), start=1):
        print(dir_name)
        dir_name = os.path.join(args.path, dir_name)
        channel_dirs = [os.path.join(args.path, dir_name, name) for name in os.listdir(dir_name) if os.path.isdir(os.path.join(args.path, dir_name, name))]
        for channel_dir in channel_dirs:
#            os.makedirs(stitched_dir_channel)
            print(os.path.basename(channel_dir))
            imgs, zeroth_field, max_ints = find_images(channel_dir, input_format, args.flip, args.field_prefix)
            fields, arr_dim, moves, starting_point = spiral_structure(channel_dir, input_format, args.scan_direction)
            img_layout = spiral_array(fields, arr_dim, moves, starting_point, zeroth_field)
            stitched_well = stitch_images(imgs, img_layout, channel_dir, args.output_format, arr_dim, stitched_dir)
            stitched_channel_name = os.path.join(stitched_dir, '{}-{}.{}'.format(os.path.basename(channel_dir), os.path.basename(dir_name), output_format))
            # TODO Add if statement that does not rescale if the output is 16-bit
            # tiff to use for running the stitched well through cellprofiler.
            # Rescale to 8bit range (0-255)
            rescaled_stitched_well = exposure.rescale_intensity(
                np.array(stitched_well), in_range=(0, cutoffs[os.path.basename(channel_dir)]), out_range=(0, 2**8 -1))
            # Intensities are rescaled to an 8bit range, but the image is still
            # in 16-bit format, so would be all black if displayed
            rescaled_stitched_well = Image.fromarray(rescaled_stitched_well.astype('uint8'))
            rescaled_stitched_well.save(stitched_channel_name, format=args.output_format)


#data = numpy.random.rand(5, 301, 219)
#imsave('temp.tif', data)
#
#from libtiff import TIFF
#tiff = TIFF.open('libtiff.tiff', mode='w')
#tiff.write_image(ar)
#tiff.close()
#from skimage import exposure, img_as_ubyte
#image = exposure.rescale_intensity(np.array(stitched_well), in_range=(0, np.percentile(np.array(stitched_well), 99.95)))
#Image.fromarray(image).save('testing', format='tiff')

#image = exposure.rescale_intensity(np.array(im), in_range=(0, np.percentile(np.array(im), 99.95)))
#Image.fromarray(img_as_ubyte(image)).save('testing', format='tiff') 
#                piral_tile(well, channel)
#                stitch_images(imgs, img_layout, dir_path, output_format, arr_dim, stiched_dir):
#            fields, arr_dim, moves, starting_point = spiral_structure(dir_name, input_format, args.scan_direction)
#            img_layout = spiral_array(fields, arr_dim, moves, starting_point, zeroth_field)
#            stitched_well = stitch_images(imgs, img_layout, dir_name, args.output_format, arr_dim, stitched_dir)
#            stitched_well_name = os.path.join(stitched_dir, os.path.basename(dir_name) + '.' + output_format)
#            stitched_well.save(stitched_well_name, format=args.output_format)
#            logging.info('Stitched image saved to ' + stitched_well_name + '\n')
        else:
            logging.info('No images found in this directory\n')
    os.rename('./well_stitch.log', os.path.join(stitched_dir, 'well_stitch.log'))

    print('\n\nStitched well images can be found in ' + stitched_dir + '.\nPlease check the log file for which images and what field layout were used to create the stitched image.\nDone.')


def sort_wells_and_channels(dir_path, well_prefix, channel_prefix, input_format):
    '''
    Sort images into wells and channels simultaneously to avoid looping through the files twice.
    If the images are already sorted into folders, this will not be executed, since the if-
    expression will not match any of the folders, just images.
    '''
    well_names = []
    channel_names = []
    for fname in os.listdir(dir_path):
        # Check if there are images in this folder
        # Need to index [1:] since the extension includes the 'dot', e.g. '.tif'
        if os.path.splitext(fname)[1][1:].lower() == input_format:
            # Create well subfolders
            #find the well id using the provided prefix and add it to the list
            well_ind = fname.index(well_prefix) + len(well_prefix)
            well_name = [fname[well_ind]]
            well_name.append([str(int(char)) for char in fname[well_ind+1:well_ind+3] if char.isdigit()])
            well_name = ''.join([char for sublist in well_name for char in sublist])
            well_names.append(well_name)
            # Create the well directory if it doesn't exist already
            if not os.path.exists(os.path.join(dir_path, well_name)):
                os.makedirs(os.path.join(dir_path, well_name))
            # Create channel subfolders
            channel_ind = fname.index(channel_prefix) + len(channel_prefix)
            channel_name = [fname[channel_ind]]
            channel_name.append([str(int(char)) for char in fname[channel_ind+1:channel_ind+3] if char.isdigit()])
            channel_name = ''.join([char for sublist in channel_name for char in sublist])
            channel_names.append(channel_name)
            if not os.path.exists(os.path.join(dir_path, well_name, channel_name)):
                os.makedirs(os.path.join(dir_path, well_name, channel_name))
#            logging.info('moving ./' + fname + ' to ./' + os.path.join(well_name, fname))
            # Move images to their respective subfolder
            shutil.move(os.path.join(dir_path, fname), os.path.join(dir_path,
                'sorted-well-images', well_name, channel_name, fname))
#    logging.info('created well directories ' + str(set(well_names)))

    return None


# Define movement function for filling in the spiral array
def move_right(x, y):
    return x, y + 1


def move_down(x, y):
    return x + 1, y


def move_left(x, y):
    return x, y - 1


def move_up(x, y):
    return x - 1, y


def spiral_structure(dir_path, input_format, scan_direction):
    '''
    Define the movement scheme and starting point for the field layout
    '''
    #find the number of fields/images matching the specified extension(s)
    fields = len([fname for fname in os.listdir(dir_path) if os.path.splitext(fname)[1][1:].lower() in input_format])
    #size the array based on the field number, array will be squared
    arr_dim = int(np.ceil(np.sqrt(fields)))
    #define the movement schema and find the starting point (middle) of the array
    if scan_direction == 'down_left':
        moves = [move_down, move_left, move_up, move_right]
        if arr_dim % 2 != 0:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
        else:
            starting_point = (int(arr_dim/2)-1, int(arr_dim/2))
    elif scan_direction == 'left_down':
        moves = [move_left, move_down, move_right, move_up]
        if arr_dim % 2 != 0:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
        else:
            starting_point = (int(arr_dim/2)-1, int(arr_dim/2))
    elif scan_direction == 'down_right':
        moves = [move_down, move_right, move_up, move_left]
        if arr_dim % 2 != 0:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
        else:
            starting_point = (int(arr_dim/2)-1, int(arr_dim/2)-1)
    elif scan_direction == 'right_down':
        moves = [move_right, move_down, move_left, move_up]
        if arr_dim % 2 != 0:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
        else:
            starting_point = (int(arr_dim/2)-1, int(arr_dim/2)-1)
    elif scan_direction == 'up_left':
        moves = [move_up, move_left, move_down, move_right]
        if arr_dim % 2 != 0:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
        else:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
    elif scan_direction == 'left_up':
        moves = [move_left, move_up, move_right, move_down]
        if arr_dim % 2 != 0:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
        else:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
    elif scan_direction == 'up_right':
        moves = [move_up, move_right, move_down, move_left]
        if arr_dim % 2 != 0:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
        else:
            starting_point = (int(arr_dim/2), int(arr_dim/2)-1)
    elif scan_direction == 'right_up':
        moves = [move_right, move_up, move_left, move_down]
        if arr_dim % 2 != 0:
            starting_point = (int(arr_dim/2), int(arr_dim/2))
        else:
            starting_point = (int(arr_dim/2), int(arr_dim/2)-1)

    return fields, arr_dim, moves, starting_point


def gen_points(end, moves, starting_point):
    '''
    Generate coordinates for each field number. From stackoverflow
    '''
    _moves = cycle(moves)
    n = 1
    pos = starting_point
    times_to_move = 1

    yield n,pos

    while True:
        for _ in range(2):
            move = next(_moves)
            for _ in range(times_to_move):
                if n >= end:
                    return None
                pos = move(*pos)
                n+=1
                yield n,pos

        times_to_move+=1


def spiral_array(fields, arr_dim, moves, starting_point, zeroth_field):
    '''
    Fill the array in the given direction
    '''
    #create an array of zeros and then fill with a number not used for any field
    img_layout = np.zeros((arr_dim, arr_dim), dtype=int)
    img_layout[:] = -1 #TODO this means that the zeroth field will be put in multiple places... fixed?
    #create a different layout depending on the numbering of the first field
    if zeroth_field:
        for point, coord in list(gen_points(fields, moves, starting_point)):
            img_layout[coord] = point -1
        logging.info('\nField layout:')
        for row in np.ma.masked_equal(img_layout, -1): #TODO fix so that lines are showing for unused field
            logging.info(' '.join(['{:>2}'.format(str(i)) for i in row]))

    else:
        for point, coord in list(gen_points(fields, moves, starting_point)):
            img_layout[coord] = point
        logging.info('Field layout:')
        for row in np.ma.masked_equal(img_layout, -1):
            logging.info(' '.join(['{:>2}'.format(str(i)) for i in row]))

    return img_layout


def find_images(dir_path, input_format, flip, field_str):
    '''
    Create a dictionary with the field numbers as keys to the field images
    '''
    zeroth_field = False #changes if a zeroth field is found in 'find_images'
    imgs = {}
    max_ints = []
    logging.info('----------------------------------------------')
    logging.info(dir_path)
    #go through each directory
    for fname in os.listdir(dir_path):
        if os.path.splitext(fname)[1][1:].lower() in input_format:
            logging.info(fname)
            #find the index of the field identifier string in the file name
            field_ind = fname.index(field_str) + len(field_str)
            fnum = int(''.join([str(int(char)) for char in fname[field_ind:field_ind+2] if char.isdigit()]))
            # If field 0 is encountered, change start numbering of the array
            if fnum == 0:
                zeroth_field = True
            # The default is to flip horizontally since this is the most common case
            if flip == 'none':
                imgs[fnum] = Image.open(os.path.join(dir_path, fname))
            elif flip == 'horizontal':
                imgs[fnum] = Image.open(os.path.join(dir_path, fname)).transpose(Image.FLIP_LEFT_RIGHT)
            elif flip == 'vertical':
                # dunno why we need to flip...
                imgs[fnum] = Image.open(os.path.join(dir_path, fname)).transpose(Image.FLIP_TOP_BOTTOM)
            elif flip == 'both':
                # I don't think thei sould ever be the case, it could just be adjusted with another
                # spiral rotation, but putting it here for completion
                imgs[fnum] = Image.open(os.path.join(dir_path, fname)).transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
            # Collect max intensities here instead of looping through an extra time
            max_ints.append(np.percentile(np.array(imgs[fnum]).ravel(), 99.999)) # make this a user variable
         #   min_ints.append(np.percentile(np.array(imgs[fnum]).ravel(), 0.11)) # make this a user variable
#            if rescale_intensity:
#               img_stretched = exposure.rescale_intensity(np.array(imgs[fnum]), in_range='uint12', out_range=('uint8'))
#               imgs[fnum] = Image.fromarray(np.uint8(img_stretched))
#    print(max_ints) # I still need to figure out what is going on in terms of jpg and png. tiff seems to be working as of 2017/04/02.
    if max_ints != []:
       # max_ints = max(max_ints) # the highest intensity in the entire plate
        max_ints = np.percentile(np.array(max_ints), 70) # the highest intensity in the entire plate
        #print(max_ints)
    return imgs, zeroth_field, max_ints





    return imgs, zeroth_field

def stitch_images(imgs, img_layout, dir_path, output_format, arr_dim, stiched_dir):
    '''
    Stitch images by going row and column wise in the img_layout and look up
    the number of the image to place at the (row, col) coordinate. So not filling
    in a spiral but using the spiral lookuptable instead.
    '''
    # Create the size of the well image to be filled in
    width, height = imgs[1].size
    num = 0
    stitched_well = Image.new('I;16', (width*arr_dim, height*arr_dim))
    for row in range(0, width*arr_dim, width):
        for col in range(0, height*arr_dim, height):
            #since the image is filled by row and col instead of sprial, this
            #error catching is needed for the empty places
            try:
                #'stitch' fields by pasting them at the appropriate place in the black background
                stitched_well.paste(imgs[img_layout.ravel()[num]], (col, row))
            except KeyError:
                pass
            num += 1
    #save image
#    stitched_name = os.path.join(dir_path, 'stitched_wells/stitched_' + timestamp + '.' + output_format)
#    stitched.save(stitched_name, format=output_format)
    # stitched.show()

    return stitched_well

# run main
if __name__ == '__main__':
    main()

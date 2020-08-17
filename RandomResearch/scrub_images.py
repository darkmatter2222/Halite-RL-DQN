# remove all images that don't meet strict gameplay
import glob
from PIL import Image
import numpy as np
import shutil
import matplotlib.pyplot as plt

# y,x format index is top left
root_image_dir = 'N:\\Halite'
train_dir = f'{root_image_dir}\\TRAIN'
review_dir = f'{root_image_dir}\\Review'
north_source_dir = f'{train_dir}\\NORTH'
north_review_dir = f'{review_dir}\\NORTH'
east_source_dir = f'{train_dir}\\EAST'
east_review_dir = f'{review_dir}\\EAST'
south_source_dir = f'{train_dir}\\SOUTH'
south_review_dir = f'{review_dir}\\SOUTH'
west_source_dir = f'{train_dir}\\WEST'
west_review_dir = f'{review_dir}\\WEST'
nothing_source_dir = f'{train_dir}\\NOTHING'
nothing_review_dir = f'{review_dir}\\NOTHING'
trash_review_dir = f'{review_dir}\\TRASH'

# plt.imshow(img)
# plt.show()
# print('bad')

files_scrubbed = 0


# Scrub out the Bad Moves that cause death
# NORTH
for file in glob.glob(f"{north_source_dir}\\*.png"):
    img = Image.open(file)
    img_array = np.array(img)
    center = img_array[5][4]
    center_n_1 = img_array[4][4]
    center_n_2 = img_array[3][4]
    center_n_1_e_1 = img_array[4][5]
    center_n_1_w_1 = img_array[4][3]
    if center_n_1[2] == 255 or center_n_2[2] == 255 or center_n_1_e_1[2] == 255 or center_n_1_w_1[2] == 255:
        shutil.move(file, north_review_dir)
        files_scrubbed += 1

# EAST
for file in glob.glob(f"{east_source_dir}\\*.png"):
    img = Image.open(file)
    img_array = np.array(img)
    center = img_array[5][4]
    center_e_1 = img_array[5][5]
    center_e_2 = img_array[5][6]
    center_e_1_n_1 = img_array[4][5]
    center_e_1_s_1 = img_array[6][5]
    if center_e_1[2] == 255 or center_e_2[2] == 255 or center_e_1_n_1[2] == 255 or center_e_1_s_1[2] == 255:
        shutil.move(file, east_review_dir)
        files_scrubbed += 1

# SOUTH
for file in glob.glob(f"{south_source_dir}\\*.png"):
    img = Image.open(file)
    img_array = np.array(img)
    center = img_array[5][4]
    center_s_1 = img_array[6][4]
    center_s_2 = img_array[7][4]
    center_s_1_e_1 = img_array[6][5]
    center_s_1_w_1 = img_array[6][3]
    if center_s_1[2] == 255 or center_s_2[2] == 255 or center_s_1_e_1[2] == 255 or center_s_1_w_1[2] == 255:
        shutil.move(file, south_review_dir)
        files_scrubbed += 1

# WEST
for file in glob.glob(f"{west_source_dir}\\*.png"):
    img = Image.open(file)
    img_array = np.array(img)
    center = img_array[5][4]
    center_w_1 = img_array[5][3]
    center_w_2 = img_array[5][2]
    center_w_1_n_1 = img_array[4][3]
    center_w_1_s_1 = img_array[6][3]
    if center_w_1[2] == 255 or center_w_2[2] == 255 or center_w_1_n_1[2] == 255 or center_w_1_s_1[2] == 255:
        shutil.move(file, west_review_dir)
        files_scrubbed += 1

# NOTHING
for file in glob.glob(f"{nothing_source_dir}\\*.png"):
    img = Image.open(file)
    img_array = np.array(img)
    center = img_array[5][4]
    center_n_1 = img_array[4][4]
    center_e_1 = img_array[5][5]
    center_s_1 = img_array[6][4]
    center_w_1 = img_array[5][3]
    if center_n_1[2] == 255 or center_e_1[2] == 255 or center_s_1[2] == 255 or center_w_1[2] == 255:
        shutil.move(file, nothing_review_dir)
        files_scrubbed += 1

# Scrub out the pointless stay stills...
for file in glob.glob(f"{nothing_source_dir}\\*.png"):
    img = Image.open(file)
    img_array = np.array(img)
    center = img_array[5][4]
    if center[1] == 0:
        shutil.move(file, trash_review_dir)
        files_scrubbed += 1

# Remove Duplicates Between classes:
master_image_dict = {
    'NORTH': [],
    'EAST': [],
    'SOUTH': [],
    'WEST': [],
    'NOTHING': []
}

for file in glob.glob(f"{north_source_dir}\\*.png", recursive=True):
    img = Image.open(file)
    img_array = np.array(img)
    master_image_dict['NORTH'].append({'image': img_array.tolist(), 'directory': file, 'remove_record': False})

for file in glob.glob(f"{east_source_dir}\\*.png", recursive=True):
    img = Image.open(file)
    img_array = np.array(img)
    master_image_dict['EAST'].append({'image': img_array.tolist(), 'directory': file, 'remove_record': False})

for file in glob.glob(f"{south_source_dir}\\*.png", recursive=True):
    img = Image.open(file)
    img_array = np.array(img)
    master_image_dict['SOUTH'].append({'image': img_array.tolist(), 'directory': file, 'remove_record': False})

for file in glob.glob(f"{west_source_dir}\\*.png", recursive=True):
    img = Image.open(file)
    img_array = np.array(img)
    master_image_dict['WEST'].append({'image': img_array.tolist(), 'directory': file, 'remove_record': False})

for file in glob.glob(f"{nothing_source_dir}\\*.png", recursive=True):
    img = Image.open(file)
    img_array = np.array(img)
    master_image_dict['NOTHING'].append({'image': img_array.tolist(), 'directory': file, 'remove_record': False})

# priority Nothing, North, East, South, West

def search(this_image, this_list):
    for searching_image_record in this_list:
        if this_image == searching_image_record['image']:
            searching_image_record['remove_record'] = True

print('Stage 2')
for this_record in master_image_dict['NOTHING']:
    search(this_record['image'], master_image_dict['NORTH'])
    search(this_record['image'], master_image_dict['EAST'])
    search(this_record['image'], master_image_dict['SOUTH'])
    search(this_record['image'], master_image_dict['WEST'])
print('Stage 2.1')
for this_record in master_image_dict['NORTH']:
    search(this_record['image'], master_image_dict['EAST'])
    search(this_record['image'], master_image_dict['SOUTH'])
    search(this_record['image'], master_image_dict['WEST'])
    search(this_record['image'], master_image_dict['NOTHING'])
print('Stage 2.2')
for this_record in master_image_dict['EAST']:
    search(this_record['image'], master_image_dict['SOUTH'])
    search(this_record['image'], master_image_dict['WEST'])
    search(this_record['image'], master_image_dict['NOTHING'])
    search(this_record['image'], master_image_dict['NORTH'])
print('Stage 2.3')
for this_record in master_image_dict['SOUTH']:
    search(this_record['image'], master_image_dict['WEST'])
    search(this_record['image'], master_image_dict['NOTHING'])
    search(this_record['image'], master_image_dict['NORTH'])
    search(this_record['image'], master_image_dict['EAST'])
print('Stage 2.4')
for this_record in master_image_dict['WEST']:
    search(this_record['image'], master_image_dict['NOTHING'])
    search(this_record['image'], master_image_dict['NORTH'])
    search(this_record['image'], master_image_dict['EAST'])
    search(this_record['image'], master_image_dict['SOUTH'])
print('Stage 2 Complete')

for this_record_name in master_image_dict:
    for this_record in master_image_dict[this_record_name]:
        if this_record['remove_record'] == True:
            print('delete')

print(f'{files_scrubbed} files scrubbed')



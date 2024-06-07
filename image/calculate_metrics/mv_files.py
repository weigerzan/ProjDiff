import os
import shutil

source_path = 'exp/image_samples/proxdiff_100steps'
output_path = 'exp/proxdiff_100steps'
os.makedirs(os.sep.join([output_path, 'orig']), exist_ok=True)
os.makedirs(os.sep.join([output_path, 'generated']), exist_ok=True)
n1 = n2 = 0
for filename in os.listdir(source_path):
    if '-1' in filename:
        shutil.copyfile(os.sep.join([source_path, filename]), os.sep.join([output_path, 'generated', filename]))
    elif 'orig' in filename:
        shutil.copyfile(os.sep.join([source_path, filename]), os.sep.join([output_path, 'orig', filename]))
    else:
        pass

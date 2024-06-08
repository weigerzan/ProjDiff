import torchaudio
import os
import resampy
import soundfile as sf
import tqdm

resample_rate = 16000

idx = 0
work_path = 'output/partial_generating/BDG/projdiff/sum/' # /path/to/your/generated/folder/sum
save_path = '/path/to/the/save/folder'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
os.makedirs(os.sep.join([save_path, 'gt']), exist_ok=True)
os.makedirs(os.sep.join([save_path, 'generated']), exist_ok=True)
for dirname in tqdm.tqdm(os.listdir(work_path)):
    onepath = os.sep.join([work_path, dirname])
    mixture, sample_rate = sf.read(os.sep.join([onepath, 'mixture.wav']), dtype="float32")
    sf.write(os.sep.join([save_path, 'generated','{}.wav'.format(idx)]), mixture, sample_rate)

    onepath = os.sep.join([work_path, dirname])
    mixture, sample_rate = sf.read(os.sep.join([onepath, 'gt_mixture.wav']), dtype="float32")
    sf.write(os.sep.join([save_path, 'gt', '{}.wav'.format(idx)]), mixture, sample_rate)
    idx += 1
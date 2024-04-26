"""This script creates an augmented copy of a sound files
dataset. It mixes each audio file with a random noise audio file

Args:
    data_dir: path to the directory with sound files (WAV)
        the directory at path data_dir should have the following structure:
        data_dir_name
        |
        |---profile_name_1
           |
           |---file_1.wav
           |---file_2.wav
           |--- ...
        |---profile_name_2
        |--- ...
    noises_dir: path to the directory with noises files (MP3)
    output_dir: path to the directory to put output files in
    --noise_scale: the target ratio between the noise max amplitude and the sound max amplitude,
        default: 0.5

Example:
    python augment_dataset.py split_data/train noises/ output/ --noise_scale 0.5

Notes:
    each provided directory as an argument must exist
    files in data_dir are assumed to be WAV files
    files in noise_dir are assumed to be MP3 files

"""

import argparse
import os
import random
from pydub import AudioSegment
from pydub.utils import ratio_to_db
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("data_dir")
parser.add_argument("noises_dir")
parser.add_argument("output_dir")
parser.add_argument("--noise_scale", default=0.5)

args = parser.parse_args()

for path in (args.data_dir, args.noises_dir, args.output_dir):
    assert os.path.isdir(path), f"Invalid path: {path}"


def mix_sound_and_noise(
    sound: AudioSegment,
    noise: AudioSegment,
    noise_scale: float
) -> AudioSegment:
    """Mixes the sound audio and the noise audio and creates
    another audio, with sound and noise playing simultaneously

    Args:
        sound: AudioSegment object with the sound
        noise: AudioSegment object with the noise to be mixed in
        noise_scale: the target ratio between the noise max amplitude
            and the sound max amplitude
    """
    ampl_ratio = sound.max / noise.max
    noise_scaled = noise.apply_gain(
        ratio_to_db(ampl_ratio * noise_scale)
    )
    return sound.overlay(
        noise_scaled, loop=True
    )


def load_noises() -> list[AudioSegment]:
    return [
        AudioSegment.from_mp3(os.path.join(args.noises_dir, fname))
        for fname in os.listdir(args.noises_dir)
        if fname.endswith('.mp3')
    ]


def process_profile(profile_name: str, noises: list[AudioSegment]):
    """For a profile dir, creates profile dir
    in the output dir, augments profile sounds with noises
    and adds augmented sounds to the output profile
    """
    os.makedirs(os.path.join(args.output_dir, profile_name))
    profile_path = os.path.join(args.data_dir, profile_name)
    for fname in os.listdir(profile_path):

        if not fname.endswith(".wav"):
            continue

        sound = AudioSegment.from_wav(
            os.path.join(profile_path, fname)
        )
        random_noise = random.choice(noises)
        mixed_sound = mix_sound_and_noise(
            sound=sound, noise=random_noise, noise_scale=float(args.noise_scale)
        )

        mixed_sound.export(
            out_f=os.path.join(args.output_dir, profile_name, fname),
            format="wav"
        )


if __name__ == '__main__':

    # load noises
    print("Noises loading ...")
    noises = load_noises()
    print(f"Loaded f{len(noises)} noises")

    # creating mixed sounds and saving to output_dir
    print("Creating mixed sounds ...")
    for profile_name in tqdm(os.listdir(args.data_dir)):

        process_profile(
            profile_name=profile_name, noises=noises
        )

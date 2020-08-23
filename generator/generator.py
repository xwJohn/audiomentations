import os
import sys
import argparse
import numpy as np
from scipy.io import wavfile
import pathlib

from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Normalize,
    AddImpulseResponse,
    FrequencyMask,
    TimeMask,
    AddGaussianSNR,
    Resample,
    ClippingDistortion,
    AddBackgroundNoise,
    AddShortNoises)

import pyaudio
import wave

in_path = './input/'
out_path = './output/'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
RECORD_SECONDS = 1
rec_times = 5    # how many times need to record
label = ''
rec_files = ["开灯","关灯","绿色","蓝色","红色","温度","湿度","气压","亮度"]

DEMO_DIR = os.path.dirname(__file__)

def load_wav_file(sound_file_path):
    sample_rate, sound_np = wavfile.read(sound_file_path)
    if sample_rate != SAMPLE_RATE:
        raise Exception(
            "Unexpected sample rate {} (expected {})".format(sample_rate, SAMPLE_RATE)
        )

    if sound_np.dtype != np.float32:
        assert sound_np.dtype == np.int16
        sound_np = (sound_np / 32767).astype(np.float32)  # ends up roughly between -1 and 1

    return sound_np

def generator(wave_file,output_dir):
    """
    For each transformation, apply it to an example sound and write the transformed sounds to
    an output folder.
    """
    samples = load_wav_file(wave_file)
    _filename = os.path.basename(wave_file).split('.')[0]
    # AddImpulseResponse
    augmenter = Compose(
        [AddImpulseResponse(p=1.0, ir_path=os.path.join(DEMO_DIR, "ir"))]
    )
    output_file_path = os.path.join(
        output_dir, _filename  + "_AddImpulseResponse{:03d}.wav".format(0)
    )
    augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)
    # FrequencyMask
    # augmenter = Compose([FrequencyMask(p=1.0)])
    # for i in range(5):
    #     output_file_path = os.path.join(
    #         output_dir, _filename  + "_FrequencyMask{:03d}.wav".format(i)
    #     )
    #     augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    #     wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # TimeMask
    # augmenter = Compose([TimeMask(p=1.0)])
    # for i in range(5):
    #     output_file_path = os.path.join(output_dir, _filename  + "_TimeMask{:03d}.wav".format(i))
    #     augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    #     wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddGaussianSNR
    augmenter = Compose([AddGaussianSNR(p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, _filename  + "_AddGaussianSNR{:03d}.wav".format(i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddGaussianNoise
    augmenter = Compose(
        [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, _filename  + "_AddGaussianNoise{:03d}.wav".format(i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # TimeStretch
    augmenter = Compose([TimeStretch(min_rate=0.5, max_rate=1.5, p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, _filename  + "_TimeStretch{:03d}.wav".format(i))
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # PitchShift
    augmenter = Compose([PitchShift(min_semitones=-6, max_semitones=12, p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(output_dir, _filename  + "_PitchShift{:03d}.wav".format(i))
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Shift
    # augmenter = Compose([Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0)])
    # for i in range(5):
    #     output_file_path = os.path.join(output_dir, _filename  + "_Shift{:03d}.wav".format(i))
    #     augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    #     wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Shift without rollover
    augmenter = Compose([Shift(min_fraction=-0.2, max_fraction=0.2, rollover=False, p=1.0)]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, _filename  + "_ShiftWithoutRollover{:03d}.wav".format(i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Normalize
    augmenter = Compose([Normalize(p=1.0)])
    output_file_path = os.path.join(output_dir, _filename  + "_Normalize{:03d}.wav".format(0))
    augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # Resample
    # augmenter = Compose([Resample(min_sample_rate=12000, max_sample_rate=44100, p=1.0)])
    # for i in range(5):
    #     output_file_path = os.path.join(output_dir, _filename  + "_Resample{:03d}.wav".format(i))
    #     augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    #     wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # ClippingDistortion
    # augmenter = Compose([ClippingDistortion(max_percentile_threshold=10,p=1.0)])
    # for i in range(5):
    #     output_file_path = os.path.join(
    #         output_dir, _filename  + "_ClippingDistortion{:03d}.wav".format(i)
    #     )
    #     augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
    #     wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddBackgroundNoise
    augmenter = Compose(
        [AddBackgroundNoise(sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0)])
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, _filename  + "_AddBackgroundNoise{:03d}.wav".format(i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

    # AddShortNoises
    augmenter = Compose(
        [
            AddShortNoises(
                sounds_path=os.path.join(DEMO_DIR, "short_noises"),
                min_snr_in_db=0,
                max_snr_in_db=8,
                min_time_between_sounds=2.0,
                max_time_between_sounds=4.0,
                burst_probability=0.4,
                min_pause_factor_during_burst=0.01,
                max_pause_factor_during_burst=0.95,
                min_fade_in_time=0.005,
                max_fade_in_time=0.08,
                min_fade_out_time=0.01,
                max_fade_out_time=0.1,
                p=1.0,
            )
        ]
    )
    for i in range(5):
        output_file_path = os.path.join(
            output_dir, _filename  + "_AddShortNoises{:03d}.wav".format(i)
        )
        augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
        wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)


def rec(file_name, num=1):
    p = pyaudio.PyAudio()

    frames = []
    for x in range(num):
        file = file_name + '/' + file_name.split('/')[2] + label + '_' + str(x) + '.wav'
        print("按Enter后开始录音......%s"%file)
        input()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)        
        frames.clear()
        for i in range(0, int(SAMPLE_RATE / CHUNK * 0.5)):
            data = stream.read(CHUNK)
        print("===录音开始===")    
        for i in range(0, int(SAMPLE_RATE / CHUNK * (RECORD_SECONDS))):
            data = stream.read(CHUNK)
            frames.append(data)

        print("===录音结束!===")
        stream.stop_stream()
        stream.close()

        wf = wave.open(file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    p.terminate()


def main(argv):
    parser = argparse.ArgumentParser(description='Record and generater some dataset')

    parser.add_argument('label', type=str, help='specific a label for the dataset')
    parser.add_argument('times', type=int, help='How many time you would like to record')
    parser.add_argument('-r', "--record", help='Would you like to record?',action="store_true")
    parser.add_argument('-g', "--generate", help='Would you like to generater?',action="store_true")
    parser.add_argument('-I', "--input", help='Input dir for source data')
    parser.add_argument('-O', '--output', help='Output dir for generated data')
    args = parser.parse_args()
    global rec_times,label
    rec_times = args.times
    label = args.label

    if args.record == True:
        #create folder to save raw wav files
        for name in rec_files:
            in_folder = in_path + name
            os.makedirs(in_folder,exist_ok=True)
            # record audio data and save to input folder
            rec(in_folder, rec_times)

    # workaround ,should be refactor !!!!!
    if args.input is not None and args.output is not None and args.generate == True:
        inputPath = pathlib.Path(args.input)
        outputPath = pathlib.Path(args.output) / 'generated'
        os.makedirs(outputPath, exist_ok=True)
        filenames = os.listdir(inputPath)
        for filename in filenames:
            generator(inputPath / filename, outputPath)
        return

    if args.generate == True:
        #create folder to save generated wav files
        for name in rec_files:
            in_folder = in_path + name
            out_folder = out_path + name
            os.makedirs(out_folder,exist_ok=True)
            print("=====Entering...%s====="%in_folder)
            #generate audio file
            filenames = os.listdir(in_folder)
            for filename in filenames:
                if label in filename:
                    generator(os.path.join(in_folder, filename), out_folder)

if __name__ == "__main__":
    main(sys.argv)

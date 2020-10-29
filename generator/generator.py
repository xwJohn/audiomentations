import os,shutil
import sys
import argparse
import numpy as np
from scipy.io import wavfile

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


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
RECORD_SECONDS = 1

#rec_files = ["开灯","关灯","绿色","蓝色","红色","温度","湿度","气压","亮度"]
rec_files = ("backward","forward","left","right","stop","yes","no")

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

class Generator:
    def __init__(self):
        self.AddImpulseResponse =    (False, 5)
        self.FrequencyMask =         (False, 5)
        self.TimeMask =              (False, 5)
        self.AddGaussianSNR =        (False, 5)
        self.AddGaussianNoise =      (False, 5)
        self.TimeStretch =           (False, 5)
        self.PitchShift =            (False, 5)
        self.Shift =                 (False, 5)
        self.ShiftWithoutRoll =      (False, 5)
        self.PitchShift =            (False, 5)
        self.PitchShift =            (False, 5)
        self.Normalize =             (False, 5)
        self.Resample =              (False, 5)
        self.ClippingDistortion =    (False, 5)        
        self.AddBackgroundNoise =    (False, 5)
        self.AddWhiteNoise =         (True , 1)
        self.AddPinkNoise =          (True , 1)
        self.AddShortNoises =        (False, 5)         
       
    def generate(self,wave_file,output_dir):
        """
        For each transformation, apply it to an example sound and write the transformed sounds to
        an output folder.
        """
        samples = load_wav_file(wave_file)
        _filename = os.path.basename(wave_file).split('.')[0]
        # AddImpulseResponse
        if self.AddImpulseResponse[0]:
            augmenter = Compose(
                [AddImpulseResponse(p=1.0, ir_path=os.path.join(DEMO_DIR, "ir"))]
            )
            output_file_path = os.path.join(
                output_dir, _filename  + "_AddImpulseResponse{:03d}.wav".format(0)
            )
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
            wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)
        # FrequencyMask
        if self.FrequencyMask[0]:
            augmenter = Compose([FrequencyMask(p=1.0)])
            for i in range(5):
                output_file_path = os.path.join(
                    output_dir, _filename  + "_FrequencyMask{:03d}.wav".format(i)
                )
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # TimeMask
        if self.TimeMask[0]:        
            augmenter = Compose([TimeMask(p=1.0)])
            for i in range(5):
                output_file_path = os.path.join(output_dir, _filename  + "_TimeMask{:03d}.wav".format(i))
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # AddGaussianSNR
        if self.AddGaussianSNR[0]:
            augmenter = Compose([AddGaussianSNR(p=1.0)])
            for i in range(5):
                output_file_path = os.path.join(
                    output_dir, _filename  + "_AddGaussianSNR{:03d}.wav".format(i)
                )
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # AddGaussianNoise
        if self.AddGaussianNoise[0]:
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
        if self.TimeStretch[0]:
            augmenter = Compose([TimeStretch(min_rate=0.5, max_rate=1.5, p=1.0)])
            for i in range(5):
                output_file_path = os.path.join(output_dir, _filename  + "_TimeStretch{:03d}.wav".format(i))
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # PitchShift
        if self.PitchShift[0]:
            augmenter = Compose([PitchShift(min_semitones=-6, max_semitones=12, p=1.0)])
            for i in range(5):
                output_file_path = os.path.join(output_dir, _filename  + "_PitchShift{:03d}.wav".format(i))
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # Shift
        if self.Shift[0]:
            augmenter = Compose([Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0)])
            for i in range(5):
                output_file_path = os.path.join(output_dir, _filename  + "_Shift{:03d}.wav".format(i))
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # Shift without rollover
        if self.ShiftWithoutRoll[0]:
            augmenter = Compose([Shift(min_fraction=-0.2, max_fraction=0.2, rollover=False, p=1.0)]
            )
            for i in range(5):
                output_file_path = os.path.join(
                    output_dir, _filename  + "_ShiftWithoutRollover{:03d}.wav".format(i)
                )
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # Normalize
        if self.Normalize[0]:
            augmenter = Compose([Normalize(p=1.0)])
            output_file_path = os.path.join(output_dir, _filename  + "_Normalize{:03d}.wav".format(0))
            augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
            wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # Resample
        if self.Resample[0]:
            augmenter = Compose([Resample(min_sample_rate=12000, max_sample_rate=44100, p=1.0)])
            for i in range(5):
                output_file_path = os.path.join(output_dir, _filename  + "_Resample{:03d}.wav".format(i))
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # ClippingDistortion
        if self.ClippingDistortion[0]:
            augmenter = Compose([ClippingDistortion(max_percentile_threshold=10,p=1.0)])
            for i in range(5):
                output_file_path = os.path.join(
                    output_dir, _filename  + "_ClippingDistortion{:03d}.wav".format(i)
                )
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

        # AddBackgroundNoise
        if self.AddBackgroundNoise[0]:
            augmenter = Compose(
                [AddBackgroundNoise(sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0)])
            for i in range(5):
                output_file_path = os.path.join(
                    output_dir, _filename  + "_AddBackgroundNoise{:03d}.wav".format(i)
                )
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)
        # AddWhiteNoise
        if self.AddWhiteNoise[0]:
            augmenter = Compose(
                [AddBackgroundNoise(sounds_path=os.path.join(DEMO_DIR, "white_noises"), p=1.0)])
            for i in range(self.AddWhiteNoise[1]):
                output_file_path = os.path.join(
                    output_dir, _filename  + "_AddWhiteNoise{:03d}.wav".format(i)
                )
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)
        # AddPinkNoise
        if self.AddPinkNoise[0]:
            augmenter = Compose(
                [AddBackgroundNoise(sounds_path=os.path.join(DEMO_DIR, "pink_noises"), p=1.0)])
            for i in range(self.AddPinkNoise[1]):
                output_file_path = os.path.join(
                    output_dir, _filename  + "_AddPinkNoise{:03d}.wav".format(i)
                )
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
                wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)
        # AddShortNoises
        if self.AddShortNoises[0]:
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
        file = file_name + '/' + file_name.split('/')[2] + args.who + '_' + str(x) + '.wav'
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

#record arguments
    parser.add_argument('-w', "--who", type=str, help='specific a user for the filename, who record this?',default='')
    parser.add_argument('-t', '--times', type=int, help='How many times you would like to record',default=5)
    parser.add_argument('-r', "--record", help='Would you like to record?',action="store_true")
 #generate arguments   
    parser.add_argument('-g', "--generate", help='Would you like to generater?',action="store_true")
    parser.add_argument('-i', "--inputPath", help='Where the data stored?', default='./input/')
    parser.add_argument('-o', "--outputPath", help='Where you want store the generated data?', default='./output/')    
    args = parser.parse_args()

    if args.record == True:
        assert(args.generate == False)
        #create folder to save raw wav files
        for name in rec_files:
            in_folder = args.inputPath + name
            os.makedirs(in_folder,exist_ok=True)
            # record audio data and save to input folder
            rec(in_folder, args.times)

    if args.generate == True:
        assert(args.record == False)
        gen = Generator()        
        #create folder to save generated wav files
        for name in rec_files:
            in_folder = args.inputPath + "/" + name
            out_folder = args.outputPath + "/" + name
            os.makedirs(out_folder,exist_ok=True)
            print("=====Entering...%s====="%in_folder)
            #generate audio file
            filenames = os.listdir(in_folder)
            for filename in filenames:
                #copy one raw data
                shutil.copy(os.path.join(in_folder, filename), out_folder)
                #generate
                gen.generate(os.path.join(in_folder, filename), out_folder)

if __name__ == "__main__":
    main(sys.argv)

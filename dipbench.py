import os
import sys
import math
import random
import colorsys
import wave
import pygame
import pygame.midi
from pygame.locals import *
import pyaudio
import numpy as np
from scipy.stats import spearmanr


app_title = 'DiP-Bench (Digital Piano Benchmark)'
app_version = '0.05'

# settings
sampling_freq = 48000
pitch_direction_threshohld = 0.95
velocity_direction_threshohld = 0.9987
latency_threshold = 0.05 # ratio from std (0.01 means 1% of std)
latency_threshold2 = 2.0 # ratio from noise floor (16.0 means 16x = 30dB)

sample_length = 2048

correl_size = 2048
shift_range = 1024

realtime_analysis_length = 8192
spectrum_size = 8192
min_level = -48.0 #dB


# size
screen_size = None
floating_window_size = (1280, 720)
window_size = None
full_screen = False
line_width = None
base_size = None
base_margin = None
vector_scope_size = None

# keyboard size
keyboard_margin_x = None
key_width = None
keyboard_top = None
energy_bottom = None
white_key_width = None
white_key_height = None
black_key_width = None
black_key_height = None

# UI velocity size
velocity_width = None
velocity_left = None
velocity_size = None
velocity_bottom = None


class Keyboard:
    keys = None

    def __init__(self):
        global keyboard_margin_x, key_width
        class Key:
            pass
        self.keys = []
        for i in range(128):
            oct = i // 12 - 1
            note = i % 12
            black_key = note in [1, 3, 6, 8, 10]
            x = round(keyboard_margin_x + key_width * (oct * 7 + [0.5, 0.925, 1.5, 2.075, 2.5, 3.5, 3.85, 4.5, 5, 5.5, 6.15, 6.5][note] - 5))
            key= Key()
            key.note_no = i
            key.black_key = black_key
            key.x = x
            key.normalized_x = round(keyboard_margin_x + 0.5 * key_width + (key_width * 7) / 12 * (i - 21))
            self.keys.append(key)


def SetUI():
    global screen_size, floating_window_size, window_size, full_screen, line_width, base_size, base_margin, vector_scope_size
    global keyboard_margin_x, key_width, keyboard_top, energy_bottom, white_key_width, white_key_height, black_key_width, black_key_height
    global velocity_width, velocity_left, velocity_size, velocity_bottom

    if full_screen:
        window_size = screen_size
    else:
        window_size = floating_window_size
    base_size = window_size[0] // 240 # 8 pixel in 1920 x 1080
    if base_size > window_size[1] // 135:
        base_size = window_size[1] // 135
    line_width = base_size // 9 + 1
    base_margin = base_size * 3
    vector_scope_size = window_size[1] / 16 # sd = height/32

    # keyboard size
    keyboard_margin_x = base_size * 3
    key_width = (window_size[0] - keyboard_margin_x * 2) / 52
    keyboard_top = base_size * 100
    energy_bottom = keyboard_top - base_size * 7
    white_key_width = round(key_width * 33 / 36) #22.5 / 23.5
    white_key_height = round(key_width * 150 / 23.5)
    black_key_width = round(key_width * 23 / 36) #15 / 23.5
    black_key_height = round(key_width * 100 / 23.5)

    # velocity size
    velocity_width = base_size * 24
    velocity_left = window_size[0] - velocity_width - base_margin
    velocity_size = base_size // 2
    velocity_bottom = keyboard_top - base_size * 19

    pygame.display.set_mode(window_size, pygame.FULLSCREEN if full_screen else pygame.RESIZABLE) # workaround
    return pygame.display.set_mode(window_size, pygame.FULLSCREEN if full_screen else pygame.RESIZABLE), Keyboard()



class DipBench:
    audio_inputs = []
    midi_inputs = []
    midi_outputs = []
    audio = None

    midi_in = None
    midi_out = None
    stream = None

    mode = 0
    monitor_mode = 0
    last_error = None
    terminated = False

    # measurement results in pitch direction
    tone = -1
    last_hue = 0.0
    pitch_waveforms = None
    pitch_spectrums = None
    duplication_in_pitch = None
    pitch_correl = None
    pitch_color = [(255,255,255)] * 88
    pitch_variation = None
    pitch_checked = None
    pitch_latency = None
    pitch_latency_average = None
    pitch_latency_std = None
    pitch_volume = None
    pitch_volume_average = None
    pitch_volume_std = None

    # measurement results in velocity direction
    velocity = -1
    velocity_waveforms = None
    velocity_spectrums = None
    duplication_in_velocity = None
    velocity_correl = None
    velocity_color = [(64,64,64)] * 127
    velocity_layer = None
    velocity_checked = None
    velocity_latency = None
    velocity_latency_average = None
    velocity_latency_std = None
    velocity_volume = None
    max_correl = 0.0

    # realtime measurement results
    realtime_waveform = None
    realtime_spectrum = None
    realtime_note_on = [False] * 88
    realtime_key_on = [False] * 88
    realtime_velocity = [0] * 88
    realtime_damper_on = False

    def __init__(self):
        self.audio = pyaudio.PyAudio()

        # prepare midi
        pygame.midi.init()
        for i in range(pygame.midi.get_count()):
            midi_device_info = pygame.midi.get_device_info(i)
            if not midi_device_info[4]: # not opened.
                device_name = midi_device_info[1].decode()
                if midi_device_info[2]: #input
                    self.midi_inputs.append(device_name)
                else:
                    self.midi_outputs.append(device_name)
        pygame.midi.quit()

        # prepare audio
        info = self.audio.get_host_api_info_by_index(0)
        device_count = info.get('deviceCount')
        for i in range(device_count):
            device = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device.get('maxInputChannels') >= 2:
                self.audio_inputs.append(device.get('name'))
        
    def __del__(self):
        self.terminate()
        self.process()
        self.audio.terminate()

    def terminate(self):
        self.terminated = True
        self.last_error = None

    def shift_audio_inputs(self):
        self.audio_inputs.append(self.audio_inputs.pop(0))

    def shift_midi_inputs(self):
        self.midi_inputs.append(self.midi_inputs.pop(0))

    def shift_midi_outputs(self):
        self.midi_outputs.append(self.midi_outputs.pop(0))

    def __open_audio(self):
        info = self.audio.get_host_api_info_by_index(0)
        device_count = info.get('deviceCount')
        index = -1
        for i in range(device_count):
            device = self.audio.get_device_info_by_host_api_device_index(0, i)
            if self.audio_inputs[0] == device.get('name') and device.get('maxInputChannels') >= 2:
                index = i
        if index >= 0:
            self.stream = self.audio.open(input=True, input_device_index = index, format=pyaudio.paInt16, channels=2, rate=sampling_freq, frames_per_buffer=sample_length)
        else:
            self.last_error = f'Can\'t open audio input "{self.audio_inputs[0]}".'

    def __close_audio(self):
        self.stream.close()
        self.stream = None

    def __open_midi_in(self):
        self.__close_midi_in()
        pygame.midi.init()
        if len(self.midi_inputs) == 0:
            self.last_error = f'No midi input available.'

        for i in range(pygame.midi.get_count()):
            midi_device_info = pygame.midi.get_device_info(i)
            if not midi_device_info[4]: # not opened.
                device_name = midi_device_info[1].decode()
                if midi_device_info[2]: # input
                    if self.midi_inputs[0] == device_name:
                        self.midi_in = pygame.midi.Input(i)
                        return
        self.last_error = f'Can\'t open midi input "{self.midi_inputs[0]}".'
        pygame.midi.quit()
        
    def __open_midi_out(self):
        self.__close_midi_out()
        pygame.midi.init()
        if len(self.midi_outputs) == 0:
            self.last_error = f'No midi output available.'

        for i in range(pygame.midi.get_count()):
            midi_device_info = pygame.midi.get_device_info(i)
            if not midi_device_info[4]: # not opened.
                device_name = midi_device_info[1].decode()
                if not midi_device_info[2]: # output
                    if self.midi_outputs[0] == device_name:
                        self.midi_out = pygame.midi.Output(i)
                        return
        self.last_error = f'Can\'t open midi output "{self.midi_outputs[0]}".'
        pygame.midi.quit()
    
    def __close_midi_in(self):
        if self.midi_in is not None:
            self.midi_in.close()
            self.midi_in = None
            pygame.midi.quit()

    def __close_midi_out(self):
        if self.midi_out is not None:
            self.midi_out.close()
            self.midi_out = None
            pygame.midi.quit()

    def set_tone(self, tone, play=True):
        self.tone = np.clip(tone, -1, 87)
        if self.pitch_waveforms is not None and self.pitch_waveforms[self.tone] is not None and tone >= 0 and play:
            self.monitor_mode = 1
            sound = pygame.sndarray.make_sound(self.pitch_waveforms[self.tone])
            sound.play()

    def shift_tone_next(self):
        self.set_tone((self.tone + 1) % 88)

    def shift_tone_previous(self):
        if self.tone >= 0:
            self.set_tone((self.tone + 87) % 88)
        else:
            self.set_tone(87)

    def set_velocity(self, velocity, play=True):
        self.velocity = np.clip(velocity, -1, 126)
        if self.velocity_waveforms is not None and self.velocity_waveforms[self.velocity] is not None and velocity >= 0 and play:
            self.monitor_mode = 2
            sound = pygame.sndarray.make_sound(self.velocity_waveforms[self.velocity])
            sound.play()

    def shift_velocity_next(self):
        self.set_velocity((self.velocity + 1) % 127)

    def shift_velocity_previous(self):
        if self.velocity >= 0:
            self.set_velocity((self.velocity + 126) % 127)
        else:
            self.set_velocity(126)

    def get_note_on(self, note):
        if note == self.tone:
            return True
        if self.realtime_note_on:
            return self.realtime_note_on[note]
        return False

    def measure_pitch_variation(self):
        self.terminated = False
        self.last_error = None
        self.mode = self.monitor_mode = 1
        self.pitch_waveforms = [None] * 88
        self.pitch_spectrums = [None] * 88
        self.duplication_in_pitch = [None] * 88
        self.pitch_correl = [0.0] * 87
        self.pitch_color = [(255,255,255)] * 88
        self.tone = 0
        self.pitch_variation = 0
        self.pitch_checked = 0
        self.pitch_latency = [None] * 88
        self.pitch_latency_average = None
        self.pitch_latency_std = None
        self.pitch_volume = [None] * 88
        self.pitch_volume_average = None
        self.pitch_volume_std = None

    def __get_spectrum(self, waveform):
        waveform = np.sum(waveform,axis=1)
        if len(waveform) < spectrum_size:
            waveform = waveform * np.hanning(len(waveform))
            waveform = np.pad(waveform, ((0, spectrum_size - len(waveform))))
        else:
            waveform = waveform[:spectrum_size]
            waveform = waveform * np.hanning(spectrum_size)
        spectrum = np.log(np.abs(np.fft.fft(waveform / 32768.0)) / spectrum_size) / np.log(2) * 6.0 # in dB
        return spectrum

    def __check_duplication(self, pos, waveform1, waveform2, nextnote=False):
        if nextnote:
            ratio = np.exp(np.log(2.0) / 12.0)
            original_pos = np.linspace(0, len(waveform1) , len(waveform1))
            interp_pos = original_pos * ratio
            waveform1 = np.stack([np.interp(interp_pos, original_pos, waveform1[:, 0]), np.interp(interp_pos, original_pos, waveform1[:, 1])], axis = 1)
            offset = -int(sample_length * (ratio - 1.0))
        else:
            waveform1 = waveform1.astype(np.float32)
            offset = 0
        max_correl = 0.0
        shift_pos = -1
        for shift in range(shift_range * 2):
            left = sample_length + (shift - shift_range) + offset
            w1 = waveform1[left:left + correl_size].flatten()
            w2 = waveform2[sample_length:sample_length + correl_size].flatten()
            correl = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
            if correl > max_correl:
                max_correl = correl
                shift_pos = shift
        print(pos, shift_pos - shift_range, max_correl)

        return max_correl

    def __check_latency(self, waveform):
        attack = np.abs(waveform[sample_length:sampling_freq // 4, 0])
        std = np.max(attack)
        noise_std = np.max(np.abs(waveform[:sample_length]))

        if std * latency_threshold > noise_std * latency_threshold2:
            threshold = std * latency_threshold
        else:
            threshold = noise_std * latency_threshold2
        indices = np.where(attack > threshold)
        if indices[0].size > 0:
            return indices[0][0] / sampling_freq
        else:
            return None

    def __check_volume(self, waveform, latency):
        volume = np.log(np.sqrt(np.mean(np.square(waveform[sample_length + int((latency if latency is not None else 0.01) * sampling_freq):] / 32768.0)))) / np.log(2.0) * 6.0
        return volume

    def __pitch_variation_measurement(self):
        if self.stream is not None:
            buf = self.stream.read(sample_length)
            if self.pitch_waveforms[self.tone] is None:
                self.pitch_waveforms[self.tone] = np.frombuffer(buf, dtype=np.int16).reshape(sample_length, 2)
                if self.midi_out:
                    self.midi_out.note_on(self.tone + 21, 100)
            else:
                previous_len = len(self.pitch_waveforms[self.tone])
                self.pitch_waveforms[self.tone] = np.concatenate([self.pitch_waveforms[self.tone], np.frombuffer(buf, dtype=np.int16).reshape(sample_length, 2)])
                self.pitch_spectrums[self.tone] = self.__get_spectrum(self.pitch_waveforms[self.tone])
                if previous_len <= sampling_freq // 2 and len(self.pitch_waveforms[self.tone]) > sampling_freq // 2:
                    # note off
                    if self.midi_out:
                        self.midi_out.note_off(self.tone + 21, 100)

                    # duplication
                    duplication = False
                    if self.tone > 0:
                        max_correl = self.__check_duplication(self.tone + 21, self.pitch_waveforms[self.tone - 1], self.pitch_waveforms[self.tone], True)
                        self.pitch_correl[self.tone - 1] = max_correl
                        duplication = max_correl > pitch_direction_threshohld
                    self.duplication_in_pitch[self.tone] = duplication
                    if not duplication:
                        self.last_hue = (self.last_hue + 0.25 + random.random() * 0.5) % 1.0
                        self.pitch_variation = self.pitch_variation + 1
                    self.pitch_color[self.tone] = colorsys.hsv_to_rgb(self.last_hue, 0.8, 255.0)
                    self.pitch_checked = self.pitch_checked + 1

                    # latency
                    self.pitch_latency[self.tone] = self.__check_latency(self.pitch_waveforms[self.tone])
                    self.pitch_latency_average = np.nanmean(np.array(self.pitch_latency, dtype=float))
                    if len([latency for latency in self.pitch_latency if latency is not None]) >= 1:
                        self.pitch_latency_std = np.nanstd(np.array(self.pitch_latency, dtype=float))

                    # volume
                    if self.tone < 88 - 18: # ignore upper 18 tones
                        self.pitch_volume[self.tone] = self.__check_volume(self.pitch_waveforms[self.tone], self.pitch_latency[self.tone])
                        self.pitch_volume_average = np.nanmean(np.array(self.pitch_volume, dtype=float))
                        if len([volume for volume in self.pitch_volume if volume is not None]) >= 1:
                            self.pitch_volume_std = np.nanstd(np.array(self.pitch_volume, dtype=float))

                elif len(self.pitch_waveforms[self.tone]) >= sampling_freq:
                    self.__close_audio()
                    self.__close_midi_out()

                    if self.tone < 87 and not self.terminated:
                        self.tone = self.tone + 1
                    else:
                        self.mode = 0
                        self.tone = -1

        if self.mode == 1 and self.stream is None and self.last_error == None:
            # start recording
            self.__open_audio()
            self.__open_midi_out()

    def measure_velocity_layer(self):
        self.terminated = False
        self.last_error = None
        self.mode = self.monitor_mode = 2
        self.velocity_waveforms = [None] * 127
        self.velocity_spectrums = [None] * 127
        self.duplication_in_velocity = [None] * 127
        self.velocity_correl = [0.0] * 126
        self.velocity_color = [(64,64,64)] * 127
        self.velocity = 0
        self.velocity_layer = 0
        self.velocity_checked = 0
        self.max_correl = 0.0
        self.velocity_latency = [None] * 127
        self.velocity_latency_average = None
        self.velocity_latency_std = None
        self.velocity_volume = [None] * 127
        if self.tone < 0:
            self.tone = 39

    def __velocity_layer_measurement(self):
        if self.stream is not None:
            buf = self.stream.read(sample_length)
            if self.velocity_waveforms[self.velocity] is None:
                self.velocity_waveforms[self.velocity] = np.frombuffer(buf, dtype=np.int16).reshape(sample_length, 2)
                if self.midi_out:
                    self.midi_out.note_on(self.tone + 21, self.velocity + 1)
            else:
                previous_len = len(self.velocity_waveforms[self.velocity])
                self.velocity_waveforms[self.velocity] = np.concatenate([self.velocity_waveforms[self.velocity], np.frombuffer(buf, dtype=np.int16).reshape(sample_length, 2)])
                self.velocity_spectrums[self.velocity] = self.__get_spectrum(self.velocity_waveforms[self.velocity])
                if previous_len <= sampling_freq // 2 and len(self.velocity_waveforms[self.velocity]) > sampling_freq // 2:
                    # note off
                    if self.midi_out:
                        self.midi_out.note_off(self.tone + 21, self.velocity + 1)

                    if np.std(self.velocity_waveforms[self.velocity][sample_length:]) > np.std(self.velocity_waveforms[self.velocity][:sample_length]) * 64.0: # SN > 36dB
                        # latency
                        self.velocity_latency[self.velocity] = self.__check_latency(self.velocity_waveforms[self.velocity])
                        self.velocity_latency_average = np.nanmean(np.array(self.velocity_latency, dtype=float))
                        if len([latency for latency in self.velocity_latency if latency is not None]) >= 1:
                            self.velocity_latency_std = np.nanstd(np.array(self.velocity_latency, dtype=float))

                    # volume
                    if np.std(self.velocity_waveforms[self.velocity][sample_length:]) > np.std(self.velocity_waveforms[self.velocity][:sample_length]) * 4.0: # SN > 12dB
                        self.velocity_volume[self.velocity] = self.__check_volume(self.velocity_waveforms[self.velocity], self.velocity_latency[self.velocity])

                    # duplication
                    duplication = False
                    if self.velocity > 0:
                        max_correl = self.__check_duplication(self.velocity, self.velocity_waveforms[self.velocity - 1], self.velocity_waveforms[self.velocity])
                        # max_correl = np.dot(self.velocity_spectrums[self.velocity - 1], self.velocity_spectrums[self.velocity]) / ((np.linalg.norm(self.velocity_spectrums[self.velocity - 1]) * np.linalg.norm(self.velocity_spectrums[self.velocity])))
                        self.velocity_correl[self.velocity - 1] = max_correl
                        duplication = max_correl > velocity_direction_threshohld
                        if max_correl > self.max_correl:
                            self.max_correl = max_correl    
                    self.duplication_in_velocity[self.velocity] = duplication and self.max_correl > velocity_direction_threshohld
                    if self.max_correl > velocity_direction_threshohld or self.velocity_latency[self.velocity] is not None:
                        if not duplication or self.velocity_layer == 0:
                            self.last_hue = (self.last_hue + 0.25 + random.random() * 0.5) % 1.0
                            self.velocity_layer = self.velocity_layer + 1
                        self.velocity_color[self.velocity] = colorsys.hsv_to_rgb(self.last_hue, 0.8, 224.0)
                        self.velocity_checked = self.velocity_checked + 1

                elif len(self.velocity_waveforms[self.velocity]) >= sampling_freq:
                    self.__close_audio()
                    self.__close_midi_out()

                    if self.velocity < 126 and not self.terminated:
                        self.velocity = self.velocity + 1
                    else:
                        self.mode = 0
                        self.velocity = -1

        if self.mode == 2 and self.stream is None and self.last_error == None:
            # start recording
            self.__open_audio()
            self.__open_midi_out()

    def realtime_analysis(self):
        self.terminated = False
        self.last_error = None
        self.mode = self.monitor_mode = 3
        self.realtime_waveform = None
        self.realtime_spectrum = None
        self.tone = -1
        self.velocity = -1
        self.realtime_note_on = [False] * 88
        self.realtime_key_on = [False] * 88
        self.realtime_velocity = [0] * 88
        self.realtime_damper_on = False
        self.__open_audio()
        self.__open_midi_in()

    def __realtime_analysis(self):
        if self.terminated:
            self.mode = 0
            self.__close_audio()
            self.__close_midi_in()
            self.realtime_waveform = None
            self.realtime_spectrum = None
            return

        if self.midi_in and self.midi_in.poll():
            midi_events = self.midi_in.read(256)
            for midi_event in midi_events:
                key = midi_event[0][1] - 21
                if key >=0 and key < 88:
                    if midi_event[0][0] & 0xf0 == 0x90 and midi_event[0][2] > 0: # note on
                        self.realtime_note_on[key] = self.realtime_key_on[key] = True
                        self.realtime_velocity[key] = midi_event[0][2]
                    if midi_event[0][0] & 0xf0 == 0x80 or (midi_event[0][0] & 0xf0 == 0x90 and midi_event[0][2] == 0): # note off
                        self.realtime_key_on[key] = False
                        self.realtime_note_on[key] = self.realtime_note_on[key] and self.realtime_damper_on
                if midi_event[0][0] & 0xf0 == 0xb0: # control
                    if midi_event[0][1] & 0xff == 0x40: # damper
                        self.realtime_damper_on = midi_event[0][2] > 0
                        if not self.realtime_damper_on:
                            self.realtime_note_on = self.realtime_key_on.copy()

        if self.stream and self.last_error == None:
            buf = self.stream.read(sample_length)
            if self.realtime_waveform is None:
                self.realtime_waveform = np.frombuffer(buf, dtype=np.int16).reshape(sample_length, 2)
            else:
                self.realtime_waveform = np.concatenate([self.realtime_waveform[-(realtime_analysis_length - sample_length):], np.frombuffer(buf, dtype=np.int16).reshape(sample_length, 2)])
                self.realtime_spectrum = self.__get_spectrum(self.realtime_waveform)

    def process(self):
        if self.mode == 1:
            # Measure pitch variation
            self.__pitch_variation_measurement()
        elif self.mode == 2:
            # Measure velocity layers
            self.__velocity_layer_measurement()
        elif self.mode == 3:
            # Realtime analysis
            self.__realtime_analysis()
    
    def __save_waveforms(self, waveforms, fn_prefix, index_offset):
        for i, waveform in enumerate(waveforms):
            if waveform is not None:
                wav = wave.Wave_write(f'{fn_prefix}{i + index_offset:0=3}.wav')
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(waveform.tobytes())
                wav.close()

    def save_waveforms(self):
        if self.pitch_waveforms is not None:
            self.__save_waveforms(self.pitch_waveforms, 'key_', 0)
        if self.velocity_waveforms is not None:
            self.__save_waveforms(self.velocity_waveforms, 'velocity_', 1)

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def main():
    global app_title, app_version
    global screen_size, floating_window_size, window_size, full_screen, line_width, base_size, base_margin, vector_scope_size
    global keyboard_margin_x, key_width, keyboard_top, energy_bottom, white_key_width, white_key_height, black_key_width, black_key_height
    global velocity_width, velocity_left, velocity_size, velocity_bottom

    # initialization
    pygame.mixer.pre_init(frequency=sampling_freq, size=-16, channels=2)
    pygame.init()
    pygame.display.set_icon(pygame.image.load(resource_path('icon_128x128.png')))
    pygame.display.set_caption(app_title)
    display_info = pygame.display.Info()
    screen_size = (display_info.current_w, display_info.current_h)

    screen, keyboard = SetUI()
    refresh_font = True

    dipbench = DipBench()

    # main_loop
    terminated = False

    clock = pygame.time.Clock()

    while (not terminated):
        if refresh_font:
            font = pygame.font.Font(None, base_size * 4)
            app_title_text = font.render(f'{app_title} version {app_version}   /   Frieve 2022-2024', True, (255,255,255))
            help_text = font.render('[A] Change Audio-In, [I] Change MIDI-In, [O] Change MIDI-Out, [P] Measure pitch variation, [V] Measure velocity layer, [R] Real-time mode, [F11] Full screen, [Q] Quit', True, (128,192,255))
            help_text2 = font.render('[ESC] Abort', True, (128,192,255))
            refresh_font = False

        screen.fill((0,0,0))

        # fps_text = font.render(f"FPS: {clock.get_fps():.2f}", True, (255, 255, 255))
        # screen.blit(fps_text, (window_size[0] - base_margin - base_size * 16, base_margin))

        # handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                terminated = True

            elif event.type == KEYDOWN:
                if event.key == K_q:
                    terminated = True
                elif event.key == K_ESCAPE:
                    dipbench.terminate()
                elif event.key == K_F11:
                    full_screen = not full_screen
                    screen, keyboard = SetUI()
                    refresh_font = True
                elif dipbench.mode == 0:
                    # mode change
                    if event.key == K_p:
                        dipbench.measure_pitch_variation()
                    elif event.key == K_v:
                        dipbench.measure_velocity_layer()
                    elif event.key == K_r:
                        dipbench.realtime_analysis()

                    # setting
                    elif event.key == K_a:
                        dipbench.shift_audio_inputs()
                    elif event.key == K_i:
                        dipbench.shift_midi_inputs()
                    elif event.key == K_o:
                        dipbench.shift_midi_outputs()

                    # preview
                    elif event.key == K_RIGHT:
                        dipbench.shift_tone_next()
                    elif event.key == K_LEFT:
                        dipbench.shift_tone_previous()
                    elif event.key == K_UP:
                        dipbench.shift_velocity_next()
                    elif event.key == K_DOWN:
                        dipbench.shift_velocity_previous()

                    # i/o
                    elif event.key == K_s:
                        dipbench.save_waveforms()

            elif event.type == pygame.VIDEORESIZE and not full_screen:
                floating_window_size = (event.w, event.h)
                screen, keyboard = SetUI()
                refresh_font = True

            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                if dipbench.mode == 0:
                    mouse_click_tone = -1    
                    mouse_click_velocity = -1    
                    for key in [key for key in keyboard.keys[21:109] if key.black_key]:
                        if event.pos[0] >= key.x - black_key_width / 2 and event.pos[0] < key.x + black_key_width / 2 and event.pos[1] >= keyboard_top and event.pos[1] < keyboard_top + black_key_height:
                            if key.note_no >= 21 and key.note_no < 109:
                                mouse_click_tone = key.note_no - 21
                    if mouse_click_tone == -1:
                        for key in [key for key in keyboard.keys[21:109] if not key.black_key]:
                            if event.pos[0] >= key.x - white_key_width / 2 and event.pos[0] < key.x + white_key_width / 2 and event.pos[1] >= keyboard_top and event.pos[1] < keyboard_top + white_key_height:
                                if key.note_no >= 21 and key.note_no < 109:
                                    mouse_click_tone = key.note_no - 21
                    if event.pos[0] >= window_size[0] - base_margin - velocity_width and event.pos[0] < window_size[0] - base_margin and event.pos[1] >= velocity_bottom - 127 * velocity_size and event.pos[1] < velocity_bottom:
                        mouse_click_velocity = (velocity_bottom - event.pos[1]) // velocity_size
                    dipbench.set_tone(mouse_click_tone)
                    dipbench.set_velocity(mouse_click_velocity)

        dipbench.process()

        # display info
        screen.blit(app_title_text, [base_margin, base_margin])
        if len(dipbench.audio_inputs) > 0 and len(dipbench.midi_inputs) > 0 and len(dipbench.midi_outputs) > 0:
            device_info_text = font.render(f'Audio input : {dipbench.audio_inputs[0]}, MIDI input : {dipbench.midi_inputs[0]}, MIDI output : {dipbench.midi_outputs[0]}', True, (255,255,255))
        else:
            device_info_text = font.render('Audio or MIDI device not available', True, (255,0,0))
        screen.blit(device_info_text, [base_margin, base_margin + base_size * 5])
        screen.blit(help_text if dipbench.mode == 0 else help_text2, [base_margin, base_margin + base_size * 10])
        if dipbench.last_error is not None:
            error_text = font.render(dipbench.last_error, True, (255,0,0))
            screen.blit(error_text, [base_margin, base_margin + base_size * 18])

        # prepare waveform
        monitor_wave = None
        waveform_std = 1.0
        display_latency = None
        display_volume = None
        if dipbench.monitor_mode == 1 and dipbench.pitch_waveforms is not None and dipbench.tone >= 0 and dipbench.pitch_waveforms[dipbench.tone] is not None:
            if len(dipbench.pitch_waveforms[dipbench.tone]) >= sample_length * 2:
                monitor_wave = dipbench.pitch_waveforms[dipbench.tone]
                waveform_std = np.std(monitor_wave[int(sample_length * 1.5):sample_length * 2])
                if dipbench.pitch_latency is not None and len(dipbench.pitch_latency) > dipbench.tone:
                    display_latency = dipbench.pitch_latency[dipbench.tone]
                if dipbench.pitch_volume is not None and len(dipbench.pitch_volume) > dipbench.tone:
                    display_volume = dipbench.pitch_volume[dipbench.tone]
        elif dipbench.monitor_mode == 2 and dipbench.velocity_waveforms is not None and dipbench.velocity >= 0 and dipbench.velocity_waveforms[dipbench.velocity] is not None:
            if len(dipbench.velocity_waveforms[dipbench.velocity]) >= sample_length * 2:
                monitor_wave = dipbench.velocity_waveforms[dipbench.velocity]
                waveform_std = np.std(monitor_wave[int(sample_length * 1.5):sample_length * 2])
                if dipbench.velocity_latency is not None and len(dipbench.velocity_latency) > dipbench.velocity:
                    display_latency = dipbench.velocity_latency[dipbench.velocity]
                if dipbench.velocity_volume is not None and len(dipbench.velocity_volume) > dipbench.velocity:
                    display_volume = dipbench.velocity_volume[dipbench.velocity]
        elif dipbench.monitor_mode == 3 and dipbench.realtime_waveform is not None:
            monitor_wave = dipbench.realtime_waveform
            waveform_std = np.std(monitor_wave)
        if waveform_std < 32768 * 2.0 ** (min_level / 6.0):
            waveform_std = 32768 * 2.0 ** (min_level / 6.0)

        # draw white keybed
        for key in [key for key in keyboard.keys[21:109] if not key.black_key]:
            screen.fill((255, 255, 255) if not dipbench.get_note_on(key.note_no - 21) else (255, 0, 0), Rect(key.x - white_key_width // 2, keyboard_top, white_key_width, white_key_height))
        # draw black keybed
        for key in [key for key in keyboard.keys[21:109] if key.black_key]:
            screen.fill((0, 0, 0), Rect(key.x - black_key_width // 2, keyboard_top, black_key_width, black_key_height))
            if dipbench.get_note_on(key.note_no - 21):
                screen.fill((255, 0, 0), Rect(key.x - black_key_width / 2 + line_width, keyboard_top, black_key_width - line_width * 2, black_key_height - line_width))

        # draw pitch summary text
        if dipbench.pitch_variation is not None and dipbench.pitch_checked > 0:
            volume_text = ''
            if dipbench.pitch_volume_std is not None:
                volume_text = f' Volume : standard deviation {dipbench.pitch_volume_std:.2f}dB.'
            latency_text = ''
            if dipbench.pitch_latency_std is not None:
                latency_text = f' Latency : average {dipbench.pitch_latency_average * 1000:.1f}ms, standard deviation {dipbench.pitch_latency_std * 1000:.1f}ms.'
            
            pitch_info_text = font.render(f'{dipbench.pitch_variation} / {dipbench.pitch_checked} waveforms for keys ({dipbench.pitch_variation * 100 / dipbench.pitch_checked:.2f}%). One waveform for {dipbench.pitch_checked / dipbench.pitch_variation:.2f} keys on average.{volume_text}{latency_text}', True, (255,128,192))
            screen.blit(pitch_info_text, [base_margin, energy_bottom - base_size * 7])

        # draw pitch variation
        if dipbench.pitch_correl is not None:
            for key in keyboard.keys[21:108]:
                x1 = key.normalized_x
                x2 = x1 + (key_width * 7) / 12
                height = base_size * 3
                point = []
                resolution = 6
                for i in range(resolution + 1):
                    point.append((x1 + (x2 - x1) * i / resolution, energy_bottom - math.sqrt(math.sin(i / resolution * 3.1415926)) * height))
                pygame.draw.lines(screen, np.array([255.0,255.0,255.0]) * dipbench.pitch_correl[key.note_no - 21]**2, False, point, line_width)

        # draw individual volume
        pygame.draw.line(screen, (96,96,96), (base_margin, energy_bottom - base_size * 4), (window_size[0] - base_margin, energy_bottom - base_size * 4), line_width)
        pygame.draw.line(screen, (96,96,96), (base_margin, energy_bottom + base_size * 4), (window_size[0] - base_margin, energy_bottom + base_size * 4), line_width)
        if dipbench.pitch_volume is not None and dipbench.pitch_volume_average is not None and len(dipbench.pitch_volume) > 1:
            for key in keyboard.keys[21:109]:
                if dipbench.pitch_volume[key.note_no - 21] is not None:
                    pygame.draw.line(screen, (0, 224, 0), (key.normalized_x, energy_bottom), (key.normalized_x, energy_bottom - (dipbench.pitch_volume[key.note_no - 21] - dipbench.pitch_volume_average) / 6.0 * base_size * 4), line_width * 2)

        # draw individual latency
        pygame.draw.line(screen, (128,128,128), (base_margin, window_size[1] - base_margin), (window_size[0] - base_margin, window_size[1] - base_margin), line_width)
        pygame.draw.line(screen, (96,96,96), (base_margin, window_size[1] - base_margin - 0.01 * base_size * 500), (window_size[0] - base_margin, window_size[1] - base_margin - 0.01 * base_size * 500), line_width)
        for key in keyboard.keys[21:109]:
            if dipbench.pitch_latency is not None and len(dipbench.pitch_latency) > key.note_no - 21 and dipbench.pitch_latency[key.note_no - 21] is not None:
                latency = dipbench.pitch_latency[key.note_no - 21]
                y = window_size[1] - base_margin - latency * base_size * 500
                pygame.draw.line(screen, (255,192,128), (key.normalized_x - base_size, y), (key.normalized_x + base_size, y), line_width * 2)

        # draw key center dot
        for key in keyboard.keys[21:109]:
            pygame.draw.circle(screen, dipbench.pitch_color[key.note_no - 21], (key.normalized_x + 1, energy_bottom), (base_size * 3) / 7, 0)

        # draw velocity summary
        if dipbench.velocity_layer is not None and dipbench.velocity_checked > 0:
            latency_text = ''
            if dipbench.velocity_volume is not None:
                volume = [vol for vol in dipbench.velocity_volume if vol is not None]
                if len(volume) > 0:
                    velocity = [i + 1 for i, vol in enumerate(dipbench.velocity_volume) if vol is not None]
                    spearman_corr, _ = spearmanr(velocity, volume)
                    volume_text = f' Volume : spearman corr {spearman_corr:.6f}.'
                
            if dipbench.velocity_latency_std is not None:
                latency_text = f' Latency : standard deviation {dipbench.velocity_latency_std * 1000:.1f}ms.'

            velocity_info_text = font.render(f'{dipbench.velocity_layer} / {dipbench.velocity_checked} waveforms for velocities ({dipbench.velocity_layer * 100 / dipbench.velocity_checked:.2f}%). One waveform for {dipbench.velocity_checked / dipbench.velocity_layer:.2f} velocity on average.{volume_text}{latency_text}', True, (255,128,192))
            screen.blit(velocity_info_text, [base_margin, energy_bottom - base_size * 11])

        # draw velocity variation
        for i in range(len(dipbench.velocity_color)):
            rect = Rect(velocity_left, velocity_bottom - (i + 1) * velocity_size, velocity_width, velocity_size)
            pygame.draw.rect(screen, dipbench.velocity_color[i], rect)

        # draw velocity latency
        pointlist = []
        pygame.draw.line(screen, (96,96,96), (velocity_left + 0.01 * base_size * 500, velocity_bottom - 127 * velocity_size), (velocity_left + 0.01 * base_size * 500, velocity_bottom), line_width)
        if dipbench.velocity_latency is not None:
            for i in range(127):
                if dipbench.velocity_latency[i] is not None:
                    x = int(velocity_left + dipbench.velocity_latency[i] * base_size * 500)
                    y = velocity_bottom - i * velocity_size - velocity_size // 2
                    pointlist.append((x,y))
        if len(pointlist) > 1:
            pygame.draw.lines(screen, (0,0,0), False, pointlist, base_size)
            pygame.draw.lines(screen, (255,192,128), False, pointlist, base_size // 2)

        # draw velocity volume
        pointlist = []
        if dipbench.velocity_volume is not None:
            velocity_min_volume = np.nanmin(np.array(dipbench.velocity_volume, dtype=float))
            velocity_max_volume = np.nanmax(np.array(dipbench.velocity_volume, dtype=float))
            if velocity_min_volume is not np.nan and velocity_max_volume is not np.nan and velocity_max_volume > velocity_min_volume:
                for i in range(127):
                    if dipbench.velocity_volume[i] is not None:
                        x = int(velocity_left + (dipbench.velocity_volume[i] - velocity_min_volume) / (velocity_max_volume - velocity_min_volume) * (velocity_width - 1))
                        y = velocity_bottom - i * velocity_size - velocity_size // 2
                        pointlist.append((x,y))
        if len(pointlist) > 1:
            pygame.draw.lines(screen, (0,0,0), False, pointlist, base_size)
            pygame.draw.lines(screen, (0,224,0), False, pointlist, base_size // 2)

        # draw velocity cursor
        velocity = []
        if dipbench.velocity >= 0:
            velocity.append(dipbench.velocity)
        if dipbench.realtime_velocity is not None:
            velocity.extend([v - 1 for i, v in enumerate(dipbench.realtime_velocity) if v > 0 and dipbench.realtime_note_on[i]])
        velocity.sort()
        last_velocity_y = window_size[1]
        for v in velocity:
            velocity_y = velocity_bottom - v * velocity_size - velocity_size // 2
            pointlist = [[velocity_left - base_size * 3, velocity_y - base_size],
                            [velocity_left - base_size * 3, velocity_y + base_size], 
                            [velocity_left - base_size * 1, velocity_y]]
            pygame.draw.polygon(screen, (255,255,255), pointlist)
            if velocity_y < last_velocity_y:
                velocity_text = font.render(f'{v + 1}', True, (255,255,255))
                screen.blit(velocity_text, [velocity_left - base_size * 4 - velocity_text.get_width(), velocity_y - velocity_text.get_height() // 2])
                last_velocity_y = velocity_y - base_margin

        # draw waveform
        if monitor_wave is not None:
            # display left ch wave form
            waveform = monitor_wave[:,0]

            if len(waveform) >= sample_length * 2:
                # display waveform
                waveform = waveform[sample_length:sample_length * 2]
                waveform_pointlist = np.column_stack((np.linspace(base_margin, window_size[0] // 2 - base_margin, sample_length), waveform / waveform_std * vector_scope_size + keyboard_top // 2))
                pygame.draw.lines(screen, (96, 96, 96), False, waveform_pointlist)

                # display vector scope
                monitor_wave = (-monitor_wave + np.stack([monitor_wave[:,1], -monitor_wave[:,0]], axis=1)) * 0.707
                pointlist = (monitor_wave[:sampling_freq // 2] - np.mean(monitor_wave)) / waveform_std * 0.8 * vector_scope_size + (window_size[0] // 4, keyboard_top // 2)
                pygame.draw.lines(screen, (0,224,0), False, pointlist)

            # display latency
            if display_latency is not None:
                latency_x = sampling_freq * display_latency / sample_length * (window_size[0] // 2 - base_margin * 2) + base_margin
                pygame.draw.line(screen, (255,192,128), (latency_x, keyboard_top // 2 - base_size * 31), (latency_x, keyboard_top // 2 + base_size * 31), line_width * 2)
                latency_text = font.render(f'{display_latency * 1000:.1f}ms latency', True, (255,192,128))
                screen.blit(latency_text, [latency_x + base_size, keyboard_top // 2 - base_size * 31])

            # display volume
            if display_volume is not None:
                volume_text = font.render(f'{display_volume:.1f}dB', True, (0,224,0))
                screen.blit(volume_text, [window_size[0] // 2 - base_margin - volume_text.get_width(), keyboard_top // 2 - base_size * 31])

        # prepare spectrum
        monitor_spectrum = None
        if dipbench.monitor_mode == 1 and dipbench.pitch_spectrums is not None and dipbench.tone >= 0 and dipbench.pitch_spectrums[dipbench.tone] is not None:
            monitor_spectrum = dipbench.pitch_spectrums[dipbench.tone]
        elif dipbench.monitor_mode == 2 and dipbench.velocity_spectrums is not None and dipbench.velocity >= 0 and dipbench.velocity_spectrums[dipbench.velocity] is not None:
            monitor_spectrum = dipbench.velocity_spectrums[dipbench.velocity]
        elif dipbench.monitor_mode == 3 and dipbench.realtime_spectrum is not None:
            monitor_spectrum = dipbench.realtime_spectrum
        if monitor_spectrum is not None and len(monitor_spectrum) > 0:
            spectrum_max = monitor_spectrum.max()
            if spectrum_max < min_level:
                spectrum_max = min_level
            monitor_spectrum = np.clip(monitor_spectrum - spectrum_max + 96, 0, 96)
            x = np.linspace(window_size[0] // 2 + base_margin, velocity_left - base_margin * 3, spectrum_size // 4)
            y = velocity_bottom - monitor_spectrum[:spectrum_size // 4] / 96 * (velocity_bottom - (keyboard_top // 2 - base_size * 31))
            pointlist = np.stack([x, y], axis=1)
            pygame.draw.lines(screen, (0,224,0), False, pointlist)

        # draw
        pygame.display.flip()

        # wait
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

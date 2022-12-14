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

app_title = 'DiP-Bench : Digital Piano Benchmark'

# settings
pitch_direction_threshohld = 0.9
velocity_direction_threshohld = 0.995

# size
screen_size = (1920, 1080)
# screen_size = (2560, 1440)
# screen_size = (3840, 2160)
line_width = screen_size[1] // 1081 + 1
base_size = screen_size[0] // 240 # 8 pixel in 1920 x 1080
base_margin = base_size * 3
vector_scope_size = screen_size[1] / 16 # sd = height/32

# keyboard size
keyboard_margin_x = base_size * 3
key_width = (screen_size[0] - keyboard_margin_x * 2) / 52
keyboard_top = base_size * 100
energy_bottom = keyboard_top - keyboard_margin_x
white_key_width = round(key_width * 33 / 36) #22.5 / 23.5
white_key_height = round(key_width * 150 / 23.5)
black_key_width = round(key_width * 23 / 36) #15 / 23.5
black_key_height = round(key_width * 100 / 23.5)

# velocity size
velocity_width = base_size * 16
velocity_size = base_size // 2
velocity_bottom = keyboard_top - base_size * 14 


sampling_freq = 48000
sample_length = 2048

correl_size = 2048
shift_range = 1024

realtime_analysis_length = 8192
spectrum_size = 8192


class Keyboard:
    keys = []

    def __init__(self):
        class Key:
            pass
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

    # measurement results in velocity direction
    velocity = -1
    velocity_waveforms = None
    velocity_spectrums = None
    duplication_in_velocity = None
    velocity_correl = None
    velocity_color = [(64,64,64)] * 127
    velocity_layer = None
    velocity_checked = None
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
            self.audio_inputs.append(self.audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        
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
            if self.audio_inputs[0] == self.audio.get_device_info_by_host_api_device_index(0, i).get('name'):
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

    def __get_spectrum(self, waveform):
        waveform = np.sum(waveform,axis=1)
        if len(waveform) < spectrum_size:
            waveform = waveform * np.hanning(len(waveform))
            waveform = np.pad(waveform, ((0, spectrum_size - len(waveform))))
        else:
            waveform = waveform[:spectrum_size]
            waveform = waveform * np.hanning(spectrum_size)
        spectrum = np.log(np.abs(np.fft.fft(waveform))) / np.log(2) * 6.0 # in dB
        return np.clip(spectrum - spectrum.max() + 96, 0, 96)

    def __check_duplication(self, waveform1, waveform2, nextnote=False):
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
        print(shift_pos - shift_range, max_correl)

        return max_correl

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

                    duplication = False
                    if self.tone > 0:
                        max_correl = self.__check_duplication(self.pitch_waveforms[self.tone - 1], self.pitch_waveforms[self.tone], True)
                        self.pitch_correl[self.tone - 1] = max_correl
                        duplication = max_correl > pitch_direction_threshohld
                    self.duplication_in_pitch[self.tone] = duplication
                    if not duplication:
                        self.last_hue = (self.last_hue + 0.25 + random.random() * 0.5) % 1.0
                        self.pitch_variation = self.pitch_variation + 1
                    self.pitch_color[self.tone] = colorsys.hsv_to_rgb(self.last_hue, 0.8, 255.0)
                    self.pitch_checked = self.pitch_checked + 1

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

                    duplication = False
                    if self.velocity > 0:
                        max_correl = self.__check_duplication(self.velocity_waveforms[self.velocity - 1], self.velocity_waveforms[self.velocity])
                        # max_correl = np.dot(self.velocity_spectrums[self.velocity - 1], self.velocity_spectrums[self.velocity]) / ((np.linalg.norm(self.velocity_spectrums[self.velocity - 1]) * np.linalg.norm(self.velocity_spectrums[self.velocity])))
                        self.velocity_correl[self.velocity - 1] = max_correl
                        duplication = max_correl > velocity_direction_threshohld
                        if max_correl > self.max_correl:
                            self.max_correl = max_correl    
                    self.duplication_in_velocity[self.velocity] = duplication and self.max_correl > velocity_direction_threshohld
                    if self.max_correl > velocity_direction_threshohld:
                        if not duplication or self.velocity_layer == 0:
                            self.last_hue = (self.last_hue + 0.25 + random.random() * 0.5) % 1.0
                            self.velocity_layer = self.velocity_layer + 1
                        self.velocity_color[self.velocity] = colorsys.hsv_to_rgb(self.last_hue, 0.8, 255.0)
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
                    if midi_event[0][0] & 0xf0 == 0x90: # note on
                        self.realtime_note_on[key] = self.realtime_key_on[key] = True
                        self.realtime_velocity[key] = midi_event[0][2]
                    if midi_event[0][0] & 0xf0 == 0x80: # note off
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
    # initialization
    pygame.mixer.pre_init(frequency=sampling_freq, size=-16, channels=2)
    pygame.init()
    pygame.display.set_icon(pygame.image.load(resource_path('icon_128x128.png')))
    screen = pygame.display.set_mode(screen_size)
    full_screen = False

    pygame.display.set_caption(app_title)

    dipbench = DipBench()
    keyboard = Keyboard()

    # main_loop
    terminated = False

    font = pygame.font.Font(None, base_size * 4)
    app_title_text = font.render(app_title + '  /  Frieve 2022', True, (255,255,255))
    help_text = font.render('[A] Chalge Audio Input, [I] Change MIDI Input, [O] Change MIDI Output, [P] Measure pitch variation, [V] Measure velocity layer, [R] Real-time mode, [Q] Quit', True, (128,192,255))
    help_text2 = font.render('[ESC] Abort', True, (128,192,255))
    while (not terminated):
        screen.fill((0,0,0))

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
                    if full_screen:
                        screen = pygame.display.set_mode(screen_size, FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode(screen_size)
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
                    if event.pos[0] >= screen_size[0] - base_margin - velocity_width and event.pos[0] < screen_size[0] - base_margin and event.pos[1] >= velocity_bottom - 127 * velocity_size and event.pos[1] < velocity_bottom:
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
        screen.blit(device_info_text, [base_margin, base_margin + base_size * 6])
        screen.blit(help_text if dipbench.mode == 0 else help_text2, [base_margin, base_margin + base_size * 12])
        if dipbench.last_error is not None:
            error_text = font.render(dipbench.last_error, True, (255,0,0))
            screen.blit(error_text, [base_margin, base_margin + base_size * 18])

        # draw white keybed
        for key in [key for key in keyboard.keys[21:109] if not key.black_key]:
            screen.fill((255, 255, 255) if not dipbench.get_note_on(key.note_no - 21) else (255, 0, 0), Rect(key.x - white_key_width // 2, keyboard_top, white_key_width, white_key_height))
        # draw black keybed
        for key in [key for key in keyboard.keys[21:109] if key.black_key]:
            screen.fill((0, 0, 0), Rect(key.x - black_key_width // 2, keyboard_top, black_key_width, black_key_height))
            if dipbench.get_note_on(key.note_no - 21):
                screen.fill((255, 0, 0), Rect(key.x - black_key_width / 2 + line_width, keyboard_top, black_key_width - line_width * 2, black_key_height - line_width))

        # draw pitch variation
        if dipbench.pitch_variation is not None and dipbench.pitch_checked > 0:
            pitch_info_text = font.render(f'{dipbench.pitch_variation} / {dipbench.pitch_checked} waveform are recorded ({dipbench.pitch_variation * 100 / dipbench.pitch_checked:.2f}%). One waveform for {dipbench.pitch_checked / dipbench.pitch_variation:.2f} keys on average.', True, (255,192,128))
            screen.blit(pitch_info_text, [base_margin, energy_bottom - base_size * 7])
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
        # draw key center dot
        for key in keyboard.keys[21:109]:
            pygame.draw.circle(screen, dipbench.pitch_color[key.note_no - 21], (key.normalized_x + 1, energy_bottom + 1), (base_size * 2) // 3, 0)

        # draw velocity layer
        if dipbench.velocity_layer is not None and dipbench.velocity_checked > 0:
            velocity_info_text = font.render(f'{dipbench.velocity_layer} / {dipbench.velocity_checked} waveform are recorded ({dipbench.velocity_layer * 100 / dipbench.velocity_checked:.2f}%). One waveform for {dipbench.velocity_checked / dipbench.velocity_layer:.2f} velocity on average.', True, (255,192,128))
            screen.blit(velocity_info_text, [base_margin, energy_bottom - base_size * 12])
        # draw velocity variation
        for i in range(len(dipbench.velocity_color)):
            rect = Rect(screen_size[0] - base_margin - velocity_width, velocity_bottom - (i + 1) * velocity_size, velocity_width, velocity_size)
            pygame.draw.rect(screen, dipbench.velocity_color[i], rect)
        # draw velocity cursor
        velocity = []
        if dipbench.velocity >= 0:
            velocity.append(dipbench.velocity)
        if dipbench.realtime_velocity is not None:
            velocity.extend([v - 1 for i, v in enumerate(dipbench.realtime_velocity) if v > 0 and dipbench.realtime_note_on[i]])
        velocity.sort()
        last_velocity_y = screen_size[1]
        for v in velocity:
            velocity_y = velocity_bottom - v * velocity_size - velocity_size // 2
            pointlist = [[screen_size[0] - base_margin - velocity_width - base_size * 3, velocity_y - base_size],
                            [screen_size[0] - base_margin - velocity_width - base_size * 3, velocity_y + base_size], 
                            [screen_size[0] - base_margin - velocity_width - base_size * 1, velocity_y]]
            pygame.draw.polygon(screen, (255,255,255), pointlist)
            if velocity_y < last_velocity_y:
                velocity_text = font.render(f'{v + 1}', True, (255,255,255))
                screen.blit(velocity_text, [screen_size[0] - base_margin - velocity_width - base_size * 4 - velocity_text.get_width(), velocity_y - velocity_text.get_height() // 2])
                last_velocity_y = velocity_y - base_margin

        # prepare waveform
        monitor_wave = None
        if dipbench.monitor_mode == 1 and dipbench.pitch_waveforms is not None and dipbench.tone >= 0 and dipbench.pitch_waveforms[dipbench.tone] is not None:
            if len(dipbench.pitch_waveforms[dipbench.tone]) >= sample_length * 2:
                monitor_wave = dipbench.pitch_waveforms[dipbench.tone]
                std = np.std(monitor_wave[sample_length:sample_length * 2])
        elif dipbench.monitor_mode == 2 and dipbench.velocity_waveforms is not None and dipbench.velocity >= 0 and dipbench.velocity_waveforms[dipbench.velocity] is not None:
            if len(dipbench.velocity_waveforms[dipbench.velocity]) >= sample_length * 2:
                monitor_wave = dipbench.velocity_waveforms[dipbench.velocity]
                std = np.std(monitor_wave[sample_length:sample_length * 2])
        elif dipbench.monitor_mode == 3 and dipbench.realtime_waveform is not None:
            monitor_wave = dipbench.realtime_waveform
            std = np.std(monitor_wave)
        if monitor_wave is not None:
            # display vector scope
            if std < 16.0:
                std = 16.0
            monitor_wave = (-monitor_wave + np.stack([monitor_wave[:,1], -monitor_wave[:,0]], axis=1)) * 0.707
            pointlist = (monitor_wave[:sampling_freq // 2] - np.mean(monitor_wave[sample_length:sample_length * 2])) / std * vector_scope_size + (screen_size[0] // 4, keyboard_top // 2)
            pygame.draw.lines(screen, (0,224,0), False, pointlist)

        # prepare spectrum
        monitor_spectrum = None
        if dipbench.monitor_mode == 1 and dipbench.pitch_spectrums is not None and dipbench.tone >= 0 and dipbench.pitch_spectrums[dipbench.tone] is not None:
            monitor_spectrum = dipbench.pitch_spectrums[dipbench.tone]
        elif dipbench.monitor_mode == 2 and dipbench.velocity_spectrums is not None and dipbench.velocity >= 0 and dipbench.velocity_spectrums[dipbench.velocity] is not None:
            monitor_spectrum = dipbench.velocity_spectrums[dipbench.velocity]
        elif dipbench.monitor_mode == 3 and dipbench.realtime_spectrum is not None:
            monitor_spectrum = dipbench.realtime_spectrum
        if monitor_spectrum is not None:
            x = np.linspace(screen_size[0] // 2 + base_margin, screen_size[0] - base_margin * 4 - velocity_width, spectrum_size // 4)
            y = velocity_bottom - monitor_spectrum[:spectrum_size // 4] / 96 * screen_size[1] / 2
            pointlist = np.stack([x, y], axis=1)
            pygame.draw.lines(screen, (0,224,0), False, pointlist)

        # draw
        pygame.display.update()

        # wait
        pygame.time.wait(1)

    pygame.quit()


if __name__ == "__main__":
    main()

import numpy
import numpy as np
import pyaudio
import struct
from scipy.fftpack import fft
import sys
import time
import pygame
import math
from tkinter import *
import threading

import http.server  # Our http server handler for http requests
import socketserver  # Establish the TCP Socket connections

PORT = 9000

CHANGE_FREQ = 10

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global at_freq_r, at_freq_g, at_freq_b
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(str(at_freq_r*255**2+at_freq_g*255+at_freq_b))



(WIDTH, HEIGHT) = (500, 500)


def activate (x):
    return 1 / (1 + math.exp(-(x-0.5)))

def avgr(l):
    return sum(l)/len(l)


class AudioStream:

    def __init__(self):

        # stream constants
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False

        # stream object
        self.p = pyaudio.PyAudio()

        for i in range(self.p.get_device_count()):
            if "stereo" in self.p.get_device_info_by_index(i)['name'].lower() or "pulse" in self.p.get_device_info_by_index(i)['name'].lower():
                self.device = i
                break

        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK,
            input_device_index = self.device
        )
        self.init_plots()
        self.start_plot()

    def init_plots(self):
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.flip()


    def start_plot(self):
        global at_freq_r, at_freq_g, at_freq_b
        global master
        global red_color_scale
        global blue_color_scale
        global green_color_scale
        global global_color_scale
        global decayScale

        last_r = 0
        last_g = 0
        last_b = 0


        vol_size = 100
        combine_vol_chunks = 2 ** 0
        print('stream started')
        frame_count = 0
        start_time = time.time()
        vol_aplifier = 15
        voL_scale = 1

        last_frames = np.ones((vol_size, self.CHUNK))

        while not self.pause:
            master.update()
            COLOR_SCALER_R = red_color_scale.get()
            COLOR_SCALER_G = green_color_scale.get()
            COLOR_SCALER_B = blue_color_scale.get()
            GLOBAL_COLOR_SCALER = global_color_scale.get()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
            data = self.stream.read(self.CHUNK)
            data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
            data_np = np.array(data_int).astype('b')[::2] + 128

            # compute FFT and update line
            yf = fft(data_int)

            # compute volatility
            last_frames[frame_count % vol_size] = data_np[:]


            # make a fast version of volatility calculation
            volatility = last_frames.std(axis=0) / np.mean(last_frames, axis=0)

            for i in range(len(volatility) // combine_vol_chunks):
                volatility[i * combine_vol_chunks:(i + 1) * combine_vol_chunks] = [
                    np.average(volatility[i * combine_vol_chunks:( i + 1) * combine_vol_chunks])
                ] * combine_vol_chunks


            volatility /= np.max(volatility)

            volatility = np.array(volatility) ** vol_aplifier
            volatility = np.array(volatility) * voL_scale
            volatility.resize(self.CHUNK)

            volatility_a:list = volatility.tolist()
            freq_r = volatility_a.index(max(volatility))
            freq_g = volatility_a.index(min(volatility))
            freq_b = numpy.argsort(volatility_a)[len(volatility_a)//2]

            #at_freq_r = data[freq_r] / np.average(last_frames[:, freq_r])  * 2 ** (COLOR_SCALER_R/15) * GLOBAL_COLOR_SCALER/100
            #at_freq_g = data[freq_g] / np.average(last_frames[:, freq_g])  * 2 ** (COLOR_SCALER_G/15) * GLOBAL_COLOR_SCALER/100
            #at_freq_b = data[freq_b] / np.average(last_frames[:, freq_b])  * 2 ** (COLOR_SCALER_B/15) * GLOBAL_COLOR_SCALER/100

            at_freq_r = (data[freq_r] / (last_frames[(frame_count-1) % vol_size, freq_r] + 1))  * 2 ** (COLOR_SCALER_R/15) * GLOBAL_COLOR_SCALER/100
            at_freq_g = (data[freq_g] / (last_frames[(frame_count-1) % vol_size, freq_g] + 1))  * 2 ** (COLOR_SCALER_G/15) * GLOBAL_COLOR_SCALER/100
            at_freq_b = (data[freq_b] / (last_frames[(frame_count-1) % vol_size, freq_b] + 1))  * 2 ** (COLOR_SCALER_B/15) * GLOBAL_COLOR_SCALER/100


            at_freq_r = int(activate(at_freq_r / 255) * 255)
            at_freq_g = int(activate(at_freq_g / 255) * 255)
            at_freq_b = int(activate(at_freq_b / 255) * 255)

            last_r = max(at_freq_r, last_r)
            last_g = max(at_freq_g, last_g)
            last_b = max(at_freq_b, last_b)

            #print(at_freq_r, ", " , at_freq_g, ", ", at_freq_b)
            self.window.fill((last_r, last_g, last_b))

            last_r -= decayScale.get()
            last_g -= decayScale.get()
            last_b -= decayScale.get()

            pygame.display.update()
            pygame.time.wait(int(1000/CHANGE_FREQ))

            save_config()

            frame_count+=1

        else:
            self.fr = frame_count / (time.time() - start_time)
            print('average frame rate = {:.0f} FPS'.format(self.fr))
            self.exit_app()

    def exit_app(self):
        print('stream closed')
        self.p.close(self.stream)



COLOR_SCALER_R = 10
COLOR_SCALER_G = 0
COLOR_SCALER_B = 28
GLOBAL_COLOR_SCALER = 100
def createSilderWindow():
    global red_freq_scale
    global blue_freq_scale
    global green_freq_scale
    global red_color_scale
    global blue_color_scale
    global green_color_scale
    global master
    global freqlen
    global global_color_scale
    global decayScale

    master = Tk()

    T = Text(master, height=1)
    T.pack()
    T.insert(END, "Decay")

    decayScale = Scale(master, from_=0, to=100, orient=HORIZONTAL)
    decayScale.set(COLOR_SCALER_R)
    decayScale.pack()

    T = Text(master, height=1)
    T.pack()
    T.insert(END, "Farb-Skala [R;G;B;*]")

    red_color_scale = Scale(master, from_=0, to=200, orient=HORIZONTAL)
    red_color_scale.set(5)
    red_color_scale.pack()

    green_color_scale = Scale(master, from_=0, to=200, orient=HORIZONTAL)
    green_color_scale.set(COLOR_SCALER_G)
    green_color_scale.pack()

    blue_color_scale = Scale(master, from_=0, to=200, orient=HORIZONTAL)
    blue_color_scale.set(COLOR_SCALER_B)
    blue_color_scale.pack()

    global_color_scale = Scale(master, from_=0, to=1000, orient=HORIZONTAL)
    global_color_scale.set(GLOBAL_COLOR_SCALER)
    global_color_scale.pack()

def create_server():
    Handler = MyHttpRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("Http Server Serving at port", PORT)
        httpd.serve_forever()

def load_config():
    global red_color_scale
    global green_color_scale
    global blue_color_scale
    global global_color_scale
    global decayScale
    try:
        with open("config.txt", "r") as f:
            lines = f.readlines()
            red_color_scale.set(int(lines[0]))
            green_color_scale.set(int(lines[1]))
            blue_color_scale.set(int(lines[2]))
            decayScale.set(int(lines[3]))
            global_color_scale.set(int(lines[4]))
    except:
        print("no config file found")
        save_config()

def save_config():
    global red_color_scale
    global green_color_scale
    global blue_color_scale
    global global_color_scale
    global decayScale
    with open("config.txt", "w+") as f:
        f.write(str(red_color_scale.get()) + "\n")
        f.write(str(green_color_scale.get()) + "\n")
        f.write(str(blue_color_scale.get()) + "\n")
        f.write(str(decayScale.get()) + "\n")
        f.write(str(global_color_scale.get()) + "\n")


if __name__ == '__main__':
    createSilderWindow()
    load_config()
    threading.Thread(target=create_server).start()
    print("kokoloress")
    AudioStream()

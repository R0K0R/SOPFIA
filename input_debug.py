import sys
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# Configuration
SAMPLE_RATE = 48000
BLOCK_SIZE = 1024
WINDOW = np.hanning(BLOCK_SIZE)

class FFTVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.stream = sd.InputStream(channels=1,
                                     samplerate=SAMPLE_RATE,
                                     blocksize=BLOCK_SIZE,
                                     callback=self.audio_callback)
        self.stream.start()

    def initUI(self):
        self.setWindowTitle("Real-Time Microphone FFT")
        self.setGeometry(100, 100, 800, 400)

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        self.plot_widget.setLabel('left', 'Magnitude')

        self.curve = self.plot_widget.plot(pen='y')
        self.x_data = np.fft.rfftfreq(BLOCK_SIZE, d=1.0/SAMPLE_RATE)

        # Timer for periodic updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)  # ~33 FPS

        self.fft_magnitude = np.zeros_like(self.x_data)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        mono = indata[:, 0]
        windowed = mono * WINDOW
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        self.fft_magnitude = magnitude

    def update_plot(self):
        self.curve.setData(self.x_data, self.fft_magnitude)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = FFTVisualizer()
    window.show()
    sys.exit(app.exec_())

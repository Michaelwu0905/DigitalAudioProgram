import sys
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLineEdit, QLabel, QHBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from scipy.io import wavfile
import sounddevice as sd
from scipy.signal import butter, filtfilt, freqz
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class AudioProcessingApp(QMainWindow):
    def __init__(self):
        """构造函数"""
        super().__init__()
        self.setWindowTitle("音频处理软件")
        self.setGeometry(100, 100, 800, 600)

        # 初始化变量
        self.media_player = QMediaPlayer()
        self.original_signal = None
        self.noisy_signal = None
        self.filtered_signal = None
        self.sample_rate = None

        # 创建UI
        self.initUI()

    def initUI(self):
        """创建按钮"""
        original_button = QPushButton('原始波形', self)
        noise_button = QPushButton('正弦噪声', self)
        noisy_button = QPushButton('带噪信号', self)
        lowpass_button = QPushButton('低通滤波器', self)
        bandpass_button = QPushButton('带通滤波器', self)

        # 创建频率输入框
        freq_label = QLabel("频率:", self)
        self.freq_input = QLineEdit(self)

        # 连接信号与槽
        original_button.clicked.connect(self.openOriginalAudio)
        noise_button.clicked.connect(self.playSineNoise)
        noisy_button.clicked.connect(self.playNoisySignal)
        lowpass_button.clicked.connect(self.applyLowpassFilter)
        bandpass_button.clicked.connect(self.applyBandpassFilter)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(original_button)
        layout.addWidget(noise_button)
        layout.addWidget(noisy_button)
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(lowpass_button)
        filter_layout.addWidget(bandpass_button)
        filter_layout.addWidget(freq_label)
        filter_layout.addWidget(self.freq_input)
        layout.addLayout(filter_layout)

        # 设定中心小部件
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 设置波形显示区
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def openOriginalAudio(self):
        """实现原始波形按钮的功能"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav)")
        if file_name:
            self.sample_rate, self.original_signal = wavfile.read(file_name)
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_name)))
            self.media_player.play()
            self.plotWaveformAndSpectrum(self.original_signal, self.sample_rate, '原始音频')

    def plotWaveformAndSpectrum(self, data, sample_rate, title=''):
        """绘制波形和频谱"""
        time = np.arange(data.shape[0]) / sample_rate

        # 清除之前的图像
        self.figure.clear()

        # 绘制波形
        ax_waveform = self.figure.add_subplot(2, 1, 1)
        ax_waveform.plot(time, data)
        ax_waveform.set_xlabel('Time [s]')
        ax_waveform.set_ylabel('Amplitude')
        ax_waveform.set_title('Wave_Form')

        # 绘制频谱
        ax_spectrum = self.figure.add_subplot(2, 1, 2)
        spectrum = np.fft.fft(data)
        freq = np.fft.fftfreq(len(spectrum), 1/sample_rate)
        ax_spectrum.plot(freq[:len(spectrum)//2], np.abs(spectrum[:len(spectrum)//2]))
        ax_spectrum.set_xlabel('Frequency [Hz]')
        ax_spectrum.set_ylabel('Magnitude')
        ax_spectrum.set_title('Spectrum')

        # 调整子图布局
        self.figure.tight_layout()

        # 更新图像
        self.canvas.draw()

    def playSineNoise(self):
        """播放正弦噪声"""
        duration = len(self.original_signal) / self.sample_rate  # 噪声持续时间与原始音频相同
        freq = 440  # A4 note
        t = np.linspace(0, duration, len(self.original_signal), endpoint=False)
        noise = 0.001 * np.sin(2 * np.pi * freq * t)  # 降低噪声幅度
        sd.play(noise, self.sample_rate)
        sd.wait()
        self.plotWaveformAndSpectrum(noise, self.sample_rate, 'sine noise')

    def playNoisySignal(self):
        """播放带噪音频"""
        if self.original_signal is not None and self.sample_rate is not None:
            duration = len(self.original_signal) / self.sample_rate
            freq = 440
            t = np.linspace(0, duration, len(self.original_signal), endpoint=False)
            noise = 0.001 * np.sin(2 * np.pi * freq * t)  # 降低噪声幅度
            self.noisy_signal = self.original_signal + noise
            sd.play(self.noisy_signal, self.sample_rate)
            sd.wait()
            self.plotWaveformAndSpectrum(self.noisy_signal, self.sample_rate, '带噪信号')

    def applyLowpassFilter(self):
        """应用低通滤波"""
        cutoff_freq = float(self.freq_input.text())
        if self.noisy_signal is not None and self.sample_rate is not None:
            nyquist = 0.5 * self.sample_rate
            normal_cutoff = cutoff_freq / nyquist
            self.b, self.a = butter(5, normal_cutoff, btype='low', analog=False)
            self.filtered_signal = filtfilt(self.b, self.a, self.noisy_signal)
            sd.play(self.filtered_signal, self.sample_rate)
            sd.wait()
            self.plotFilteredSignal('Low_Pass Filter')

    def applyBandpassFilter(self):
        """带通滤波器"""
        center_freq = float(self.freq_input.text())
        if self.noisy_signal is not None and self.sample_rate is not None:
            nyquist = 0.5 * self.sample_rate
            freq_range = [center_freq - 100, center_freq + 100]
            normal_freq_range = [freq / nyquist for freq in freq_range]
            self.b, self.a = butter(5, normal_freq_range, btype='band', analog=False)
            self.filtered_signal = filtfilt(self.b, self.a, self.noisy_signal)
            sd.play(self.filtered_signal, self.sample_rate)
            sd.wait()
            self.plotFilteredSignal('Band_Pass Filter')

    def plotFilteredSignal(self, filter_type):
        """绘制滤波前后的波形和频谱"""
        # 清除之前的图像
        self.figure.clear()

        # 绘制滤波前的波形
        ax_before_waveform = self.figure.add_subplot(2, 2, 1)
        ax_before_waveform.plot(self.noisy_signal)
        ax_before_waveform.set_xlabel('Time [s]')
        ax_before_waveform.set_ylabel('Amplitude')
        ax_before_waveform.set_title('Wave before filter')
        # 绘制滤波前的频谱
        ax_before_spectrum = self.figure.add_subplot(2, 2, 2)
        spectrum_before = np.fft.fft(self.noisy_signal)
        freq = np.fft.fftfreq(len(spectrum_before), 1/self.sample_rate)
        ax_before_spectrum.plot(freq[:len(spectrum_before)//2], np.abs(spectrum_before[:len(spectrum_before)//2]))
        ax_before_spectrum.set_xlabel('Frequency [Hz]')
        ax_before_spectrum.set_ylabel('Magnitude')
        ax_before_spectrum.set_title('Spectrum before filter')

        # 绘制滤波后的波形
        ax_after_waveform = self.figure.add_subplot(2, 2, 3)
        ax_after_waveform.plot(self.filtered_signal)
        ax_after_waveform.set_xlabel('Time [s]')
        ax_after_waveform.set_ylabel('Amplitude')
        ax_after_waveform.set_title(f'Wave after {filter_type}')

        # 绘制滤波后的频谱
        ax_after_spectrum = self.figure.add_subplot(2, 2, 4)
        spectrum_after = np.fft.fft(self.filtered_signal)
        ax_after_spectrum.plot(freq[:len(spectrum_after)//2], np.abs(spectrum_after[:len(spectrum_after)//2]))
        ax_after_spectrum.set_xlabel('Frequency [Hz]')
        ax_after_spectrum.set_ylabel('Magnitude')
        ax_after_spectrum.set_title(f'Spectrum After {filter_type}')

        # # 绘制滤波器的频谱波形
        # w, h = freqz(self.b, self.a)
        # ax_filter = self.figure.add_subplot(2, 2, 4)
        # ax_filter.plot(w, 20 * np.log10(abs(h)), 'r')
        # ax_filter.set_xlabel('Frequency [rad/sample]')
        # ax_filter.set_ylabel('Amplitude [dB]')
        # ax_filter.set_title('Filter Spectrum')

        # 调整子图布局
        self.figure.tight_layout()

        # 更新图像
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioProcessingApp()
    window.show()
    sys.exit(app.exec_())
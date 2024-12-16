clc;
clear;
% close all;

%Read the samples from the audio file
[inputSound, sampleRate]=audioread('test_9_16_2.m4a');
sampleTimes = (1:length(inputSound))*(1/sampleRate);

%Play sound
% sound(inputSound, sampleRate);

%Plot the sound
figure;
plot(sampleTimes, inputSound);
ax = gca;
F_size = 20;
ax.FontSize = F_size;
xlabel('Time (sec)', 'FontSize', F_size);
ylabel('Amplitude', 'FontSize', F_size);

%Plot spectrogram
Window = 1000;
Overlap = 900;
NumFFT = 5000;
figure; spectrogram(inputSound, Window, Overlap, NumFFT, sampleRate, 'yaxis');
ax = gca;
F_size = 20;
ax.FontSize = F_size;
xlabel('Time (sec)', 'FontSize', F_size);
ylabel('Frequency (kHz)', 'FontSize', F_size);
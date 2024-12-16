clc;
clear;
close all;
% function fft_test()
freq = 200;
sampleRate = 10000;
time_ticks = 0:1/sampleRate:1;

realSignal = sin(2*pi*freq*time_ticks); 
plot(realSignal);
% xlim([0 500]);

plot_fft(realSignal, sampleRate); 

complexSignal = 	exp(j*2*pi*freq*time_ticks);
plot_fft(complexSignal, sampleRate);
% ylim([0 6000]);
% xlim([0 500]);
% end

set(gca,'LooseInset',get(gca,'TightInset'))
f = figure;
f.Position(3:4) = [1331 331];
%%% draw the pic of corn spectra
load('dataset/corn.mat');
x = m5spec.data;
wave_length = m5spec.axisscale{2, 1};
preprocess;
subplot(1, 4, 1)
plot(wave_length(1, 1:end-1), x');
xlim([wave_length(1) wave_length(end)]);
xlabel('Wavelength(nm)');
ylabel('Absorbance');
clear

%%% draw the pic of Marzipan spectra
load('dataset/marzipan.mat');
x = NIRS1;
wave_length = NIRS1_axis;
preprocess;
subplot(1, 4, 2)
plot(wave_length(1, 1:end-1), x');
xlim([wave_length(1) wave_length(end)]);
xlabel('Wavelength(nm)');
ylabel('Absorbance');
clear

%%% draw the pic of Marzipan spectra
load('dataset/soil.mat');
x = soil.data;
wave_length = soil.axisscale{2, 1};
preprocess;
subplot(1, 4, 3)
plot(wave_length(1, 1:end-1), x');
xlim([wave_length(1) wave_length(end)]);
xlabel('Wavelength(nm)');
ylabel('Absorbance');
clear

% draw the pic of Mango spectra
load('dataset/mango/mango_preprocessed.mat');
wave_length = 687: 3: 990;
subplot(1, 4, 4)
plot(wave_length, x');
xlim([wave_length(1) wave_length(end)]);
xlabel('Wavelength(nm)');
ylabel('Signal intensity');
clear
%% x preprocessing
clear;
load('dataset/mango/mango_origin.mat')
x = x';
x = sgolayfilt(x,2,17);
x =diff(x);
max_x=max(max(x));
min_x=min(min(x));
x=(x-min_x)/(max_x-min_x);
x = x';
y = y';
min_y = min(min(y));
max_y = max(max(y));
y = (y-min_y)/(max_y-min_y);
save('dataset/mango/mango_preprocessed.mat')

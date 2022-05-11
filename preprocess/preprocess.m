%% x preprocessing
x = x';
x = sgolayfilt(x,2,17);
x =diff(x);
max_x=max(max(x));
min_x=min(min(x));
x=(x-min_x)/(max_x-min_x);
x = x';

clear all
%Read image
blur = im2double(imread('blur.jpg'));
figure(1);
imshow(blur);
title('Blur Image');

%Simulate a motion blur
LEN = 21;
THETA = 11;
PSF = fspecial('motion', LEN, THETA);
%Restore the blurred image
wnr1 = deconvwnr(blur, PSF, 0.1);
figure(2);
imshow(wnr1);
title('Restored Image');
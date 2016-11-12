clear all
%读取模糊图像
blur = im2double(imread('blur.jpg'));
figure(1);imshow(blur);
title('Blur Image');
%运动模糊卷积核
len = 21;
theta = 11;
h = fspecial('motion', len, theta);
%维纳滤波复原，信噪比设为0.02
res = deconvwnr(blur, h, 0.02);
figure(2);imshow(res);
title('Restored Image');
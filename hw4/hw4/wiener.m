clear all
%��ȡģ��ͼ��
blur = im2double(imread('blur.jpg'));
figure(1);imshow(blur);
title('Blur Image');
%�˶�ģ�������
len = 21;
theta = 11;
h = fspecial('motion', len, theta);
%ά���˲���ԭ���������Ϊ0.02
res = deconvwnr(blur, h, 0.02);
figure(2);imshow(res);
title('Restored Image');
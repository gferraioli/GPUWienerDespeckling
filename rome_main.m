% Main for the experiment over Rome Simulated data 
% Intensity format and L=1 for simulated images
% This demo shows results for EWF filter
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (c) 2019 University of Naples "Parthenope".
% All rights reserved.
% this software should be used, reproduced and modified only for informational and nonprofit purposes.
% 
% By downloading and/or using any of these files, you implicitly agree to all the
% terms of the license, as specified in the document LICENSE.txt (included in this package)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;
load rome_clean;
load rome_noisy;

tic,
EWF_rome = wiener_gpu(z,20);
toc;

% display 
figure(1); 
subplot(1,3,1); imagesc(log10(x),[3 5.5]); title('clean'), colormap(gray), axis image, axis off
subplot(1,3,2); imshow(log10(z),[3 5.5]); title('noisy'), colormap(gray), axis image, axis off 
subplot(1,3,3); imshow(log10(EWF_rome),[3 5.5]); title('filtered'), colormap(gray), axis image, axis off

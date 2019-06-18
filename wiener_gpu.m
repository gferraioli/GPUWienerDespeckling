function [im_filt] = wiener_gpu(im_noisy, alpha_max)

%   A new developed frequency domain-based approach properly designed in
%   GPU presented as an Enhancement of the classical Wiener Filter that takes 
%   into account the local characteristics of the image. This technique is able
%   to provide effective results within a very limited processing time as 
%   described in "Fast GPU-Based Algorithm For Despeckling SAR Data filter", 
%   written by B. Kanoun, G. Ferraioli, V. Pascazio and G. Schirinzi,
%   Remote Sensing, no. in press ,(2019). 

%   Please refer to this paper for a more detailed description of the algorithm.
%
%   im_filt = wiener_gpu(im_noisy, alpha_max)
%
%       ARGUMENT DESCRIPTION:
%               im_noisy  - Noisy image (in intensity format) 
%               alpha_max - The maximum value of the alpha Vector.
%
%       OUTPUT DESCRIPTION:
%               im_filt  - Filtered image 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (c) 2019 University of Naples "Parthenope".
% All rights reserved.
% This work should only be used for nonprofit purposes.
% 
% By downloading and/or using any of these files, you implicitly agree to all the
% terms of the license, as specified in the document LICENSE.txt 
% (included in this package)


% Insert the input image in GPU
y=gpuArray(single(im_noisy+0.0001));

% Assume the Lambda/mean of the noise = 1
ECon = 0.5772156649;
varw = (2 - ECon.^2);

% Homomorphic Filtering
%---------------------------------------------------------------------------------------------

% Apply log to convert multiplicative noise to Additive Noise
ylog= log(y);

% Normalization
[m1,n1]=size(ylog);
meanylog = mean(ylog(:));
 
% Padarray circular
ylog2=padarray(ylog-meanylog,[m1/2 n1/2],'circular');

% Power spectrum of the degraded Image
Yf = fft2(ylog2);

% Initializing
xmedf=medfilt2(y,[5,5])+1e-3;
xmedlog= gpuArray(zeros(2*m1,2*n1,'single'));
xmedlog(m1/2+1:m1/2+m1,n1/2+1:n1/2+n1)=log(xmedf)-mean(log(xmedf(:)));

X0=fft2(xmedlog);

clear xmedf y ylog xmedlog ylog2

mu = meanylog + ECon;

min_alpha=1;
K=100;  
ALPHA=linspace(min_alpha, alpha_max, K); 

numelX0=numel(X0);
B=varw/2;


A=abs(X0).^2/numelX0;
for krt = 1 : 100
    Wk=A./(A+B);
    A=abs(Wk .* Yf).^2/numelX0;
end

xk_stack=gpuArray(zeros([m1,n1,K],'single'));
totale_stack=gpuArray(zeros([m1,n1,K],'single'));

clear X0 


for krt = 1 : K

    Wk=A./(A+(ALPHA(krt)*B));
    X1=Wk .* Yf;
    Xlogk =ifft2(X1)+ mu;
    xk=abs( exp(Xlogk(m1/2+1:m1/2+m1,n1/2+1:n1/2+n1)));
  
    b=[xk(1,:); xk ; xk(m1,:)];
    b=[b(:,1), b , b(:,n1)];
    v2=(xk-b(3:end, 2:end-1)).^2;     % sotto
    v2=v2+(xk-b(1:end-2, 2:end-1)).^2;   % sopra
    v2=v2+(xk-b(2:end-1, 3:end)).^2;     % destra
    v2=v2+(xk-b(2:end-1, 1:end-2)).^2;   % sinistra
    v2=v2+(xk-b(3:end, 1:end-2)).^2;     % sotto sinistra
    v2=v2+(xk-b(3:end, 3:end)).^2;       % sotto destra
    v2=v2+(xk-b(1:end-2, 1:end-2)).^2;   % sopra sinistra
    v2=v2+(xk-b(1:end-2, 3:end)).^2;     % sopra destra
    
    totale=v2./xk;          % hyperparametrs
    totale=totale./max(totale(:));
  
    xk_stack(:,:, krt)=xk;
    totale_stack(:,:, krt)=totale;
end

%%%%%
% alpha-MAP
totale=sum(totale_stack,3);
totale=medfilt2(totale, [5 5]);
totale=totale./max(totale(:));
perc=prctile(totale(:),95);
bordi=(1-totale/perc); 
bordi(bordi<=0.01)=0.05;
bordi=round(medfilt2(K*bordi)+0.5);

% Construct the final Output Soultion
im_filt=gpuArray(zeros(m1,n1,'single'));


for m=1:K
    index = find(bordi==m);
    x = xk_stack(:,:,m);
    x=x(:);
    im_filt(index)=x(index);
end

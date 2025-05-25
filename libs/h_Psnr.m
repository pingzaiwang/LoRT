function [ PSNR, MSE] =h_Psnr(T_true,T_hat)
T_hat = double(T_hat);
T_true = double(T_true);
dT=T_true-T_hat;
MSE = mean(dT(:).*dT(:));
peak = max(T_true(:));
PSNR = 10*log10(peak^2/MSE);

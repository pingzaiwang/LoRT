function TNN = f_comp_TNN(Y)
Y = fft(Y,[],3);
[~,~,n3]=size(Y);
mid3=floor(n3/2)+1;
vNN=zeros(n3,1);
for k=1:mid3
    [~,S,~] =svd(squeeze(Y(:,:,k)),'econ');
    vNN(k) = sum(S(:));
end
for k=mid3+1:n3
    vNN(k) = vNN(n3+2-k);
end
TNN=sum(vNN(:));



function [f,U,S,V]=f_dual_norm_SpKPSup(M,p,k)
% SVD using propack
% 注意U,S,V的重用性
q=1/(1-1/p);
[U,S,V]=lansvd(M,k,'L');
s=diag(S);
f=power( sum( power(s, q) ), 1/q);
s=power( s/f, 1.0/(p-1) );
S=diag(s);
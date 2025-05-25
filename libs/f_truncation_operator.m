function X=f_truncation_operator(X,alpha)
X=sign(X).*min(abs(X),alpha);
end

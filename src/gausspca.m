function F = gausspca(Fs,N)

[parts,moments]=iterate_gauss(Fs,2);
C = cov(parts');
[u,s,v] = svd(C);
F = u(:,1:N)'*parts;

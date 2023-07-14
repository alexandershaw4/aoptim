function y = featfft(x,n)

[y,c] = atcm.fun.approxlinfitgaussian(spm_vec(x));

y = [y(:); spm_vec(c.mu(1:n)); spm_vec(c.amp(1:n)); spm_vec(c.wid(1:n))];
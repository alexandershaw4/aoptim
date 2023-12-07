function [x0,ex]=agpropt(g,x0,V,nit,k,pp)
% super simplistic surrogate model optimisation routine based on Gaussian
% Process Regression
%
%
%

ex = g(x0);
f  = @(p) denan(g(p),exp(ex).^2);

if nargin < 5 || isempty(k)
    k  = 8;
end

if nargin < 6 || isempty(pp)
    pp = [];
end

x0 = x0(:);
V  = V(:);

if nargin < 4 || isempty(nit)
    nit=64;
end

errorplot = 1;
ee = ex;

for N = 1:nit
    % generate sample space to which we will fit Gaussian Processes
    p = [x0 - V,...
         x0 + V,...
         x0 + sqrt(V).*randn(length(x0),10)];
    p = [p pp];
    p=p';
    for i = 1:size(p,1)
        e(i) = f(p(i,:));
    end

    if k < 1e-6
        fprintf('reset k\n');
        k = 8;
    end
    
    for i = 1:length(x0)
        gprm = fitrgp(e(:),p(:,i),'Basis','constant','FitMethod','exact',...
            'PredictMethod','exact');
        [yp(i),se(i,:),yint(i,:)] = predict(gprm,ex/k);
    end
    
    % one more call...
    de = f(yp);
    %de1(2) = f(yint(:,1));
    %de1(3) = f(yint(:,2));

    
    if de < ex
        fprintf('It %d: accept (F = %d -> dF = %d\n',N,ex,de);
        ex = de;
        x0 = yp(:);
        ee  = [ee; ex];
    else
        fprintf('It %d: reject\n',N);
        k = k / 2;
        ee = [ee;ex];
    end

    if errorplot
        plot(1:N+1,ee,'b');hold on;
        plot(1:N+1,ee,'r*'); grid on; hold off;
        title(['k = ' num2str(k)]);
        drawnow;
    end

end

end
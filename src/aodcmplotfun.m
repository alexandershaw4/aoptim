function aodcmplotfun(p,params)

[~,~,y,t] = params.aopt.fun(p,params);

y(:,1:100)=0;
plot(t,real(y));
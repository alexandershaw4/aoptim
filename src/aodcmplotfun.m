function aodcmplotfun(p,params)

[~,~,y,t] = params.aopt.fun(p,params);

plot(t,real(y));
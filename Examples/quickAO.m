function [X,F,CV,Pp,Hi] = quickAO(fun,x0,V,y)


op = AO('options');

op.fun = fun;        % function/model f(x0)
op.x0  = x0(:);      % start values: x0
op.y   = y(:);       % data we're fitting (for computation of objective fun, e.g. e = Y - f(x)
op.V   = V(:);       % variance / step for each parameter, e.g. ones(length(x0),1)/8

op.objective='gauss'; % select smooth Gaussian error function

op.hyperparams = 0;
op.doparallel  = 1;

%Run the routine:
[X,F,CV,~,Hi] = AO(op); 

end
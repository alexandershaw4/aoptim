
% Example usage of AO.m optimiser to solve the Ackley function.
%---------------------------------------------------------------

% Remove any prevous global aoptim structure
clear global

% Define function, parameters (means), variances and data (or 0)
fun = @ackley_fun;
x0  = [3 .5];
V   = [1 1]/512;
y   = 0;

% Other options
maxit       = 128;  % number of iteraction
inner_loop  = inf;  % with an iter, num loops on same ascent
Q           = [];   % precision operator (ignored)
criterion   = 1e-13;% convergence value
min_df      = -inf; % minimum average imporvement expected (-inf=off)
order       = 2;    % see jaco: derivates order (recomnd 2 or 1)
writelog    = 0;    % write outputs to a log=1, or console=0
objective   = 'fe'; % objective: 'fe' (free energy), 'sse', 'mse', 'rmse', 'logevidence'
ba          = 0;    % explicitly (Bayesian) adjust param predictions 
im          = 1;    % include momentum term (recommend)
da          = 0;    % divergence adjustment
step_method = 1;    % param step: 1=aggressive or 3=carful

[X,F,Cp,PP,Hist] = AO(fun,x0,V,y,maxit,inner_loop,Q,criterion,min_df,...
                                order,writelog,objective,ba,im,da,step_method)
                            

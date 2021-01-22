clear global ; clear; close all;

% Example AO.m optimisation
%-------------------------------------------------------------
% This script creates a surface function, defined as sin(x)*cos(x)', with
% known parameters x = 2:20, then using random starting positions recovers
% x values that produce the same surface, using the AO free energy optimiser.


% Function
f = @(x) HighResMeanFilt( sin(spm_vec(x))*cos(spm_vec(x))', 2, 8 );

% True x-values we'll uncover, and true output (surface), y...

x  = 2:20; % actual optimsed values to find
y  = f(x); % surface given correct parameter values

rng default;

x0 = randn(size(x));     % random start positions for optimiser
V  = ones(size(x0))/128; % corresponding variances/step sizes

% Setting up the optimiser
%-------------------------------------------------------------
op = AO('options');  % this returns the optimiser input options structure
op.fun = f;          % function/model
op.x0  = x0;      % start values
op.y   = y;       % data we're fitting (for computation of objective fun)
op.V   = V;       % corresponding vars/step sizes for each param (x0)

op.step_method = 1;   % aggressive steps = 1, careful = 3, vanilla = 4.
op.maxit       = 156; % maximum number of iterations

op.doimagesc   = 1; % flag to plot a surface instead of 2d plot (for 2d outputs only)
op.hyperparams = 0; % switch off hyperparameter estimation/ascent
op.criterion   = -12000; 
op.ismimo      = 1;

%op.EnforcePriorProb=1;
%op.mleselect=1;

[X,F,CV] = AO(op);    % RUN IT

% Compare data with prediction
[f(x) f(X)]
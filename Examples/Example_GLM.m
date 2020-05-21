% example_aoglm
%
% Example of fitting a GLM using AO.m curvature-descent optimsation.
% Comparison of returned model (ß) to that using glmfit.m
close all; clear global;

dat = load([fileparts(fileparts(mfilename('fullpath'))) '/src/test_glm_data.mat']); 

% [disclaimer: somewhat unconventionally I have called the variable of
% predictor data 'y' and the thing we're predicting 'x' ]

% run the AO glm:
[b,F,cp,fit] = ao_glm(dat.x,dat.y);

% AO glm with no orthogonalisation
%[b,F,cp,fit] = ao_glm_no(dat.x,dat.y);

% AO pls
%[b,F,cp,fit] = ao_glm_eig(dat.x,dat.y,3);

% run glmfit
B = glmfit(dat.y,dat.x,'normal');

% take a look at the betas:
[B' ; b]
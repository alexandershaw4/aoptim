function plot_atcm_opt(M)

if ~isa(M,'AODCM')
    error('Input should be AODCM object after optimisation');
end

% get the optimisation history structure
h = M.history;
n = length(h.e);

for i = 1:n
    y(:,i) = spm_vec(M.opts.fun(h.p{i}));
end




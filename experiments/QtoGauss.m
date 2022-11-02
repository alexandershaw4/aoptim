function G = QtoGauss(Q,w)
% given a vector that has converted to a smoothed symmetrical matrix,
% perform explicit conversion to Gaussians
%
% e.g. x = [-10:1:10];           % a vector of data
%      Q = atcm.fun.AGenQn(x,4); % convert to smoothed matrix
%      Q = .5*(Q+Q');            % ensure symmetric
%      G = atcm.fun.QtoGauss(Q); % convert to Gaussian series
%
% AS22

if nargin < 2 || isempty(w)
    w = 4;
end

if isvector(Q)
    Q = atcm.fun.AGenQn(Q,4);
end

G = Q*0;
x = 1:length(Q);

for i = 1:length(Q)
    
    t = Q(:,i);
    
    [v,I] = max(t);
    
    G(:,i) = atcm.fun.makef(x,I-1,v,w);

end


end
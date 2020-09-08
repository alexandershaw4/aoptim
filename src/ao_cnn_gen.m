function [pred,x,m,a,data] = ao_cnn_gen(p)
global aocnn

p  = p(:);
lm = length(spm_vec(aocnn.m));
m  = spm_unvec(p(1:lm),aocnn.m);
a  = p(lm+1:end);

for i = 1:length(m)
    m{i} = m{i} * act(a(i)) ;
end

% Run it
for s = 1:size(aocnn.x,1)
    xx = aocnn.x(s,:);
    for i = 1:length(m)
        xx = xx*m{i};
    end
    pred(s,:) = xx;
end

pred = {pred};
data = aocnn.y;
x    = aocnn.x;

end

function y = act(x)

a = 1;
c = 0;
y = 1./(1 + exp(-a.*(x-c)));

end
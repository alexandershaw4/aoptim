function q = Qtoq(Q)
% AS

for i = 1:size(Q,1)    
    q{i} = Q*0;
end

for i = 1:length(q)
    j = i;
    
    q{i}(i,j) = Q(i,j);
    
    try; q{i}(i+1,j+0) = Q(i+1,j+0);end
    try; q{i}(i-1,j+0) = Q(i-1,j+0);end
    
    try; q{i}(i+0,j+1) = Q(i+0,j+1);end
    try; q{i}(i+0,j-1) = Q(i+0,j-1);end
    
    try; q{i}(i+1,j+1) = Q(i+1,j+1);end
    try; q{i}(i-1,j-1) = Q(i-1,j-1);end
    
    try; q{i}(i+1,j-1) = Q(i+1,j-1);end
    try; q{i}(i-1,j+1) = Q(i-1,j+1);end   

end
function visualisenn(m,v)

m.modelspace = spm_unvec(v,m.modelspace);

if isa(m,'AONN') && any(strcmp(properties(m),'modelspace'))
    m = m.modelspace;
end

figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off',...
    'position',[706         380        1226         486]);

s(1) = subplot(151); imagesc(m{1});
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.XTick = [];
ax.YTick = [];
s(1).YColor = [1 1 1];
s(1).XColor = [1 1 1];
s(1).Color  = [.3 .3 .3];
ylabel('Inputs','fontsize',18);xlabel('Hidden Layer: Neurons','fontsize',18);
title('MAPPING','color','w','fontsize',18);

s(2) = subplot(152); imagesc(diag(1./(1+exp(-m{2}))));
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.XTick = [];
ax.YTick = [];
s(2).YColor = [1 1 1];
s(2).XColor = [1 1 1];
s(2).Color  = [.3 .3 .3];
axis square;
xlabel('Diagonals','fontsize',18);
title('Neuron Activation Functions','color','w','fontsize',18);

s(3) = subplot(153); imagesc(m{3});
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.XTick = [];
ax.YTick = [];
s(3).YColor = [1 1 1];
s(3).XColor = [1 1 1];
s(3).Color  = [.3 .3 .3];
ylabel('Hidden Layer Neurons'' Self-Gain','fontsize',18);xlabel('OUTPUTS','fontsize',18);
title('MAPPING','color','w','fontsize',18);


s(4) = subplot(154); imagesc(m{4});
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.XTick = [];
ax.YTick = [];
s(3).YColor = [1 1 1];
s(3).XColor = [1 1 1];
s(3).Color  = [.3 .3 .3];
ylabel('Hidden Layer: Neurons','fontsize',18);xlabel('OUTPUTS','fontsize',18);
title('MAPPING','color','w','fontsize',18);

s(5) = subplot(155); imagesc(diag(1./(1+exp(-m{5}))));
ax = gca;
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.XTick = [];
ax.YTick = [];
s(4).YColor = [1 1 1];
s(4).XColor = [1 1 1];
s(4).Color  = [.3 .3 .3];
ylabel('Diagonals','fontsize',18);xlabel('OUTPUTS','fontsize',18);
title('Ouput Activation Functions','color','w','fontsize',18);





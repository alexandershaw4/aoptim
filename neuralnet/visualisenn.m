function visualisenn(m)

figure('Name','AO','Color',[.3 .3 .3],'InvertHardcopy','off',...
    'position',[706         380        1226         486]);

s(1) = subplot(131); imagesc(m{1});
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

s(2) = subplot(132); imagesc(diag(1./(1+exp(-m{2}))));
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

s(3) = subplot(133); imagesc(m{3});
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






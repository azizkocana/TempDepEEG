clear all
close all

load barplot_dat

FS1 = 13;
FS2 = 15;

[mean_rsvp, sort_idx] = sort(mean_rsvp);
var_rsvp = var_rsvp(sort_idx);
vars = vars(sort_idx);
means = means(sort_idx);

m_cat = [means; mean_rsvp].';
v_cat = [vars; var_rsvp].';

f=figure();
s=subplot(2,1,1);
 h = barwitherr(v_cat, m_cat);
 set(gca,'XTickLabel',{'U-1','U-2','U-3','U-4','U-5','U-6','U-7','U-8','U-9'}, 'FontSize', FS1)
ylim([0.5 1])
leg = legend('RNN','RDA-KDE');
set(leg,'FontSize',FS1,'Location','southeast')
ylabel('AUC', 'FontSize', FS2)
xlabel('Users', 'FontSize', FS2)
set(gcf, 'PaperPositionMode', 'auto');
iptsetpref('ImshowBorder','tight');
box off
saveas(s,'C:\Users\Aziz\Desktop\GIT\nnjunkyard\latex_fig\auc-bg','jpg');



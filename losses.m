proj_name='retinet';
exp_name='exp1';
load(['/',proj_name,'/',exp_name,'/evalloss.mat'])
load(['/',proj_name,'/',exp_name,'/trainloss.mat'])
plot(1:length(evalloss),evalloss,1:length(trainloss),trainloss)
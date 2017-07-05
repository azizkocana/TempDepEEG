clear classes;
clear;
close all;

% adds current folder to MATLAB's python search path (kludge: current
% folder must contain langModelMod)
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

% Reload python module

mod = py.importlib.import_module('triggerdecoder');

if strcmp(pyversion, '2.7')
    py.reload(mod);    
elseif strcmp(pyversion, '3.5')
    py.importlib.reload(mod);
end

load dummytrig

a = py.triggerdecoder.trigger_decoder(afterFrontendFilterTrigger.',triggerPartitioner);


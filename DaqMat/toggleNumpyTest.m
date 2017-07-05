clear classes;
clear;
clc;
close all;

% you should have python installed
pyversion;

% adds current folder to MATLAB's python search path (kludge: current
% folder must contain langModelMod)
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

% Reload python module

mod = py.importlib.import_module('toggleNumpyTest');

if strcmp(pyversion, '2.7')
    py.reload(mod);    
elseif strcmp(pyversion, '3.5')
    py.importlib.reload(mod);
end

% Generate dummy data
Y(:,:) = [1:4; 5:8];

X = Y;

% Transfer data. Python function prints so we get a sense that the output
% is laid out the same
outputCell = (toggleNumpy(X));

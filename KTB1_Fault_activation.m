%% KTB1_Fault_activation.m
clc; clear; close all;

model = 'KTB1_F';

% ---------------- USER SETTINGS ----------------
fault_1_to_4 = 0;   % 0..4
fault_5_to_8 = 4;   % 0..4
% fault_9  = 0;
% fault_10 = 0;
% fault_11 = 0;
% fault_12 = 0;

t_fault_on = 501;
t_final    = 2024;

% ---------------- LOAD MODEL ----------------
load_system(model);
% Make algebraic loop solver more robust
set_param(model, 'AlgebraicLoopSolver', 'LineSearch');  % or 'Auto'
set_param(model, 'RelTol', '1e-4');                     % slightly tighter tolerance

% Ensure model does NOT try to load state from config
set_param(model, 'LoadInitialState', 'off');
set_param(model, 'InitialState', '');

%% ======== PHASE 1: 0–50s (All faults OFF) ========
disp("=== Phase 1 ===");

% Turn all faults off
set_param([model '/Clac'],   'Value','0');
set_param([model '/Cn'],     'Value','0');
% set_param([model '/RFC'],    'Value','0');
% set_param([model '/ClovC'],  'Value','0');
% set_param([model '/ClovrC'], 'Value','0');
% set_param([model '/TC'],     'Value','0');

simOut1 = sim(model, ...
    'StartTime','0', ...
    'StopTime',num2str(t_fault_on), ...
    'SaveFinalState','on', ...
    'FinalStateName','xFinal_phase1');

% GET the state object:
xstate = simOut1.xFinal_phase1;

% Place into BASE WORKSPACE so Simulink can see it
assignin('base', 'xFinal_phase1', xstate);

%% ======== PHASE 2: 50–120s (Faults ON) ========
disp("=== Phase 2 ===");

% Activate selected faults
set_param([model '/Clac'],   'Value',num2str(fault_1_to_4));
set_param([model '/Cn'],     'Value',num2str(fault_5_to_8));
% set_param([model '/RFC'],    'Value',num2str(fault_9));
% set_param([model '/ClovC'],  'Value',num2str(fault_10));
% set_param([model '/ClovrC'], 'Value',num2str(fault_11));
% set_param([model '/TC'],     'Value',num2str(fault_12));

% Continue simulation FROM the saved state
% IMPORTANT: Pass state variable NAME as string!
simOut2 = sim(model, ...
    'StartTime',num2str(t_fault_on), ...
    'StopTime', num2str(t_final), ...
    'LoadInitialState','on', ...
    'InitialState','xFinal_phase1');   % NOTICE THE QUOTES

disp("=== Simulation complete ===");


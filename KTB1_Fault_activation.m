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

% Zero-crossing warnings appear on several Sign/Abs blocks when the adaptive
% detector refuses to shrink the step size. Relax the diagnostic, keep the
% adaptive algorithm (per the Simulink recommendation), and disable
% zero-crossing detection on the known chatter-prone blocks.
set_param(model, 'IgnoredZcDiagnostic', 'none');         % turn warning off
set_param(model, 'ZeroCrossAlgorithm', 'Adaptive');      % recommended algorithm

zc_blocks = {
    'Abs';
    'PID Controller1/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator';
    'PID Controller2/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator';
    'PID Controller3/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator';
    'PID Controller4/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator';
    'PID Controller7/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator'
};

for i = 1:numel(zc_blocks)
    blk = [model '/' zc_blocks{i}];
    try
        set_param(blk, 'ZeroCross', 'off');
    catch ME
        % Some library blocks promote ZeroCross as a mask parameter, making it
        % read-only from the child. Skip those blocks to avoid fatal errors but
        % still proceed with the remaining configuration.
        warning("Skipping zero-crossing disable on %s (%s)", blk, ME.message);
    end
end

% Zero-crossing warnings appear on several Sign/Abs blocks when the adaptive
% detector refuses to shrink the step size. Relax the diagnostic, keep the
% adaptive algorithm (per the Simulink recommendation), and disable
% zero-crossing detection on the known chatter-prone blocks.
set_param(model, 'IgnoredZcDiagnostic', 'none');         % turn warning off
set_param(model, 'ZeroCrossAlgorithm', 'Adaptive');      % recommended algorithm

zc_blocks = {
    'Abs';
    'PID Controller1/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator';
    'PID Controller2/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator';
    'PID Controller3/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator';
    'PID Controller4/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator';
    'PID Controller7/Anti-windup/Cont. Clamping Parallel/SignPreIntegrator'
};

for i = 1:numel(zc_blocks)
    set_param([model '/' zc_blocks{i}], 'ZeroCross', 'off');
end

% Zero-crossing warnings appear on several Sign/Abs blocks when the adaptive
% detector refuses to shrink the step size. Relax the diagnostic and use the
% non-adaptive algorithm to avoid repeated warnings without changing the
% model structure.
set_param(model, 'IgnoredZcDiagnostic', 'none');         % turn warning off
set_param(model, 'ZeroCrossAlgorithm', 'Nonadaptive');   % simpler ZC handling

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



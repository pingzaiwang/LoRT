% Add necessary function paths
addpath('./libs/');
addpath('./data');
clear; clc; close all;

% Set random seed for reproducibility
rng(42);

% Model parameters
lambda0 = 1e-2;         % Nuclear norm regularization parameter
alpha = 1e-1;           % Scaling factor for model parameter alpha
lambdaTil = 1e0;        % Regularization parameter for refined model
gamma = 1e3;            % Learning rate for the first step
gammaStep2 = 1e3;       % Learning rate for the second step
maxIterTSnc = 500;      % Maximum iterations for TensorSync

% Dataset and number of source tasks
dataFolder = './data';
files = dir(fullfile(dataFolder, '*.mat'));
dataSet = {};
for i = 1:length(files)
    dataSet{end+1} = files(i).name;
end

% Number of source tasks
KSource = 4;

% Target task sampling rates
targetSamplingRates = [0.05, 0.1, 0.15, 0.2];

% targetSamplingRates = [0.08, 0.12, 0.18];
% Initialize results structure
if isfile('experiment_results_different_SR_target.mat')
    load('experiment_results_different_SR_target.mat', 'results');
else
    results = struct();
end

for dataIdx = 1:length(dataSet)
    dataFile = dataSet{dataIdx};
    
    % Load data
    load(fullfile('data', dataFile), 'T');  % Load RGB video data
    sz = size(T(:,:,:,1));            % Tensor size
    
    for srIdx = 1:length(targetSamplingRates)
        targetSR = targetSamplingRates(srIdx);
        fprintf('Processing %s, Target Task Sampling Rate = %.2f\n', dataFile, targetSR);
                
        % Target and source tasks
        tWtgt = T(:,:,:,KSource+1);             % Target task tensor
        cWsrc = cell(KSource,1);                % Cell array for source task tensors
        for k = 1:KSource
            cWsrc{k} = T(:,:,:,k);        % Assign each source task tensor
        end

        % Noise level configuration
        noiseLevel = 0; % Set to 0, can be adjusted to 1e-1*1/sqrt(prod(sz)) to add noise

        % Data generation configuration (dataCfg)
        dataCfg.tSize = sz;                       % Tensor size
        dataCfg.tgtRank = 2;                      % Rank of target task tensor
        dataCfg.rankMax = 2;                      % Maximum rank of source task tensors
        dataCfg.nSourceTask = KSource;            % Number of source tasks (K)
        dataCfg.vDiffTNN = 2 * ones(KSource, 1);  % deltaTNN parameter (related to source tasks)
        dataCfg.SRTgt = targetSR;                 % Sampling rate for target task
        dataCfg.vSRSrc = 0.8 * ones(KSource, 1);  % Sampling rate for each source task
        dataCfg.noiseLevelTarget = noiseLevel;    % Noise level for target task
        dataCfg.vNoiseLevelSrc = noiseLevel * ones(KSource, 1); % Noise level for each source task

        % Algorithm configuration (algCfg)
        algCfg.lambda0 = lambda0;                 % Regularization parameter for nuclear norm
        algCfg.lambdaTil = lambdaTil;             % Regularization parameter for nuclear norm in refinement
        algCfg.gamma = gamma;                     % Learning rate
        algCfg.gammaStep2 = gammaStep2;           % Learning rate for the second step
        algCfg.vAlpha = alpha * ones(KSource, 1); % Alpha parameter
        algCfg.maxIter = maxIterTSnc;             % Maximum number of iterations
        algCfg.tol = 1e-15;                       % Convergence tolerance
        algCfg.verbose = 1;                       % Verbose mode (output detailed iteration information)
        algCfg.verboseInterval = 10;              % Interval for outputting information

        % Initialize weight tensors for target and source tasks
        algCfg.targetW0 = zeros(dataCfg.tSize);    % Initial weight tensor for target task
        algCfg.sourceW0 = cell(dataCfg.nSourceTask, 1); % Initial weight tensors for source tasks
        for k = 1:dataCfg.nSourceTask
            algCfg.sourceW0{k} = zeros(dataCfg.tSize);
        end
        algCfg.maxIter_step2 = 50;

        % Generate data for target and source tasks
        [targetObj, sourceObj] = f_generate_observation_from_existing_tensors(dataCfg, tWtgt, cWsrc);

        % Run TensorSync for computation
        fprintf('Running f_TensorSync...\n');
        [memo_TensorSync_step1, memo_TensorSync_step2] = f_TensorSync_Completion_v2(targetObj, sourceObj, algCfg);

        % Run D-TensorSync for computation
        fprintf('Running f_D_TensorSync...\n');
        alpha = 1e0;
        algCfg.gamma = 1e3;              % Adjust learning rate
        algCfg.lambda0 = 1e-2;           % Adjust regularization parameter
        algCfg.lambdaFuse = 1e3;         % Fusion parameter for D-TensorSync
        algCfg.gammaFuse = 1e-1;         % Learning rate for fusion
        algCfg.gammaFuse1 = 1e-1;
        algCfg.vAlpha = alpha * ones(KSource, 1);       % Alpha parameter
        algCfg.lambdaTil = 1e0;         % Adjust regularization parameter
        algCfg.maxIter = 500;            % Adjust maximum number of iterations
        algCfg.tol = 1e-5;               % Set convergence tolerance
        algCfg.verbose = 1;              % Enable verbose mode
        algCfg.verboseInterval = 10;     % Set interval for verbose output
        algCfg.maxIter_D_refine = 50;
        memo_D_TensorSync = f_D_TensorSync_Completion_v2(targetObj, sourceObj, algCfg);

        % Run target-only estimation (using TNN)
        fprintf('Running target-only estimation (using TNN)...\n');
        algCfg.rho = 1e-5;               % ADMM parameter
        algCfg.nu = 1.1;                 % ADMM update multiplier
        algCfg.maxRho = 1e6;             % Maximum value for rho
        algCfg.maxIter = 100;            % Set maximum number of iterations
        algCfg.tol = 1e-5;               % Set convergence tolerance
        memo_TargetOnly = f_target_only_completion_tnn_admm_v2(targetObj, algCfg);

        % Store results
        results.(strrep(dataFile, '.mat', '')).(['SR_' num2str(targetSR*100)]) = struct();
        results.(strrep(dataFile, '.mat', '')).(['SR_' num2str(targetSR*100)]).TensorSync_step1 = memo_TensorSync_step1;
        results.(strrep(dataFile, '.mat', '')).(['SR_' num2str(targetSR*100)]).TensorSync_step2 = memo_TensorSync_step2;
        results.(strrep(dataFile, '.mat', '')).(['SR_' num2str(targetSR*100)]).D_TensorSync = memo_D_TensorSync;
        results.(strrep(dataFile, '.mat', '')).(['SR_' num2str(targetSR*100)]).TargetOnly = memo_TargetOnly;
        results.(strrep(dataFile, '.mat', '')).(['SR_' num2str(targetSR*100)]).WtGT.tWtgt = tWtgt;
        results.(strrep(dataFile, '.mat', '')).(['SR_' num2str(targetSR*100)]).Wobs.Wobs = targetObj.tY;
    end
end

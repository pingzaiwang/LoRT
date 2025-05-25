% Add necessary function paths
addpath('./libs/');
addpath('./data');
clear; clc; close all;

% Set random seed for reproducibility
rng(42);


% Model parameters for source tasks=4
lambda0 = 1e-2;         % Regularization parameter for nuclear norm
alpha = 1e-1;           % Scaling factor for the model parameter alpha
lambdaTil = 1e0;        % Regularization parameter for the refined model
gamma = 1e3;            % Learning rate for the first step
gammaStep2 = 1e3;       % Learning rate for the second step
maxIterTSnc = 500;      % Maximum number of iterations for TensorSync

% Data sets and source task numbers
dataFolder = './data';
files = dir(fullfile(dataFolder, '*.mat'));
dataSet = {};
for i = 1:length(files)
    dataSet{end+1} = files(i).name;
end

% Source Data
KSource = [1,2,3,4,5,6,7,8];

% Initialize results structure
if isfile('experiment_results_different_source_tasks.mat')
    load('experiment_results_different_source_tasks.mat', 'results');
else
    results = struct();
end

for dataIdx = 1:length(dataSet)
    dataFile = dataSet{dataIdx};
    
    % Load data
    load(fullfile('data', dataFile), 'T');  % Load the RGB video data
    sz = size(T(:,:,:,1));            % Size of the tensor
    
    for K = KSource
        fprintf('Processing %s with K=%d\n', dataFile, K);
        
        % Target and source tasks
        tWtgt = T(:,:,:,K+1);             % Target task tensor
        cWsrc = cell(K,1);                % Cell array for source task tensors
        for k = 1:K
            cWsrc{k} = T(:,:,:,k);        % Assign each source task tensor
        end
        
        % Noise level configuration
        noiseLevel = 0; % Set to 0, can be adjusted to 1e-1*1/sqrt(prod(sz)) for noise

        % Data generation configuration (dataCfg)
        dataCfg.tSize = sz;                       % Tensor size
        dataCfg.tgtRank = 2;                      % Rank of the target task tensor
        dataCfg.rankMax = 2;                      % Maximum rank for source task tensors
        dataCfg.nSourceTask = K;                  % Number of source tasks (K)
        dataCfg.vDiffTNN = 2 * ones(K, 1);        % deltaTNN parameter (related to source tasks)
        dataCfg.SRTgt = 0.05;                      % Sampling rate for the target task
        dataCfg.vSRSrc = 0.8 * ones(K, 1);        % Sampling rate for each source task
        dataCfg.noiseLevelTarget = noiseLevel;    % Noise level for the target task
        dataCfg.vNoiseLevelSrc = noiseLevel * ones(K, 1); % Noise level for each source task

        % Algorithm configuration (algCfg)
        algCfg.lambda0 = lambda0;                 % Regularization parameter for nuclear norm
        algCfg.lambdaTil = lambdaTil;             % Regularization parameter for nuclear norm in refinement
        algCfg.gamma = gamma;                     % Learning rate
        algCfg.gammaStep2 = gammaStep2;           % Learning rate for the second step
        algCfg.vAlpha = alpha * ones(K, 1);       % Alpha parameter
        algCfg.maxIter = maxIterTSnc;             % Maximum number of iterations
        algCfg.tol = 1e-15;                       % Tolerance for convergence
        algCfg.verbose = 1;                       % Verbose mode (output detailed iteration info)
        algCfg.verboseInterval = 10;              % Interval of iterations to output information

        % Initialize weight tensors for the target task and source tasks
        algCfg.targetW0 = zeros(dataCfg.tSize);    % Initial weight tensor for the target task
        algCfg.sourceW0 = cell(dataCfg.nSourceTask, 1); % Initial weight tensors for the source tasks
        for k = 1:dataCfg.nSourceTask
            algCfg.sourceW0{k} = zeros(dataCfg.tSize);
        end
        algCfg.maxIter_step2 = 50;

        % Generate data for target and source tasks
        [targetObj, sourceObj] = f_generate_observation_from_existing_tensors(dataCfg, tWtgt, cWsrc);
        
        
        % save('experiment_results_different_source_tasks.mat', 'results');
        % Run TensorSync for the calculation
        fprintf('Running f_TensorSync...\n');
        [memo_TensorSync_step1, memo_TensorSync_step2] = f_TensorSync_Completion_v2(targetObj, sourceObj, algCfg);

        % Run D-TensorSync for the calculation
        fprintf('Running f_D_TensorSync...\n');
        alpha = 1e2;
        algCfg.gamma = 1e3;              % Adjust learning rate
        algCfg.lambda0 = 1e-2;           % Adjust regularization parameter
        algCfg.lambdaFuse = 1e6;         % Fusion parameter for D-TensorSync
        algCfg.gammaFuse = 1e3;          % Learning rate for fusion
        algCfg.gammaFuse1 = 1e3;
        algCfg.vAlpha = alpha * ones(K, 1);       % Alpha parameter
        algCfg.lambdaTil = 1e-2;         % Adjust regularization parameter
        algCfg.maxIter = 500;            % Adjust maximum number of iterations
        algCfg.maxIter_D_refine = 50;
        algCfg.tol = 1e-5;               % Set tolerance for convergence
        algCfg.verbose = 1;              % Enable verbose mode
        algCfg.verboseInterval = 10;     % Set interval for verbose output
        memo_D_TensorSync = f_D_TensorSync_Completion_v2(targetObj, sourceObj, algCfg);

        % Run Target-Only Estimation by TNN
        fprintf('Running Target Only Estimation by TNN...\n');
        algCfg.rho = 1e-5;               % ADMM parameter
        algCfg.nu = 1.1;                 % ADMM update multiplier
        algCfg.maxRho = 1e6;             % Maximum value for rho
        algCfg.maxIter = 100;            % Set maximum number of iterations
        algCfg.tol = 1e-5;               % Set tolerance for convergence
        memo_TargetOnly = f_target_only_completion_tnn_admm_v2(targetObj, algCfg);

        % Store results
        results.(strrep(dataFile, '.mat', '')).(['K_' num2str(K)]) = struct();
        results.(strrep(dataFile, '.mat', '')).(['K_' num2str(K)]).TensorSync_step1 = memo_TensorSync_step1;
        results.(strrep(dataFile, '.mat', '')).(['K_' num2str(K)]).TensorSync_step2 = memo_TensorSync_step2;
        results.(strrep(dataFile, '.mat', '')).(['K_' num2str(K)]).D_TensorSync = memo_D_TensorSync;
        results.(strrep(dataFile, '.mat', '')).(['K_' num2str(K)]).TargetOnly = memo_TargetOnly;
        results.(strrep(dataFile, '.mat', '')).(['K_' num2str(K)]).WtGT.tWtgt = tWtgt;
        results.(strrep(dataFile, '.mat', '')).(['K_' num2str(K)]).Wobs.Wobs = targetObj.tY;
    end
end
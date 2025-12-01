classdef FaultDetectionBridge < matlab.System ...
                               & matlab.system.mixin.Propagates ...
                               & matlab.system.mixin.CustomIcon
    % FaultDetectionBridge
    %
    % Publishes Simulink signal to Python LSTM via MQTT.
    % Uses a global mqttclient managed by getGlobalMqttClient.m.
    %
    % INPUT u:
    %   [Time, Data_ 1, Data_ 2, ..., Data_17]
    %
    % OUTPUTS:
    %   FaultID = mqtt_fault_id   (0..8)
    %   Prob    = mqtt_fault_prob (0..1)

    properties(Nontunable)
        BrokerAddress = 'tcp://localhost';
        Port          = 1883;

        PubTopic      = 'fault_detection/live/features';
        SubTopic      = 'fault_detection/live/result';

        ArtifactsRoot = ...
            'C:\Users\Muckbul\Desktop\Ṣtudies\Thesis\KTB1Matlab\Fault Detection and Diagnosis\Fault Detection and Diagnosis\Code';
    end

    properties(Access = private)
        MQTTClient
        FeatureNames cell = {};
        nFeat double = 0;
    end

    methods(Access = protected)

        function setupImpl(obj)
            % --- Load feature names from JSON ---
            try
                jsonPath = fullfile(obj.ArtifactsRoot, 'feature_names.json');
                txt      = fileread(jsonPath);
                names    = jsondecode(txt);
                if isstring(names); names = cellstr(names); end

                obj.FeatureNames = names;
                obj.nFeat        = numel(names);

                disp('[FaultBridge] Loaded feature_names.json:');
                disp(obj.FeatureNames);
            catch ME
                warning('[FaultBridge] Could not load feature_names.json: %s', ME.message);
                obj.FeatureNames = {};
                obj.nFeat        = 0;
            end

            % --- Get global mqttclient (shared across simulations) ---
            try
                obj.MQTTClient = getGlobalMqttClient(obj.BrokerAddress, ...
                                                     obj.Port, ...
                                                     obj.SubTopic);
            catch ME
                warning('[FaultBridge] Could not get global MQTT client: %s', ME.message);
                obj.MQTTClient = [];
            end
        end

        function [FaultID, Prob] = stepImpl(obj, u)
            FaultID = 0;
            Prob    = 0;

            % --- DEBUG: show that stepImpl is running ---
            % disp('[FaultBridge] stepImpl called');

            % ---------- Publish features over MQTT ----------
            if ~isempty(obj.MQTTClient)
                try
                    featMap = containers.Map('KeyType','char','ValueType','double');

                    if ~isempty(obj.FeatureNames)
                        nLoc = min(numel(u), obj.nFeat);
                        for i = 1:nLoc
                            key = obj.FeatureNames{i};  % 'Time', 'Data_ 1', ...
                            featMap(key) = u(i);
                        end
                    else
                        for i = 1:numel(u)
                            featMap(sprintf('f%d', i)) = u(i);
                        end
                    end

                    payloadStruct = struct('features', featMap);
                    jsonStr       = jsonencode(payloadStruct);

                    write(obj.MQTTClient, obj.PubTopic, jsonStr);
                    % disp('[FaultBridge] published one feature message'); % optional debug
                catch ME
                    warning('[FaultBridge] Error while publishing: %s', ME.message);
                end
            end

            % ---------- Read last result from base workspace ----------
            try
                if evalin('base','exist(''mqtt_fault_id'',''var'')')
                    FaultID = evalin('base','double(mqtt_fault_id)');
                end
                if evalin('base','exist(''mqtt_fault_prob'',''var'')')
                    Prob = evalin('base','double(mqtt_fault_prob)');
                end
            catch
                % keep defaults on error
            end
        end

        function releaseImpl(obj)
            % DO NOT delete the global client here.
            disp('[FaultBridge] releaseImpl called (global client kept alive).');
        end

        % ---------- Output characteristics ----------
        function [sz1, sz2] = getOutputSizeImpl(~)
            sz1 = [1 1]; sz2 = [1 1];
        end
        function [dt1, dt2] = getOutputDataTypeImpl(~)
            dt1 = 'double'; dt2 = 'double';
        end
        function [c1, c2] = isOutputComplexImpl(~)
            c1 = false; c2 = false;
        end
        function [f1, f2] = isOutputFixedSizeImpl(~)
            f1 = true; f2 = true;
        end
        function icon = getIconImpl(~)
            icon = 'FaultBridge';
        end
    end
end

































































































































































































































































































































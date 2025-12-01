function client = getGlobalMqttClient(brokerAddress, port, subTopic)
% getGlobalMqttClient  Return a persistent MQTT client subscribed to result topic.

persistent gClient

% Reuse an existing, connected client if available
if ~isempty(gClient) && isvalid(gClient)
    try
        if gClient.Connected
            client = gClient;
            return;
        else
            % Drop stale, disconnected client so we can recreate it
            delete(gClient);
            gClient = [];
        end
    catch
        % If anything unexpected happens while checking connection state,
        % fall through and create a fresh client.
        gClient = [];
    end
end

% Create new client
fprintf('[MQTT-Bridge] Creating mqttclient at %s:%d\n', brokerAddress, port);

gClient = mqttclient(brokerAddress, 'Port', port);

% If connection failed, surface a clear error so callers can handle it.
if ~gClient.Connected
    delete(gClient);
    gClient = [];
    error('MQTT client failed to connect to %s:%d. Ensure the broker is reachable.', ...
          brokerAddress, port);
end

% Subscribe to result topic and attach callback
subscribe(gClient, subTopic, @mqtt_fault_result_callback);

client = gClient;
end

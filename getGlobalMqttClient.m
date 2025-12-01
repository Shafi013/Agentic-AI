function client = getGlobalMqttClient(brokerAddress, port, subTopic)
% getGlobalMqttClient  Return a persistent MQTT client subscribed to result topic.

persistent gClient

if ~isempty(gClient) && isvalid(gClient)
    client = gClient;
    return;
end

% Create new client
fprintf('[MQTT-Bridge] Creating mqttclient at %s:%d\n', brokerAddress, port);

gClient = mqttclient(brokerAddress, 'Port', port);

% Subscribe to result topic and attach callback
subscribe(gClient, subTopic, @mqtt_fault_result_callback);

client = gClient;
end

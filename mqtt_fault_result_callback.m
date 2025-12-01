function mqtt_fault_result_callback(~, msg)
% msg is a timetable with variables 'Topic' and 'Data'

try
    % Last message only
    raw = msg.Data{end};       % cell -> char
    txt = char(raw);
    data = jsondecode(txt);

    winner = data.winner;
    prob   = data.winner_prob;

    if strcmp(winner, 'OM0')
        fault_id = 0;
    else
        num = str2double(extractAfter(winner, "OM"));
        if isnan(num); num = 0; end
        fault_id = num;
    end

    assignin('base', 'mqtt_fault_id',  fault_id);
    assignin('base', 'mqtt_fault_prob', prob);

    % Uncomment for debugging:
    % fprintf('[MQTT-Bridge] winner=%s, id=%d, prob=%.3f\n', winner, fault_id, prob);

catch ME
    warning('[MQTT-Bridge] callback error: %s', ME.message);
end
end

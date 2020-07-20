DATA1 = LOAD './dataverse_files/' USING PigStorage('\t') AS (gridID:chararray,timeInterval:long,countryCode:chararray,smsIn:double,smsOut:double,callIn:double,callOut:double,internet:double);
DATAG = GROUP DATA1 BY gridID;
DATAF = FOREACH DATAG GENERATE group AS gridID, SUM(DATA1.smsIn) AS smsIn_tot,SUM(DATA1.smsOut) AS smsOut_tot,SUM(DATA1.callIn) AS callIn_tot,SUM(DATA1.callOut) AS callOut_tot,SUM(DATA1.internet) AS internet_tot;
STORE DATAF INTO './finalsum' USING PigStorage ('\t');
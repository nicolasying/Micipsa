# Computing tasks for 19/03/21, Ridge GS for base, base+SIM, base+ASN

# echo "Running base for all subjects"
# . setmodel ./models/fr/rms-wrate-cwrate/
# make first-level-ridge |& tee -a night_190321_base.log


# echo "Running ASN for all subjects"
# . setmodel ./models/fr/rms-wrate-cwrate-asn200/
# make first-level-ridge |& tee -a night_190322_asn.log

# echo "Running SIM for all subjects"
# . setmodel ./models/fr/rms-wrate-cwrate-sim103/
# make first-level-ridge |& tee -a night_190322_sim.log

# Computing tasks for 19/03/30, Ridge GS in dim and alpha for base, base+SIM, base+ASN
# Got results base+SIM == base


# Changing base to rms + wrate
# Computing tasks for 19/04/01, Ridge GS in dim and alpha for base, base+SIM, base+ASN

# python ./models/Micipsa\ Pipeline.py |& tee >( ts "%d-%m-%y %H_%M_%S" >> pipeline_190401.log) &
python main.py |& tee >( ts "%d-%m-%y %H_%M_%S" >> ./log/fit_intercept_fix.log) &
# python ./models/notifyer.py -f pipeline_190401.log -i 3600 & 
#systemctl suspend
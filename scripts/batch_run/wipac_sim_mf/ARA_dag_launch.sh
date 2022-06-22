mkdir logs
condor_submit_dag -config dagman_config A2_noise.dag
condor_submit_dag -config dagman_config A3_noise.dag
condor_submit_dag -config dagman_config A2_signal.dag
condor_submit_dag -config dagman_config A3_signal.dag

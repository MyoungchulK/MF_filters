mkdir logs
condor_submit_dag -config dagman_config A2.dag
condor_submit_dag -config dagman_config A3.dag

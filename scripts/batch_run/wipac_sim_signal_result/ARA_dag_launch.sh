mkdir logs
#condor_submit_dag -config dagman_config A2.dag
condor_submit_dag -config dagman_config A2_g.dag
#condor_submit_dag -config dagman_config A3.dag
condor_submit_dag -config dagman_config A3_g.dag

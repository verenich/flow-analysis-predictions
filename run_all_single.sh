#!/bin/bash -l
for MIN_CASES in 30 3000 30000 3000000
do
    for DATASET_NAME in BPI2012A.csv BPI2012O.csv BPI2012W.csv BPI2012W_no_dup.csv CreditRequirement.csv helpdesk.csv hospital_billing_977.csv traffic_fines_139.csv
    #for DATASET_NAME in bpic2017 traffic_fines
    do
        for BUCKET_METHOD in zero
        do
            for CLS_ENCODING in agg laststate combined
            do
                if [ $DATASET_NAME == "BPI2012O.csv" ] ; then
                    memory=3gb
                elif [ $DATASET_NAME == "helpdesk.csv" ] ; then
                    memory=3gb
                elif [ $DATASET_NAME == "BPI2012A.csv" ] ; then
                    memory=4gb
                else
                    memory=20gb
                fi
                qsub -l mem=$memory -l walltime=23:00:00 -N job_"$DATASET_NAME"_"$BUCKET_METHOD"_"$CLS_ENCODING"_"$MIN_CASES" -v dataset=$DATASET_NAME,method=$BUCKET_METHOD,encoding=$CLS_ENCODING,min_cases=$MIN_CASES run.sh
            done
        done
    done
done

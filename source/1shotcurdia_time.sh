#! /bin/csh

cd curdia
#rm shot-time-dia-mic.txt
#@ hosei = $1
@ shotno1 = $2
@ stime = $3
@ etime = $4
#@ shotno2 = $2 
# @ micro = $3

#while ($shotno1 <= $shotno2)
./IpatWp_time.sh $1 $shotno1 $stime $etime 1
#@ shotno1 = $shotno1 + 1
#end

#cp shot-time-dia-mic.txt Wpmaxdata/$1-$2Wpmax.txt

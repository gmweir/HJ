#! /bin/csh

cd curdia
rm shot-time-dia-mic.txt
#@ hosei = $1
@ shotno1 = $2
@ shotno2 = $3 
# @ micro = $3

#echo $1

while ($shotno1 <= $shotno2)
./IpatWp.sh $1 $shotno1 1
@ shotno1 = $shotno1 + 1
end

cp shot-time-dia-mic.txt Wpmaxdata/$2-$3Wpmax.txt


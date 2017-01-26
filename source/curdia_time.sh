#! /bin/csh

   @ hosei = $1
   @ shotno1 = $2
   @ shotno2 = $3
   @ starttime = $4
   @ endtime = $5
   set flag = 1
   set diatime = 305

   set a = curdiamic
   set uni = ms
   set diauni = _$diatime$uni

   cd bin

   while ($shotno1 <= $shotno2)

      ./curb $shotno1 $flag
      ./diamag$diauni $shotno1 $flag
      ./micro.out $hosei $shotno1

      mv microdata MICRO$shotno1.dat

      ./graph.out $shotno1
      ./Ip-Wp_freetime.out $shotno1 $starttime $endtime

      gnuplot gracom

      mv $shotno1$a.txt $shotno1$a$diauni.txt

      rm cur.txt
      rm dia.data
      rm MICRO$shotno1.dat
      rm tmaxmin
      rm tmaxmin_draw1
      rm tmaxmin_draw2
      rm gracom

      if ($flag != 1) then
	 rm $shotno1$a$diauni.txt
      else
	 mv $shotno1$a$diauni.txt ../DATA
      endif

      @ shotno1 = $shotno1 + 1
   end

   mv shot-time-dia-mic.txt ../curdia/Wpmaxdata/$2-$3Wpmax_$4-$5.txt

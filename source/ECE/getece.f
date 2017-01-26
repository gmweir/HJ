CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
	PROGRAM GETECE
C	 			ver2.0
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


  	implicit none
 	character*5 argv2,shotname
	character*16 signalnm
	character*16 modulenm
	character*100  paneldt
	character*21 datascl,dataunit
	character*6  stime
	character*9  sdate
	character*16  snm(20)

        real bes(2**18,16)
	real dataary(2**18+1),tary(2**18+1),data(2**18+1),length
! 	real dataary(8192),tary(8192),data(8192),length
	real tsamp, tdelay, ampgain,sc_max, sc_min, bitzero

	integer ishotno,iswch,iadc_bit,iampfilt
	integer ibhv,ibta,ibtb,ibav,ibiv,i
	integer ich,isample,ierror,ishot(1:5),shot(5)
	integer nm,tmp,baseno,type,type1,type2,type3
	integer j,isw,nshot




        snm(1) ='ECE1FAST'
        snm(2) ='ECE2FAST'
        snm(3) ='ECE3FAST'
        snm(4) ='ECE4FAST'
        snm(5) ='ECE5FAST'
        snm(6) ='ECE6FAST'
        snm(7) ='ECE7FAST'
        snm(8) ='ECE8FAST'
        snm(9) ='ECE9FAST'
        snm(10) ='ECE10FAST'
        snm(11) ='ECE11FAST'
        snm(12) ='ECE12FAST'
        snm(13) ='ECE13FAST'
        snm(14) ='ECE14FAST'
        snm(15) ='ECE15FAST'
        snm(16) ='ECE16FAST'

      	baseno=1
	length=1.0D0
	iswch = 1


	write(*,*)'shot number -R?'         
	read(*,*)ishotno
!	write(*,*)ishotno

	write(*,*)'(0)1 shot or (1)Auto ?'
	read(*,*)isw
        if(isw.eq.1)then
         write(*,*)'number of shots ?'
  	 read (*,*)nshot
  	 write (*,*)ishotno, "--", ishotno+nshot
        end if

        do ishotno=ishotno,ishotno+nshot
        open(22)
	write(22,*)ishotno
        close(22)
        open(22)
 	read(22,*)shotname
 	write(*,*)shotname
        close(22)



        
        do j  = 1,16

         signalnm=snm(j)

	call getdata(signalnm, modulenm, paneldt, datascl, dataunit,
     &   sdate, stime,
     &  dataary, tary, tsamp, tdelay, ampgain, 
     &  sc_max, sc_min, bitzero,
     &  iadc_bit, iampfilt, ishotno, ibhv, ibta, ibtb, ibav, ibiv,
     &  ich, isample, iswch, ierror)

        do  i=1,isample,1
	    bes(i,j)= dataary(i)
        END DO 


       END DO

       write(*,*)"isample:", isample



        open(18)
           do i=1,isample
!            write(18,*)dataary(i)
           end do
        close(18)



        open(unit = 1, file = 'ECEFAST.'//shotname//'')
	do  i=1,isample,1
	      write(1,'(19f15.7)') tary(i), (bes(i,j), j=1, 16)
        END DO
	close(unit=1)




        END DO



!	open(unit = 11, file = 'sample.plt')
!         write(11,*)'plot RAW_mag'//shotname//'.dat'
!        close(11)
	

!	CALL SYSTEM('gnuplot sample.plt')



	
	END PROGRAM GETECE



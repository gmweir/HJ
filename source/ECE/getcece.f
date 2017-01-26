CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
	PROGRAM GETCECE
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
	character*16  snm(2)

        real bes(2**18,2)
	real dataary(2**18+1),tary(2**18+1),data(2**18+1),length
! 	real dataary(8192),tary(8192),data(8192),length
	real tsamp, tdelay, ampgain,sc_max, sc_min, bitzero

	integer ishotno,iswch,iadc_bit,iampfilt
	integer ibhv,ibta,ibtb,ibav,ibiv,i
	integer ich,isample,ierror,ishot(1:5),shot(5)
	integer nm,tmp,baseno,type,type1,type2,type3
	integer j,isw,nshot

        snm(1) ='CECE-RF'
        snm(2) ='CECE-IF'

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

        do j  = 1,2

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

        open(unit = 1, file = 'CECE.'//shotname//'')
	do  i=1,isample,1
	      write(1,'(19f15.7)') tary(i), (bes(i,j), j=1, 2)
        END DO
	close(unit=1)

        END DO

!	open(unit = 11, file = 'sample.plt')
!         write(11,*)'plot RAW_cece'//shotname//'.dat'
!        close(11)
	

!	CALL SYSTEM('gnuplot sample.plt')



	
	END PROGRAM GETCECE



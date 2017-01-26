program main
  implicit none
!c
  real dataary(8,262144),timeary(262144), tmpary(262144)
  real timeary1(8000), Ip(2,8000), wp(8000)
  real ne(8000), neraw(8000)
  real tsamp, tdelay, ampgain
  real sc_max, sc_min, bitzero
  real time_start, time_end
!c
  integer i, j, k, l
  integer itime, sshotno, eshotno
  integer ishotno, iswch, iadc_bit, iampfilt, ibhv, ibta, ibav, tr
  integer ibtb, isample, ierror, ibiv, ich
!c
  character*15  signalnm(9)
  character*5   cishotno
  character*16  modulenm
  character*100 paneldt
  character*21  datascl
  character*21  dataunit
  character*6   stime
  character*9   sdate
!
  signalnm(1)='ECHRG500'
  signalnm(2)='GASPUFF#2'
  signalnm(3)='NBIS3I'
  signalnm(4)='NBIS9I'
  signalnm(5)='VISIBLE3'
  signalnm(6)='VISIBLE4'
  signalnm(7)='AXUV1'
  signalnm(8)='HALPHA11.5'
!c
  write(6,*) 'Input start shot number.'
  read(5,*) sshotno
  write(6,*) 'Input end shot number'
  read(5,*) eshotno
  write(6,*) 'Input start time [ms].'
  read(5,*) time_start
  write(6,*) 'Input end time [ms].'
  read(5,*) time_end
!c
  loop:  do ishotno=sshotno, eshotno
     write(cishotno,'(I5)') ishotno
     
     iswch = 1
     
     do i=1,8
        call getdata(signalnm(i), modulenm, paneldt, datascl, dataunit,&
             sdate, stime,tmpary, timeary, tsamp, tdelay, ampgain, &
             sc_max, sc_min, bitzero,iadc_bit, iampfilt, ishotno, ibhv,&
             ibta, ibtb, ibav, ibiv,ich, isample, iswch, ierror)
        do j=1,262144
           dataary(i,j) = tmpary(j)
        enddo
     enddo
!    
     call system('./micro.out 001 '//cishotno)
     call system('./curdia_time.sh 001 '//cishotno//' '//cishotno//' 150 450')

     open(200, file='./para_data/'//cishotno//'_parameters1.txt')
     open(201, file='./para_data/'//cishotno//'_parameters2.txt')
     open(210, file='curdia/DATA/'//cishotno//'curdiamic_305ms.txt')
     open(220, file='./para_data/'//cishotno//'_wp_ne.txt') 
     
! write to parameters1.dat
     do j=1,isample
        if((timeary(j).ge.time_start).and.(timeary(j).le.time_end)) then
           write(200,250) timeary(j),dataary(1,j),dataary(2,j),&
                dataary(3,j),dataary(4,j)
        endif
250     format(5E15.7)
     enddo

! write to parameters2.dat
     do j=1,isample
        if((timeary(j).ge.time_start).and.(timeary(j).le.time_end)) then
           write(201,251) timeary(j),dataary(5,j),&
                dataary(6,j),dataary(7,j),dataary(8,j),&
                dataary(9,j)
        endif
251     format(6E15.7)
     enddo
!

! write to wp_ne.dat
     do k=1, 4157
        read(210,*) timeary1(k), Ip(1,k), Ip(2,k), wp(k),&
             ne(k), neraw(k)
     enddo
!c
     do k=1, 4157
        if((timeary1(k).ge.time_start).and.(timeary1(k).le.time_end)) then
           write(220,260) timeary1(k), Ip(1,k), Ip(2,k), wp(k),&
                ne(k), neraw(k)
        endif
260     format(6E15.7)
     enddo

!c
     write(*,*) 'output to "./para_data/'//cishotno//'_parameters1&2.txt'
     write(*,*) 'output to "./para_data/'//cishotno//'_wp_ne.txt'
     close(200)
     close(201)
     close(210)
     close(220)
!c
  enddo loop
!c
  stop
end program main
